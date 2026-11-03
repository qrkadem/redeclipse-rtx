// Ray traced ambient occlusion and sun shadows with Vulkan ray queries, for the GL renderer. The world's opaque
// triangles go into an acceleration structure whenever the world changes. Each frame, once the G-buffer is drawn, GL
// writes world positions and normals into images shared with Vulkan, a compute pass traces rays, blends them into the
// history of the same surface and filters them over its neighbours, and GL's lighting reads the result: ambient
// occlusion in place of SSAO, and the sun's shadow in place of its shadow map.
#include "engine.h"
#include "vkint.h"

static const uint32_t rttracespv[] =
#include "vkshaders/rttrace.spv.h"
;
static const uint32_t rtfilterspv[] =
#include "vkshaders/rtfilter.spv.h"
;
static const uint32_t rtskinspv[] =
#include "vkshaders/rtskin.spv.h"
;

VARF(IDF_PERSIST, rtao, 0, 1, 1, cleardeferredlightshaders());  // ray traced ambient occlusion, when the card can trace
VARF(IDF_PERSIST, rtsun, 0, 1, 1, cleardeferredlightshaders()); // ray traced sun shadows, when the card can trace
FVAR(IDF_PERSIST, rtaoradius, 1, 12, 512);  // how far occluders count, in world units
VAR(IDF_PERSIST, rtaorays, 1, 2, 16);       // ambient occlusion rays per pixel each frame
FVAR(IDF_PERSIST, rtsunsize, 0, 0.5f, 10);  // the sun's radius in degrees, larger gives softer shadows
VAR(IDF_PERSIST, rthistory, 1, 12, 64);     // most frames a surface's samples are averaged over
VAR(IDF_PERSIST, rtfilter, 0, 2, 4);        // spatial filter radius, in steps of two pixels
VAR(IDF_PERSIST, rtplayers, 0, 1, 1);      // players and other animated models are in the traced scene: their sun shadows and ambient occlusion are traced
VAR(0, dbgrt, 0, 0, 1);

namespace vk
{
    static PFN_vkGetAccelerationStructureBuildSizesKHR GetASBuildSizes = NULL;
    static PFN_vkCreateAccelerationStructureKHR CreateAS = NULL;
    static PFN_vkDestroyAccelerationStructureKHR DestroyAS = NULL;
    static PFN_vkCmdBuildAccelerationStructuresKHR CmdBuildAS = NULL;
    static PFN_vkGetAccelerationStructureDeviceAddressKHR GetASAddress = NULL;
    static VkDeviceSize scratchalign = 256;

    struct rtbuffer
    {
        VkBuffer buf = VK_NULL_HANDLE;
        VkDeviceMemory mem = VK_NULL_HANDLE;
        VkDeviceAddress addr = 0;
        VkDeviceSize size = 0;
        void *mapped = NULL;
    };

    static bool createbuffer(rtbuffer &b, VkDeviceSize size, VkBufferUsageFlags usage, bool host)
    {
        b.size = size;
        VkBufferCreateInfo bci = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        bci.size = size;
        bci.usage = usage|VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        if(vkCreateBuffer(device, &bci, NULL, &b.buf) != VK_SUCCESS) return false;
        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(device, b.buf, &req);
        int memtype = -1;
        if(host)
        {
            // straight into video memory when the CPU can see it, else through system memory
            memtype = findmemorytype(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT|VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            if(memtype < 0) memtype = findmemorytype(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        }
        else memtype = findmemorytype(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if(memtype < 0) return false;
        VkMemoryAllocateFlagsInfo flags = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO };
        flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        VkMemoryAllocateInfo mai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
        mai.pNext = &flags;
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = memtype;
        if(vkAllocateMemory(device, &mai, NULL, &b.mem) != VK_SUCCESS) return false;
        vkBindBufferMemory(device, b.buf, b.mem, 0);
        if(host && vkMapMemory(device, b.mem, 0, VK_WHOLE_SIZE, 0, &b.mapped) != VK_SUCCESS) return false;
        VkBufferDeviceAddressInfo ai = { VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO };
        ai.buffer = b.buf;
        b.addr = vkGetBufferDeviceAddress(device, &ai);
        return true;
    }

    static void destroybuffer(rtbuffer &b)
    {
        if(b.mapped) vkUnmapMemory(device, b.mem);
        if(b.buf) vkDestroyBuffer(device, b.buf, NULL);
        if(b.mem) vkFreeMemory(device, b.mem, NULL);
        b = rtbuffer();
    }

    struct accelstruct
    {
        VkAccelerationStructureKHR as = VK_NULL_HANDLE;
        rtbuffer buf;
        VkDeviceAddress addr = 0;
    };

    static void destroyaccel(accelstruct &a)
    {
        if(a.as) DestroyAS(device, a.as, NULL);
        destroybuffer(a.buf);
        a = accelstruct();
    }

    static accelstruct worldblas, tlas;
    static rtbuffer worldverts, instances, tlasinstances[2], tlasscratch;
    static int numstaticinsts = 0, tlascapacity = 0;
    #define MAXRTSKINNED 256 // animated model parts the traced scene takes in a frame

    // animated models: each mesh group's vertices with their bones and weights, uploaded once, and each entity's
    // parts skinned again every frame into their own bottom level structure
    struct rtskelgroup
    {
        const void *key;
        rtbuffer verts, indices;
        int numverts, numtris;
    };
    static vector<rtskelgroup *> rtskelgroups;
    struct rtskinned
    {
        const void *owner, *part;
        rtskelgroup *group;
        rtbuffer positions, scratch;
        VkAccelerationStructureKHR as;
        rtbuffer asbuf;
        VkDeviceAddress asaddr;
        int lastframe, lastbuild;
        bool built;
    };
    static vector<rtskinned *> rtskins;
    struct rtcapture
    {
        const void *group, *owner, *part;
        int bonestart;
        matrix4 world;
    };
    // GPU timestamps of each frame's ray tracing: start, scene built, traced
    static VkQueryPool rtqueries = VK_NULL_HANDLE;
    static bool rtqueried[2] = { false, false };
    static float rtscenems = 0, rttracems = 0;
    static vector<rtcapture> rtcaptures;
    static vector<dualquat> rtcapturebones;
    static rtbuffer rtbonebufs[2];
    static bool rtcapturewindow = false, rtcapturepaused = false;
    static int rtskinframe = 0, rtskinnedcount = 0;
    static VkPipelineLayout skinlayout = VK_NULL_HANDLE;
    static VkPipeline skinpipe = VK_NULL_HANDLE;
    struct skinparams
    {
        VkDeviceAddress verts, bones, positions;
        uint numverts, pad;
    };
    static bool worlddirty = true, haveworld = false, rtloaded = false, rtfailed = false;
    static int worldtris = 0, modeltris = 0, modelinstances = 0;

    // a map model's triangles, from its collision meshes, built once and placed by every entity using it
    struct rtmodel
    {
        model *m;
        accelstruct blas;
        rtbuffer verts;
        int tris;
    };
    static vector<rtmodel *> rtmodels;

    // per frame images: GL draws positions and normals, Vulkan writes the result, history stays on Vulkan's side
    static sharedimage rtpos, rtnormal, rtresult;
    static deviceimage rthist[2];
    static GLuint rtfbo = 0;
    static sharedsemaphore rtready, rtdone;
    static int rtw = 0, rth = 0, rtcur = 0;
    static VkCommandBuffer rtcmds[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    static VkFence rtfences[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    static int rtcmdindex = 0;
    static VkDescriptorSetLayout tracesetlayout = VK_NULL_HANDLE, filtersetlayout = VK_NULL_HANDLE;
    static VkPipelineLayout tracelayout = VK_NULL_HANDLE, filterlayout = VK_NULL_HANDLE;
    static VkPipeline tracepipe = VK_NULL_HANDLE, filterpipe = VK_NULL_HANDLE;
    static VkDescriptorPool descpool = VK_NULL_HANDLE;
    static VkDescriptorSet tracesets[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE }, filtersets[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    static matrix4 rtprevclip;
    static vec rtprevcam(0, 0, 0);
    static bool rtprevvalid = false, rtfresh = false;
    static uint rtframe = 0;

    struct traceparams
    {
        float prevclip[16];
        float campos[4], prevcampos[4], sun[4], params[4];
    };
    struct filterparams
    {
        float campos[4], params[4];
    };

    static bool loadrt()
    {
        if(rtloaded) return true;
        if(rtfailed || !hasrt) return false;
        rtfailed = true;
        GetASBuildSizes = (PFN_vkGetAccelerationStructureBuildSizesKHR)vkGetDeviceProcAddr(device, "vkGetAccelerationStructureBuildSizesKHR");
        CreateAS = (PFN_vkCreateAccelerationStructureKHR)vkGetDeviceProcAddr(device, "vkCreateAccelerationStructureKHR");
        DestroyAS = (PFN_vkDestroyAccelerationStructureKHR)vkGetDeviceProcAddr(device, "vkDestroyAccelerationStructureKHR");
        CmdBuildAS = (PFN_vkCmdBuildAccelerationStructuresKHR)vkGetDeviceProcAddr(device, "vkCmdBuildAccelerationStructuresKHR");
        GetASAddress = (PFN_vkGetAccelerationStructureDeviceAddressKHR)vkGetDeviceProcAddr(device, "vkGetAccelerationStructureDeviceAddressKHR");
        if(!GetASBuildSizes || !CreateAS || !DestroyAS || !CmdBuildAS || !GetASAddress) return false;
        VkPhysicalDeviceAccelerationStructurePropertiesKHR asprops = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR };
        VkPhysicalDeviceProperties2 props = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
        props.pNext = &asprops;
        vkGetPhysicalDeviceProperties2(physdev, &props);
        scratchalign = max(VkDeviceSize(asprops.minAccelerationStructureScratchOffsetAlignment), VkDeviceSize(1));

        // the trace pass: scene, positions, normals, history, accumulation
        VkDescriptorSetLayoutBinding tb[5];
        memset(tb, 0, sizeof(tb));
        loopi(5)
        {
            tb[i].binding = i;
            tb[i].descriptorType = i ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
            tb[i].descriptorCount = 1;
            tb[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo lci = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        lci.bindingCount = 5;
        lci.pBindings = tb;
        if(vkCreateDescriptorSetLayout(device, &lci, NULL, &tracesetlayout) != VK_SUCCESS) return false;
        // the filter pass: accumulation, positions, normals, result
        VkDescriptorSetLayoutBinding fb[4];
        memset(fb, 0, sizeof(fb));
        loopi(4)
        {
            fb[i].binding = i;
            fb[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            fb[i].descriptorCount = 1;
            fb[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        lci.bindingCount = 4;
        lci.pBindings = fb;
        if(vkCreateDescriptorSetLayout(device, &lci, NULL, &filtersetlayout) != VK_SUCCESS) return false;

        VkPushConstantRange range = { VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(traceparams) };
        VkPipelineLayoutCreateInfo plci = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        plci.setLayoutCount = 1;
        plci.pSetLayouts = &tracesetlayout;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges = &range;
        if(vkCreatePipelineLayout(device, &plci, NULL, &tracelayout) != VK_SUCCESS) return false;
        range.size = sizeof(filterparams);
        plci.pSetLayouts = &filtersetlayout;
        if(vkCreatePipelineLayout(device, &plci, NULL, &filterlayout) != VK_SUCCESS) return false;

        const uint32_t *code[2] = { rttracespv, rtfilterspv };
        size_t codesize[2] = { sizeof(rttracespv), sizeof(rtfilterspv) };
        VkPipelineLayout layouts[2] = { tracelayout, filterlayout };
        VkPipeline *pipes[2] = { &tracepipe, &filterpipe };
        loopi(2)
        {
            VkShaderModuleCreateInfo smci = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
            smci.codeSize = codesize[i];
            smci.pCode = code[i];
            VkShaderModule module = VK_NULL_HANDLE;
            if(vkCreateShaderModule(device, &smci, NULL, &module) != VK_SUCCESS) return false;
            VkComputePipelineCreateInfo cpci = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
            cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            cpci.stage.module = module;
            cpci.stage.pName = "main";
            cpci.layout = layouts[i];
            VkResult r = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, NULL, pipes[i]);
            vkDestroyShaderModule(device, module, NULL);
            if(r != VK_SUCCESS) return false;
        }

        // skinning works on buffer addresses alone, so it has no descriptors
        {
            VkPushConstantRange skinrange = { VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(skinparams) };
            VkPipelineLayoutCreateInfo slci = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
            slci.pushConstantRangeCount = 1;
            slci.pPushConstantRanges = &skinrange;
            if(vkCreatePipelineLayout(device, &slci, NULL, &skinlayout) != VK_SUCCESS) return false;
            VkShaderModuleCreateInfo smci = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
            smci.codeSize = sizeof(rtskinspv);
            smci.pCode = rtskinspv;
            VkShaderModule module = VK_NULL_HANDLE;
            if(vkCreateShaderModule(device, &smci, NULL, &module) != VK_SUCCESS) return false;
            VkComputePipelineCreateInfo cpci = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
            cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            cpci.stage.module = module;
            cpci.stage.pName = "main";
            cpci.layout = skinlayout;
            VkResult r = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, NULL, &skinpipe);
            vkDestroyShaderModule(device, module, NULL);
            if(r != VK_SUCCESS) return false;
        }

        VkDescriptorPoolSize sizes[2] = { { VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 2 }, { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 16 } };
        VkDescriptorPoolCreateInfo dpci = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        dpci.maxSets = 4;
        dpci.poolSizeCount = 2;
        dpci.pPoolSizes = sizes;
        if(vkCreateDescriptorPool(device, &dpci, NULL, &descpool) != VK_SUCCESS) return false;
        VkDescriptorSetLayout setlayouts[4] = { tracesetlayout, tracesetlayout, filtersetlayout, filtersetlayout };
        VkDescriptorSet sets[4];
        VkDescriptorSetAllocateInfo dsai = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        dsai.descriptorPool = descpool;
        dsai.descriptorSetCount = 4;
        dsai.pSetLayouts = setlayouts;
        if(vkAllocateDescriptorSets(device, &dsai, sets) != VK_SUCCESS) return false;
        tracesets[0] = sets[0];
        tracesets[1] = sets[1];
        filtersets[0] = sets[2];
        filtersets[1] = sets[3];

        loopi(2)
        {
            VkCommandBufferAllocateInfo cai = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
            cai.commandPool = cmdpool;
            cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            cai.commandBufferCount = 1;
            if(vkAllocateCommandBuffers(device, &cai, &rtcmds[i]) != VK_SUCCESS) return false;
            VkFenceCreateInfo fci = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
            fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
            if(vkCreateFence(device, &fci, NULL, &rtfences[i]) != VK_SUCCESS) return false;
        }
        rtfailed = false;
        rtloaded = true;
        return true;
    }

    // the models' structures stay for the next rebuild, models stay loaded for the whole game
    static void destroymodels()
    {
        loopv(rtmodels)
        {
            destroyaccel(rtmodels[i]->blas);
            destroybuffer(rtmodels[i]->verts);
        }
        rtmodels.deletecontents();
    }

    static void destroyskinned()
    {
        loopv(rtskins)
        {
            rtskinned &s = *rtskins[i];
            if(s.as) DestroyAS(device, s.as, NULL);
            destroybuffer(s.asbuf);
            destroybuffer(s.positions);
            destroybuffer(s.scratch);
        }
        rtskins.deletecontents();
    }

    static void destroyskelgroups()
    {
        loopv(rtskelgroups)
        {
            destroybuffer(rtskelgroups[i]->verts);
            destroybuffer(rtskelgroups[i]->indices);
        }
        rtskelgroups.deletecontents();
    }

    static void destroyworld()
    {
        destroyaccel(tlas);
        destroyaccel(worldblas);
        destroybuffer(worldverts);
        destroybuffer(instances);
        loopi(2) destroybuffer(tlasinstances[i]);
        destroybuffer(tlasscratch);
        numstaticinsts = tlascapacity = 0;
        haveworld = false;
        worldtris = modeltris = modelinstances = 0;
    }

    static VkAccelerationStructureGeometryKHR instancegeom(VkDeviceAddress addr)
    {
        VkAccelerationStructureGeometryKHR igeom = { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR };
        igeom.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
        igeom.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
        igeom.geometry.instances.data.deviceAddress = addr;
        return igeom;
    }

    static void asbarrier(VkCommandBuffer cmd, VkPipelineStageFlags2 srcstage, VkAccessFlags2 srcaccess, VkPipelineStageFlags2 dststage, VkAccessFlags2 dstaccess)
    {
        VkMemoryBarrier2 mb = { VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        mb.srcStageMask = srcstage;
        mb.srcAccessMask = srcaccess;
        mb.dstStageMask = dststage;
        mb.dstAccessMask = dstaccess;
        VkDependencyInfo dep = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        dep.memoryBarrierCount = 1;
        dep.pMemoryBarriers = &mb;
        vkCmdPipelineBarrier2(cmd, &dep);
    }

    // the top level structure, with room for the animated models it takes again every frame
    static bool createtlas(int capacity)
    {
        tlascapacity = capacity;
        loopi(2) if(!createbuffer(tlasinstances[i], capacity*sizeof(VkAccelerationStructureInstanceKHR), VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR, true)) return false;
        VkAccelerationStructureGeometryKHR igeom = instancegeom(tlasinstances[0].addr);
        VkAccelerationStructureBuildGeometryInfoKHR bi = { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR };
        bi.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
        bi.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
        bi.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        bi.geometryCount = 1;
        bi.pGeometries = &igeom;
        uint32_t prims = capacity;
        VkAccelerationStructureBuildSizesInfoKHR sizes = { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR };
        GetASBuildSizes(device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &bi, &prims, &sizes);
        if(!createbuffer(tlas.buf, sizes.accelerationStructureSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR, false)) return false;
        VkAccelerationStructureCreateInfoKHR aci = { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR };
        aci.buffer = tlas.buf.buf;
        aci.size = sizes.accelerationStructureSize;
        aci.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
        if(CreateAS(device, &aci, NULL, &tlas.as) != VK_SUCCESS) return false;
        return createbuffer(tlasscratch, sizes.buildScratchSize + scratchalign, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false);
    }

    static void buildtlas(VkCommandBuffer cmd, int slot, int count)
    {
        VkAccelerationStructureGeometryKHR igeom = instancegeom(tlasinstances[slot].addr);
        VkAccelerationStructureBuildGeometryInfoKHR bi = { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR };
        bi.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
        bi.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
        bi.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        bi.geometryCount = 1;
        bi.pGeometries = &igeom;
        bi.dstAccelerationStructure = tlas.as;
        bi.scratchData.deviceAddress = (tlasscratch.addr + scratchalign - 1) & ~(scratchalign - 1);
        VkAccelerationStructureBuildRangeInfoKHR range = { uint32_t(count), 0, 0, 0 };
        const VkAccelerationStructureBuildRangeInfoKHR *ranges = &range;
        CmdBuildAS(cmd, 1, &bi, &ranges);
        asbarrier(cmd, VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR);
    }

    static bool buildaccel(VkCommandBuffer cmd, accelstruct &a, VkAccelerationStructureTypeKHR type, const VkAccelerationStructureGeometryKHR &geom, uint32_t prims, rtbuffer &scratch)
    {
        VkAccelerationStructureBuildGeometryInfoKHR bi = { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR };
        bi.type = type;
        bi.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
        bi.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        bi.geometryCount = 1;
        bi.pGeometries = &geom;
        VkAccelerationStructureBuildSizesInfoKHR sizes = { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR };
        GetASBuildSizes(device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &bi, &prims, &sizes);
        if(!createbuffer(a.buf, sizes.accelerationStructureSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR, false)) return false;
        VkAccelerationStructureCreateInfoKHR aci = { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR };
        aci.buffer = a.buf.buf;
        aci.size = sizes.accelerationStructureSize;
        aci.type = type;
        if(CreateAS(device, &aci, NULL, &a.as) != VK_SUCCESS) return false;
        VkAccelerationStructureDeviceAddressInfoKHR dai = { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR };
        dai.accelerationStructure = a.as;
        a.addr = GetASAddress(device, &dai);
        if(!createbuffer(scratch, sizes.buildScratchSize + scratchalign, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false)) return false;
        bi.dstAccelerationStructure = a.as;
        bi.scratchData.deviceAddress = (scratch.addr + scratchalign - 1) & ~(scratchalign - 1);
        VkAccelerationStructureBuildRangeInfoKHR range = { prims, 0, 0, 0 };
        const VkAccelerationStructureBuildRangeInfoKHR *ranges = &range;
        CmdBuildAS(cmd, 1, &bi, &ranges);
        VkMemoryBarrier2 mb = { VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        mb.srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
        mb.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        mb.dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR|VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        mb.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        VkDependencyInfo dep = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        dep.memoryBarrierCount = 1;
        dep.pMemoryBarriers = &mb;
        vkCmdPipelineBarrier2(cmd, &dep);
        return true;
    }

    static bool buildtriangles(VkCommandBuffer cmd, accelstruct &blas, rtbuffer &vbuf, const vector<vec> &verts, rtbuffer &scratch)
    {
        if(!createbuffer(vbuf, verts.length()*sizeof(vec), VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR, true)) return false;
        memcpy(vbuf.mapped, verts.getbuf(), verts.length()*sizeof(vec));
        VkAccelerationStructureGeometryKHR geom = { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR };
        geom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
        geom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
        geom.geometry.triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
        geom.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
        geom.geometry.triangles.vertexData.deviceAddress = vbuf.addr;
        geom.geometry.triangles.vertexStride = sizeof(vec);
        geom.geometry.triangles.maxVertex = verts.length()-1;
        geom.geometry.triangles.indexType = VK_INDEX_TYPE_NONE_KHR;
        return buildaccel(cmd, blas, VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, geom, verts.length()/3, scratch);
    }

    static rtmodel *getrtmodel(model *m, VkCommandBuffer cmd, vector<rtbuffer> &scratch)
    {
        loopv(rtmodels) if(rtmodels[i]->m == m) return rtmodels[i]->blas.as ? rtmodels[i] : NULL;
        rtmodel *rm = new rtmodel;
        rm->m = m;
        rm->tris = 0;
        rtmodels.add(rm);
        BIH *bih = m->bih ? m->bih : m->setBIH();
        if(!bih) return NULL;
        vector<vec> verts;
        loopi(bih->nummeshes)
        {
            // what is drawn and solid; leaves and other alpha tested parts would block too much
            const BIH::mesh &mesh = bih->meshes[i];
            if(!(mesh.flags&BIH::MESH_RENDER) || mesh.flags&BIH::MESH_ALPHA) continue;
            loopj(mesh.numtris) loopk(3) verts.add(mesh.xform.transform(mesh.getpos(mesh.tris[j].vert[k])));
        }
        if(verts.length() < 3 || !buildtriangles(cmd, rm->blas, rm->verts, verts, scratch.add())) return NULL;
        rm->tris = verts.length()/3;
        return rm;
    }

    // where a map model entity puts its model, as mmintersect works it out the other way
    static void mapmodeltransform(const extentity &e, VkTransformMatrixKHR &t)
    {
        float size = e.attrs[5] ? e.attrs[5]/100.0f : 1.0f;
        int yaw = e.attrs[1], pitch = e.attrs[2], roll = e.attrs[3];
        vec axes[3] = { vec(1, 0, 0), vec(0, 1, 0), vec(0, 0, 1) };
        loopi(3)
        {
            vec &v = axes[i];
            if(roll) v.rotate_around_y(sincosmod360(-roll));
            if(pitch) v.rotate_around_x(sincosmod360(pitch));
            if(yaw) v.rotate_around_z(sincosmod360(yaw));
            v.mul(size);
        }
        loopj(3)
        {
            t.matrix[j][0] = axes[0][j];
            t.matrix[j][1] = axes[1][j];
            t.matrix[j][2] = axes[2][j];
            t.matrix[j][3] = e.o[j];
        }
    }

    static int lastbuild = 0;

    // the world's opaque triangles, as the vertex arrays hold them, and the map models
    static bool buildworld()
    {
        worlddirty = false;
        lastbuild = totalmillis;
        queueidle();
        destroyworld();

        vector<vec> verts;
        loopv(valist)
        {
            vtxarray *va = valist[i];
            const ushort *idx = va->edata + va->eoffset;
            loopj(va->texs + va->blends)
            {
                const elementset &e = va->texelems[j];
                loopk(e.length) verts.add(va->vdata[idx[k]].pos);
                idx += e.length;
            }
        }
        if(verts.length() < 3) return false;
        uint32_t tris = verts.length()/3;

        VkCommandBuffer cmd = begincommands();
        if(!cmd) return false;
        vector<rtbuffer> scratch;
        bool ok = buildtriangles(cmd, worldblas, worldverts, verts, scratch.add());

        vector<VkAccelerationStructureInstanceKHR> insts;
        if(ok)
        {
            VkAccelerationStructureInstanceKHR &inst = insts.add();
            memset(&inst, 0, sizeof(inst));
            inst.transform.matrix[0][0] = inst.transform.matrix[1][1] = inst.transform.matrix[2][2] = 1;
            inst.mask = 0xFF;
            inst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
            inst.accelerationStructureReference = worldblas.addr;

            const vector<extentity *> &ents = entities::getents();
            loopv(ents)
            {
                const extentity &e = *ents[i];
                if(e.type != ET_MAPMODEL || e.flags&(EF_NOSHADOW|EF_NOVIS)) continue;
                model *m = loadmapmodel(e.attrs[0]);
                if(!m) continue;
                rtmodel *rm = getrtmodel(m, cmd, scratch);
                if(!rm) continue;
                VkAccelerationStructureInstanceKHR &mi = insts.add();
                memset(&mi, 0, sizeof(mi));
                mapmodeltransform(e, mi.transform);
                mi.mask = 0x02; // ambient occlusion only: the sun's shadow map keeps map models, alpha tested leaves and all
                mi.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
                mi.accelerationStructureReference = rm->blas.addr;
                modeltris += rm->tris;
                modelinstances++;
            }
        }

        if(ok) ok = createtlas(insts.length() + MAXRTSKINNED);
        if(ok)
        {
            // the world and map models stay at the start of both copies, animated models go after them each frame
            loopi(2) memcpy(tlasinstances[i].mapped, insts.getbuf(), insts.length()*sizeof(VkAccelerationStructureInstanceKHR));
            numstaticinsts = insts.length();
            asbarrier(cmd, VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR);
            buildtlas(cmd, 0, numstaticinsts);
        }
        ok = endcommands(cmd) && ok;
        loopv(scratch) destroybuffer(scratch[i]);
        if(!ok)
        {
            conoutf(colourred, "Ray tracing: could not build the world's acceleration structure");
            destroyworld();
            return false;
        }
        worldtris = tris;
        haveworld = true;
        if(dbgrt) conoutf(colourwhite, "Ray tracing: world of %d triangles, %d map models of %d triangles", worldtris, modelinstances, modeltris);
        return true;
    }

    static void destroyrtimages()
    {
        if(!rtw) return;
        glFinish();
        queueidle();
        if(rtfbo) glDeleteFramebuffers_(1, &rtfbo);
        rtfbo = 0;
        destroysharedimage(rtpos);
        destroysharedimage(rtnormal);
        destroysharedimage(rtresult);
        loopi(2) destroydeviceimage(rthist[i]);
        destroysharedsemaphore(rtready);
        destroysharedsemaphore(rtdone);
        rtw = rth = 0;
        rtprevvalid = false;
    }

    static bool creatertimages(int w, int h)
    {
        rtw = w;
        rth = h;
        if(!createsharedimage(rtpos, w, h, VK_FORMAT_R32G32B32A32_SFLOAT, GL_RGBA32F) ||
           !createsharedimage(rtnormal, w, h, VK_FORMAT_R16G16B16A16_SFLOAT, GL_RGBA16F) ||
           !createsharedimage(rtresult, w, h, VK_FORMAT_R16G16_SFLOAT, GL_RG16F, GL_TEXTURE_RECTANGLE))
            return false;
        loopi(2) if(!createdeviceimage(rthist[i], w, h, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT|VK_IMAGE_USAGE_TRANSFER_DST_BIT)) return false;
        if(!createsharedsemaphore(rtready) || !createsharedsemaphore(rtdone)) return false;

        glGenFramebuffers_(1, &rtfbo);
        glBindFramebuffer_(GL_FRAMEBUFFER, rtfbo);
        glFramebufferTexture2D_(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, rtpos.gltex, 0);
        glFramebufferTexture2D_(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, rtnormal.gltex, 0);
        bool complete = glCheckFramebufferStatus_(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
        glBindFramebuffer_(GL_FRAMEBUFFER, renderfbo);
        if(!complete) return false;
        // GL samples the result linearly, it is the same size as the G-buffer
        glBindTexture(GL_TEXTURE_RECTANGLE, rtresult.gltex);
        glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glBindTexture(GL_TEXTURE_RECTANGLE, 0);

        // the history starts empty, the shared images go to GL
        VkCommandBuffer cmd = begincommands();
        if(!cmd) return false;
        loopi(2)
        {
            imagebarrier(cmd, rthist[i].image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE, VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
            VkClearColorValue zero = {{ 0, 0, 0, 0 }};
            VkImageSubresourceRange range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            vkCmdClearColorImage(cmd, rthist[i].image, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &range);
            imagebarrier(cmd, rthist[i].image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT|VK_ACCESS_2_SHADER_WRITE_BIT);
        }
        // the result starts as nothing occluded
        imagebarrier(cmd, rtresult.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE, VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
        VkClearColorValue one = {{ 1, 1, 1, 1 }};
        VkImageSubresourceRange fullrange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        vkCmdClearColorImage(cmd, rtresult.image, VK_IMAGE_LAYOUT_GENERAL, &one, 1, &fullrange);
        imagebarrier(cmd, rtresult.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE, queuefamily, VK_QUEUE_FAMILY_EXTERNAL);
        sharedimage *shared[2] = { &rtpos, &rtnormal };
        loopi(2) imagebarrier(cmd, shared[i]->image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE, VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE, queuefamily, VK_QUEUE_FAMILY_EXTERNAL);
        if(!endcommands(cmd)) return false;

        // descriptor sets for both history parities: write rthist[i], read the other
        loopi(2)
        {
            VkWriteDescriptorSetAccelerationStructureKHR aswrite = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR };
            aswrite.accelerationStructureCount = 1;
            aswrite.pAccelerationStructures = &tlas.as;
            VkDescriptorImageInfo imgs[8];
            VkImageView views[8] = { rtpos.view, rtnormal.view, rthist[1-i].view, rthist[i].view, rthist[i].view, rtpos.view, rtnormal.view, rtresult.view };
            loopj(8) { imgs[j].sampler = VK_NULL_HANDLE; imgs[j].imageView = views[j]; imgs[j].imageLayout = VK_IMAGE_LAYOUT_GENERAL; }
            VkWriteDescriptorSet writes[9];
            memset(writes, 0, sizeof(writes));
            loopj(9)
            {
                writes[j].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[j].dstSet = j < 5 ? tracesets[i] : filtersets[i];
                writes[j].dstBinding = j < 5 ? j : j-5;
                writes[j].descriptorCount = 1;
                writes[j].descriptorType = j ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
                if(j) writes[j].pImageInfo = &imgs[j-1];
                else writes[j].pNext = &aswrite;
            }
            // the scene changes with the world, so only the images are written here when there is none yet
            vkUpdateDescriptorSets(device, tlas.as ? 9 : 8, tlas.as ? writes : writes+1, 0, NULL);
        }
        rtprevvalid = false;
        return true;
    }

    static rtskelgroup *findskelgroup(const void *key)
    {
        loopv(rtskelgroups) if(rtskelgroups[i]->key == key) return rtskelgroups[i];
        return NULL;
    }

    static VkAccelerationStructureGeometryKHR skingeom(const rtskinned &s)
    {
        VkAccelerationStructureGeometryKHR geom = { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR };
        geom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
        geom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
        geom.geometry.triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
        geom.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
        geom.geometry.triangles.vertexData.deviceAddress = s.positions.addr;
        geom.geometry.triangles.vertexStride = sizeof(vec);
        geom.geometry.triangles.maxVertex = s.group->numverts-1;
        geom.geometry.triangles.indexType = VK_INDEX_TYPE_UINT32;
        geom.geometry.triangles.indexData.deviceAddress = s.group->indices.addr;
        return geom;
    }

    static VkAccelerationStructureBuildGeometryInfoKHR skinbuildinfo(const VkAccelerationStructureGeometryKHR &geom)
    {
        VkAccelerationStructureBuildGeometryInfoKHR bi = { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR };
        bi.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        // refitted each frame as the model moves, built from scratch now and then to keep tracing fast
        bi.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR|VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
        bi.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        bi.geometryCount = 1;
        bi.pGeometries = &geom;
        return bi;
    }

    // an entity's part, with its own skinned copy of the mesh group and structure, kept while it keeps being drawn
    static rtskinned *getskinned(const rtcapture &c, rtskelgroup *g)
    {
        loopv(rtskins)
        {
            rtskinned &s = *rtskins[i];
            if(s.owner == c.owner && s.part == c.part && s.group == g) return s.as ? &s : NULL;
        }
        rtskinned *s = new rtskinned;
        memset((void *)s, 0, sizeof(rtskinned));
        s->owner = c.owner;
        s->part = c.part;
        s->group = g;
        rtskins.add(s);
        if(!createbuffer(s->positions, g->numverts*sizeof(vec), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR, false)) return NULL;
        VkAccelerationStructureGeometryKHR geom = skingeom(*s);
        VkAccelerationStructureBuildGeometryInfoKHR bi = skinbuildinfo(geom);
        uint32_t prims = g->numtris;
        VkAccelerationStructureBuildSizesInfoKHR sizes = { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR };
        GetASBuildSizes(device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &bi, &prims, &sizes);
        if(!createbuffer(s->asbuf, sizes.accelerationStructureSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR, false) ||
           !createbuffer(s->scratch, max(sizes.buildScratchSize, sizes.updateScratchSize) + scratchalign, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false))
            return NULL;
        VkAccelerationStructureCreateInfoKHR aci = { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR };
        aci.buffer = s->asbuf.buf;
        aci.size = sizes.accelerationStructureSize;
        aci.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        if(CreateAS(device, &aci, NULL, &s->as) != VK_SUCCESS) { s->as = VK_NULL_HANDLE; return NULL; }
        VkAccelerationStructureDeviceAddressInfoKHR dai = { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR };
        dai.accelerationStructure = s->as;
        s->asaddr = GetASAddress(device, &dai);
        return s;
    }

    // this frame's animated models: skinned, built, and placed in the top level with the world, which is built again
    static void buildscene(VkCommandBuffer cmd, int slot)
    {
        int ninsts = numstaticinsts;
        rtskinnedcount = 0;
        if(rtcaptures.length() && skinpipe)
        {
            rtbuffer &bones = rtbonebufs[slot];
            VkDeviceSize need = rtcapturebones.length()*sizeof(dualquat);
            if(bones.size < need)
            {
                destroybuffer(bones);
                createbuffer(bones, max(need*2, VkDeviceSize(64*1024)), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true);
            }
            if(bones.mapped)
            {
                memcpy(bones.mapped, rtcapturebones.getbuf(), need);
                VkAccelerationStructureInstanceKHR *insts = (VkAccelerationStructureInstanceKHR *)tlasinstances[slot].mapped;
                vector<rtskinned *> built;
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, skinpipe);
                loopv(rtcaptures)
                {
                    if(ninsts >= tlascapacity) break;
                    const rtcapture &c = rtcaptures[i];
                    rtskelgroup *g = findskelgroup(c.group);
                    if(!g || !g->numverts || !g->numtris) continue;
                    rtskinned *s = getskinned(c, g);
                    if(!s) continue;
                    s->lastframe = rtskinframe;
                    skinparams sp;
                    sp.verts = g->verts.addr;
                    sp.bones = bones.addr + c.bonestart*sizeof(dualquat);
                    sp.positions = s->positions.addr;
                    sp.numverts = g->numverts;
                    sp.pad = 0;
                    vkCmdPushConstants(cmd, skinlayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(sp), &sp);
                    vkCmdDispatch(cmd, (g->numverts+63)/64, 1, 1);
                    built.add(s);
                    VkAccelerationStructureInstanceKHR &inst = insts[ninsts++];
                    memset(&inst, 0, sizeof(inst));
                    loopj(3)
                    {
                        inst.transform.matrix[j][0] = c.world.a[j];
                        inst.transform.matrix[j][1] = c.world.b[j];
                        inst.transform.matrix[j][2] = c.world.c[j];
                        inst.transform.matrix[j][3] = c.world.d[j];
                    }
                    inst.mask = 0x04; // ambient occlusion and the sun
                    inst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
                    inst.accelerationStructureReference = s->asaddr;
                }
                if(built.length())
                {
                    asbarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                        VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, VK_ACCESS_2_SHADER_READ_BIT|VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR);
                    // all of them in one call, so the GPU works on them together
                    vector<VkAccelerationStructureGeometryKHR> geoms;
                    vector<VkAccelerationStructureBuildGeometryInfoKHR> infos;
                    vector<VkAccelerationStructureBuildRangeInfoKHR> ranges;
                    vector<const VkAccelerationStructureBuildRangeInfoKHR *> rangeptrs;
                    loopv(built) geoms.add(skingeom(*built[i]));
                    loopv(built)
                    {
                        rtskinned &s = *built[i];
                        VkAccelerationStructureBuildGeometryInfoKHR &bi = infos.add(skinbuildinfo(geoms[i]));
                        bi.pGeometries = &geoms[i];
                        bool refit = s.built && rtskinframe - s.lastbuild < 120;
                        if(refit)
                        {
                            bi.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
                            bi.srcAccelerationStructure = s.as;
                        }
                        else
                        {
                            s.built = true;
                            s.lastbuild = rtskinframe;
                        }
                        bi.dstAccelerationStructure = s.as;
                        bi.scratchData.deviceAddress = (s.scratch.addr + scratchalign - 1) & ~(scratchalign - 1);
                        VkAccelerationStructureBuildRangeInfoKHR &range = ranges.add();
                        range.primitiveCount = s.group->numtris;
                        range.primitiveOffset = range.firstVertex = range.transformOffset = 0;
                    }
                    loopv(ranges) rangeptrs.add(&ranges[i]);
                    CmdBuildAS(cmd, infos.length(), infos.getbuf(), rangeptrs.getbuf());
                    asbarrier(cmd, VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                        VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR);
                    rtskinnedcount = built.length();
                }
            }
        }
        buildtlas(cmd, slot, ninsts);
        rtcaptures.setsize(0);
        rtcapturebones.setsize(0);
    }

    static void updatescene()
    {
        loopi(2)
        {
            VkWriteDescriptorSetAccelerationStructureKHR aswrite = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR };
            aswrite.accelerationStructureCount = 1;
            aswrite.pAccelerationStructures = &tlas.as;
            VkWriteDescriptorSet write = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
            write.pNext = &aswrite;
            write.dstSet = tracesets[i];
            write.dstBinding = 0;
            write.descriptorCount = 1;
            write.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
            vkUpdateDescriptorSets(device, 1, &write, 0, NULL);
        }
    }
}

using namespace vk;

static string rtdumpname = "";
ICOMMAND(0, rtdump, "s", (char *name), copystring(rtdumpname, *name ? name : "rtdump"));

ICOMMANDV(0, rtavailable, vkactive && hasrt ? 1 : 0);

bool vkrt_aoavailable() { return rtao && vkactive && hasrt && !rtfailed; }
bool vkrt_sunavailable() { return rtsun && vkactive && hasrt && !rtfailed; }
bool vkrt_active() { return rtfresh; }
bool vkrt_sunready() { return vkrt_sunavailable() && haveworld && rtloaded; }
// nothing occluded, for when no result was traced yet
static GLuint rtwhitetex = 0;
GLuint vkrt_texture()
{
    if(rtresult.gltex) return rtresult.gltex;
    if(!rtwhitetex)
    {
        static const ushort white[2] = { 0x3C00, 0x3C00 }; // 1.0 as half floats
        glGenTextures(1, &rtwhitetex);
        glBindTexture(GL_TEXTURE_RECTANGLE, rtwhitetex);
        glTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_RG16F, 1, 1, 0, GL_RG, GL_HALF_FLOAT, white);
        glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    }
    return rtwhitetex;
}

void vkrt_worldchanged() { worlddirty = true; }

matrix4 vkrt_partworld;

bool vkrt_capturing()
{
    return rtcapturewindow && !rtcapturepaused && !drawtex && rtplayers && vkactive && hasrt && !rtfailed && (rtao || rtsun);
}

bool vkrt_playershadows() { return rtplayers && vkrt_sunready(); }

void vkrt_beginframe()
{
    rtcaptures.setsize(0);
    rtcapturebones.setsize(0);
    rtcapturewindow = true;
}

void vkrt_pausecapture(bool pause) { rtcapturepaused = pause; }

bool vkrt_hasskel(const void *group) { return findskelgroup(group) != NULL; }

void vkrt_registerskel(const void *group, const rtskinvert *verts, int numverts, const uint *indices, int numindices)
{
    rtskelgroup *g = new rtskelgroup;
    memset((void *)g, 0, sizeof(rtskelgroup));
    g->key = group;
    rtskelgroups.add(g); // even when it fails, so it isn't tried again every frame
    if(numverts <= 0 || numindices < 3 || numverts > 0xFFFFFF) return;
    if(!createbuffer(g->verts, numverts*sizeof(rtskinvert), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true) ||
       !createbuffer(g->indices, numindices*sizeof(uint), VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR, true))
    {
        destroybuffer(g->verts);
        destroybuffer(g->indices);
        return;
    }
    memcpy(g->verts.mapped, verts, numverts*sizeof(rtskinvert));
    memcpy(g->indices.mapped, indices, numindices*sizeof(uint));
    g->numverts = numverts;
    g->numtris = numindices/3;
}

void vkrt_captureskel(const void *group, const void *owner, const void *part, const dualquat *bones, int numbones)
{
    if(numbones <= 0) return;
    loopv(rtcaptures) if(rtcaptures[i].owner == owner && rtcaptures[i].part == part && rtcaptures[i].group == group) return; // the first pass drawing it this frame
    if(rtcaptures.length() >= MAXRTSKINNED) return;
    rtcapture &c = rtcaptures.add();
    c.group = group;
    c.owner = owner;
    c.part = part;
    c.bonestart = rtcapturebones.length();
    c.world = vkrt_partworld;
    rtcapturebones.put(bones, numbones);
}

void vkrt_cleanupgl()
{
    if(rtwhitetex) glDeleteTextures(1, &rtwhitetex);
    rtwhitetex = 0;
    if(!device) return;
    destroyrtimages();
    rtfresh = false;
}

void vkrt_shutdown()
{
    if(!device) return;
    queueidle();
    destroyworld();
    destroymodels();
    destroyskinned();
    destroyskelgroups();
    loopi(2) destroybuffer(rtbonebufs[i]);
    if(rtqueries) vkDestroyQueryPool(device, rtqueries, NULL);
    rtqueries = VK_NULL_HANDLE;
    rtqueried[0] = rtqueried[1] = false;
    if(skinpipe) vkDestroyPipeline(device, skinpipe, NULL);
    if(skinlayout) vkDestroyPipelineLayout(device, skinlayout, NULL);
    skinpipe = VK_NULL_HANDLE;
    skinlayout = VK_NULL_HANDLE;
    loopi(2)
    {
        if(rtfences[i]) vkDestroyFence(device, rtfences[i], NULL);
        if(rtcmds[i]) vkFreeCommandBuffers(device, cmdpool, 1, &rtcmds[i]);
        rtfences[i] = VK_NULL_HANDLE;
        rtcmds[i] = VK_NULL_HANDLE;
    }
    if(descpool) vkDestroyDescriptorPool(device, descpool, NULL);
    if(tracepipe) vkDestroyPipeline(device, tracepipe, NULL);
    if(filterpipe) vkDestroyPipeline(device, filterpipe, NULL);
    if(tracelayout) vkDestroyPipelineLayout(device, tracelayout, NULL);
    if(filterlayout) vkDestroyPipelineLayout(device, filterlayout, NULL);
    if(tracesetlayout) vkDestroyDescriptorSetLayout(device, tracesetlayout, NULL);
    if(filtersetlayout) vkDestroyDescriptorSetLayout(device, filtersetlayout, NULL);
    descpool = VK_NULL_HANDLE;
    tracepipe = filterpipe = VK_NULL_HANDLE;
    tracelayout = filterlayout = VK_NULL_HANDLE;
    tracesetlayout = filtersetlayout = VK_NULL_HANDLE;
    rtloaded = rtfailed = false;
    worlddirty = true;
}

// between the G-buffer and lighting: GL hands over positions and normals, Vulkan traces, GL waits for the result
void vkrt_trace()
{
    rtfresh = false;
    rtcapturewindow = false; // what was drawn so far this frame is what goes in the scene
    bool doao = vkrt_aoavailable(), dosun = vkrt_sunavailable() && !getpielight().iszero();
    if((!doao && !dosun) || drawtex || !loadrt()) return;
    // editing changes the world piece by piece, so it is rebuilt a few times a second at most
    if(worlddirty && (!editmode || !haveworld || totalmillis - lastbuild >= 250))
    {
        bool had = haveworld;
        buildworld();
        if(haveworld && rtw) updatescene();
        if(!haveworld && had) rtprevvalid = false;
    }
    if(!haveworld) return;
    if(rtw != gw || rth != gh)
    {
        destroyrtimages();
        if(!creatertimages(gw, gh))
        {
            conoutf(colourred, "Ray tracing: could not make the %dx%d images", gw, gh);
            destroyrtimages();
            rtfailed = true;
            return;
        }
    }
    Shader *inputshader = useshaderbyname("rtinputs"), *clearshader = useshaderbyname("rtclearavatar");
    if(!inputshader || !clearshader) return;

    // positions and normals from the G-buffer
    GLint oldfbo = 0, oldviewport[4];
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &oldfbo);
    glGetIntegerv(GL_VIEWPORT, oldviewport);
    glBindFramebuffer_(GL_FRAMEBUFFER, rtfbo);
    bool stencil = gdepthformat && useavatarmask();
    // full screen passes: the scene's depth must stay as it is
    GLboolean depthtest = glIsEnabled(GL_DEPTH_TEST), depthwrite = GL_TRUE;
    glGetBooleanv(GL_DEPTH_WRITEMASK, &depthwrite);
    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);
    glViewport(0, 0, gw, gh);
    static const GLenum drawbufs[2] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
    glDrawBuffers_(2, drawbufs);
    inputshader->set();
    LOCALPARAM(worldmat, viewworld);
    glActiveTexture_(GL_TEXTURE0);
    if(msaalight) glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, msdepthtex);
    else glBindTexture(GL_TEXTURE_RECTANGLE, gdepthtex);
    glActiveTexture_(GL_TEXTURE1);
    if(msaalight) glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, msnormaltex);
    else glBindTexture(GL_TEXTURE_RECTANGLE, gnormaltex);
    glActiveTexture_(GL_TEXTURE0);
    screenquad(gw, gh);
    glstencilstate oldstencil;
    if(stencil)
    {
        oldstencil.save();
        bindgdepth(); // only for its stencil, which marks the first person model
    }
    if(stencil)
    {
        // the first person model is drawn with its own projection, it stays out of the traced surfaces
        glDrawBuffers_(1, drawbufs);
        glEnable(GL_STENCIL_TEST);
        glStencilFunc(GL_EQUAL, 0x40, 0x40);
        glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);
        clearshader->set();
        screenquad(gw, gh);
        oldstencil.restore();
        glFramebufferRenderbuffer_(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, 0);
        glFramebufferRenderbuffer_(GL_FRAMEBUFFER, GL_STENCIL_ATTACHMENT, GL_RENDERBUFFER, 0);
    }
    if(depthtest) glEnable(GL_DEPTH_TEST);
    glDepthMask(depthwrite);
    glDrawBuffers_(1, drawbufs);
    glBindFramebuffer_(GL_FRAMEBUFFER, oldfbo);
    glViewport(oldviewport[0], oldviewport[1], oldviewport[2], oldviewport[3]);

    GLuint texs[3] = { rtpos.gltex, rtnormal.gltex, rtresult.gltex };
    GLenum layouts[3] = { GL_LAYOUT_GENERAL_EXT, GL_LAYOUT_GENERAL_EXT, GL_LAYOUT_GENERAL_EXT };
    glsignal(rtready, 3, texs, layouts);
    glFlush();

    matrix4 vp;
    vp.muld(viewproj, viewcam);
    if(!rtprevvalid || rtprevcam.dist(viewpos) > 64)
    {
        rtprevclip = vp;
        rtprevcam = viewpos;
    }

    int slot = rtcmdindex;
    VkCommandBuffer cmd = rtcmds[rtcmdindex];
    VkFence fence = rtfences[rtcmdindex];
    rtcmdindex = (rtcmdindex+1)%2;
    vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
    if(!rtqueries)
    {
        VkQueryPoolCreateInfo qpci = { VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
        qpci.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qpci.queryCount = 6;
        vkCreateQueryPool(device, &qpci, NULL, &rtqueries);
    }
    if(rtqueries && rtqueried[slot])
    {
        uint64_t stamps[3];
        if(vkGetQueryPoolResults(device, rtqueries, slot*3, 3, sizeof(stamps), stamps, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT) == VK_SUCCESS)
        {
            float tick = devprops.limits.timestampPeriod*1e-6f;
            rtscenems += ((stamps[1]-stamps[0])*tick - rtscenems)*0.05f;
            rttracems += ((stamps[2]-stamps[1])*tick - rttracems)*0.05f;
        }
    }
    // parts not drawn for a while go, once in a while, as that waits for the queue
    if(++rtskinframe%300 == 0)
    {
        bool stale = false;
        loopv(rtskins) if(rtskins[i]->lastframe < rtskinframe - 300) stale = true;
        if(stale)
        {
            queueidle();
            loopvrev(rtskins) if(rtskins[i]->lastframe < rtskinframe - 300)
            {
                rtskinned *s = rtskins.removeunordered(i);
                if(s->as) DestroyAS(device, s->as, NULL);
                destroybuffer(s->asbuf);
                destroybuffer(s->positions);
                destroybuffer(s->scratch);
                delete s;
            }
        }
    }
    vkResetFences(device, 1, &fence);
    vkResetCommandBuffer(cmd, 0);
    VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);
    if(rtqueries)
    {
        vkCmdResetQueryPool(cmd, rtqueries, slot*3, 3);
        vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, rtqueries, slot*3);
    }
    buildscene(cmd, slot);
    if(rtqueries) vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, rtqueries, slot*3+1);
    sharedimage *shared[3] = { &rtpos, &rtnormal, &rtresult };
    loopi(3) imagebarrier(cmd, shared[i]->image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT|VK_ACCESS_2_SHADER_WRITE_BIT,
        VK_QUEUE_FAMILY_EXTERNAL, queuefamily);

    traceparams tp;
    memcpy(tp.prevclip, rtprevclip.a.v, sizeof(tp.prevclip));
    tp.campos[0] = viewpos.x; tp.campos[1] = viewpos.y; tp.campos[2] = viewpos.z; tp.campos[3] = float(rtframe++ & 0xFFFF);
    tp.prevcampos[0] = rtprevcam.x; tp.prevcampos[1] = rtprevcam.y; tp.prevcampos[2] = rtprevcam.z; tp.prevcampos[3] = 0;
    const vec &sundir = getpielightdir();
    tp.sun[0] = sundir.x; tp.sun[1] = sundir.y; tp.sun[2] = sundir.z; tp.sun[3] = dosun ? tanf(rtsunsize*RAD) : -1.0f;
    tp.params[0] = rtaoradius;
    tp.params[1] = rthistory;
    tp.params[2] = rtaorays;
    tp.params[3] = (doao ? 1 : 0) | (dosun ? 2 : 0);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, tracepipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, tracelayout, 0, 1, &tracesets[rtcur], 0, NULL);
    vkCmdPushConstants(cmd, tracelayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(tp), &tp);
    vkCmdDispatch(cmd, (gw+7)/8, (gh+7)/8, 1);
    imagebarrier(cmd, rthist[rtcur].image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT|VK_ACCESS_2_SHADER_WRITE_BIT);

    filterparams fp;
    fp.campos[0] = viewpos.x; fp.campos[1] = viewpos.y; fp.campos[2] = viewpos.z; fp.campos[3] = 0;
    fp.params[0] = rtfilter; fp.params[1] = fp.params[2] = fp.params[3] = 0;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, filterpipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, filterlayout, 0, 1, &filtersets[rtcur], 0, NULL);
    vkCmdPushConstants(cmd, filterlayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(fp), &fp);
    vkCmdDispatch(cmd, (gw+7)/8, (gh+7)/8, 1);

    if(rtqueries)
    {
        vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, rtqueries, slot*3+2);
        rtqueried[slot] = true;
    }
    loopi(3) imagebarrier(cmd, shared[i]->image, VK_IMAGE_LAYOUT_GENERAL, i == 2 ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT|VK_ACCESS_2_SHADER_READ_BIT, VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
        queuefamily, VK_QUEUE_FAMILY_EXTERNAL);
    vkEndCommandBuffer(cmd);

    VkSemaphoreSubmitInfo wait = { VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO };
    wait.semaphore = rtready.sem;
    wait.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    VkSemaphoreSubmitInfo signal = { VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO };
    signal.semaphore = rtdone.sem;
    signal.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    VkCommandBufferSubmitInfo cbi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
    cbi.commandBuffer = cmd;
    VkSubmitInfo2 si = { VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
    si.waitSemaphoreInfoCount = 1;
    si.pWaitSemaphoreInfos = &wait;
    si.commandBufferInfoCount = 1;
    si.pCommandBufferInfos = &cbi;
    si.signalSemaphoreInfoCount = 1;
    si.pSignalSemaphoreInfos = &signal;
    if(submit(si, fence, currentframeid()) != VK_SUCCESS)
    {
        conoutf(colourred, "Ray tracing: submitting failed, turning it off");
        rtfailed = true;
        return;
    }
    GLenum backlayouts[3] = { GL_LAYOUT_GENERAL_EXT, GL_LAYOUT_GENERAL_EXT, GL_LAYOUT_SHADER_READ_ONLY_EXT };
    glwait(rtdone, 3, texs, backlayouts);

    if(rtdumpname[0])
    {
        // development aid: the traced terms and which pixels had a surface, as images
        float *result = new float[gw*gh*2], *pos = new float[gw*gh*4];
        glBindTexture(GL_TEXTURE_RECTANGLE, rtresult.gltex);
        glGetTexImage(GL_TEXTURE_RECTANGLE, 0, GL_RG, GL_FLOAT, result);
        glBindTexture(GL_TEXTURE_2D, rtpos.gltex);
        glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, pos);
        glBindTexture(GL_TEXTURE_2D, 0);
        ImageData ao(gw, gh, 3);
        loopi(gw*gh)
        {
            ao.data[i*3+0] = uchar(clamp(result[i*2], 0.0f, 1.0f)*255);
            ao.data[i*3+1] = uchar(clamp(result[i*2+1], 0.0f, 1.0f)*255);
            ao.data[i*3+2] = pos[i*4+3] > 0 ? 255 : 0;
        }
        defformatstring(name, "%s.png", rtdumpname);
        saveimage(name, ao, IFMT_PNG, 9, true);
        conoutf(colourwhite, "Ray tracing: saved %s (red ao, green sun, blue surface)", name);
        delete[] result;
        delete[] pos;
        rtdumpname[0] = '\0';
    }

    rtprevclip = vp;
    rtprevcam = viewpos;
    rtprevvalid = true;
    rtcur ^= 1;
    rtfresh = true;
}

ICOMMAND(0, rtinfo, "", (),
{
    if(!vkactive || !hasrt) { conoutf(colourwhite, "Ray tracing: not available"); return; }
    const vec &sundir = getpielightdir();
    conoutf(colourwhite, "Ray tracing: GPU %.2f ms building the scene, %.2f ms tracing and filtering", rtscenems, rttracems);
    conoutf(colourwhite, "Ray tracing: world %d triangles, %d map models of %d triangles, %d animated parts this frame, images %dx%d, ao %s, sun %s (towards %.2f %.2f %.2f%s)%s", worldtris, modelinstances, modeltris, rtskinnedcount, rtw, rth,
        vkrt_aoavailable() ? "on" : "off", vkrt_sunavailable() ? "on" : "off", sundir.x, sundir.y, sundir.z, getpielight().iszero() ? ", no sunlight" : "", rtfailed ? ", failed" : "");
});
