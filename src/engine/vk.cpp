// Vulkan presenting, see vk.h. The GL renderer is unchanged: it draws into screen images that Vulkan allocated
// (GL_EXT_memory_object_fd), GL and Vulkan hand each image back and forth with shared semaphores
// (GL_EXT_semaphore_fd), and a presenting thread copies finished images into the swapchain and presents them.
// GL binding framebuffer 0 lands on the current screen image, so code drawing "to the screen" needs no change.
// With DLSS frame generation on, GL also fills motion vectors, depth and a copy of the frame without the HUD, and
// the presenting thread shows the generated frames between the rendered ones.
#include "engine.h"
#include "vkint.h"
#include "vk.h"
#include <SDL_vulkan.h>

static const uint32_t hdrencodespv[] =
#include "vkshaders/hdrencode.spv.h"
;

VARF(IDF_INIT, vulkan, 0, 1, 1, initwarning("Vulkan presenting", INIT_RESET));
VAR(IDF_INIT, vkvalidation, 0, 0, 1); // Khronos validation layer, for development
VAR(0, dbgvkhdrfail, 0, 0, 2); // development aid: 1 makes the display refuse HDR swapchains, 2 stop taking HDR frames
VAR(0, dbgvk, 0, 0, 1);

bool vkactive = false;
extern int dlssg, refreshrate, lowlatency;
extern float hdrgamma;
extern void cleanupscale();
VAR(0, gvelocity, 1, 0, 0); // the G-buffer has a velocity target that models write their motion into, see model.cfg

// HDR output: HDR10 (PQ, Rec.2020) to a display that takes it. The SDR picture keeps its shadows and midtones at
// paper white and its highlights reach up to the peak, see vkshaders/hdrencode.comp
static SDL_atomic_t hdrmetadirty; // the display hears of a new peak at the next present
VAR(IDF_PERSIST, hdroutput, 0, 0, 1);
FVARF(IDF_PERSIST, hdrpeak, 100, 1000, 10000, SDL_AtomicSet(&hdrmetadirty, 1));       // the display's brightest, in nits
FVARF(IDF_PERSIST, hdrpaperwhite, 40, 200, 1000, SDL_AtomicSet(&hdrmetadirty, 1));    // what SDR white becomes, in nits
// how bright light must be to reach the peak, in the tonemap's terms (it shows light k as 1 - 2^-k, so 1 is half
// white): lower makes more of the picture's highlights reach it, 15 is the most the frame's 16 bits hold
FVAR(IDF_PERSIST, hdrrange, 6, 12, 15);
bool hdrscreen = false; // HDR output is on: the tonemapped frame goes to the screen in 16 bits

#define MAXVKFRAMES 2 // screen images GL and Vulkan take turns with

namespace vk
{
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physdev = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    uint32_t queuefamily = 0;
    VkQueue queue = VK_NULL_HANDLE;
    VkCommandPool cmdpool = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties devprops;
    bool hasrt = false;

    static VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
    static PFN_vkSetDebugUtilsObjectNameEXT SetDebugUtilsObjectNameEXT = NULL;

    // with the validation layer: a name its messages give the object by
    void nameobject(VkObjectType type, uint64_t handle, const char *name)
    {
        if(!SetDebugUtilsObjectNameEXT || !device || !handle) return;
        VkDebugUtilsObjectNameInfoEXT ni = { VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT };
        ni.objectType = type;
        ni.objectHandle = handle;
        ni.pObjectName = name;
        SetDebugUtilsObjectNameEXT(device, &ni);
    }
    static VkSurfaceKHR surface = VK_NULL_HANDLE;
    static VkPhysicalDeviceMemoryProperties memprops;
    static SDL_Window *glwindow = NULL;

    static PFN_vkGetMemoryFdKHR GetMemoryFdKHR = NULL;
    static PFN_vkGetSemaphoreFdKHR GetSemaphoreFdKHR = NULL;

    static PFNGLCREATEMEMORYOBJECTSEXTPROC glCreateMemoryObjectsEXT_ = NULL;
    static PFNGLDELETEMEMORYOBJECTSEXTPROC glDeleteMemoryObjectsEXT_ = NULL;
    static PFNGLMEMORYOBJECTPARAMETERIVEXTPROC glMemoryObjectParameterivEXT_ = NULL;
    static PFNGLIMPORTMEMORYFDEXTPROC glImportMemoryFdEXT_ = NULL;
    static PFNGLTEXSTORAGEMEM2DEXTPROC glTexStorageMem2DEXT_ = NULL;
    static PFNGLGENSEMAPHORESEXTPROC glGenSemaphoresEXT_ = NULL;
    static PFNGLDELETESEMAPHORESEXTPROC glDeleteSemaphoresEXT_ = NULL;
    static PFNGLIMPORTSEMAPHOREFDEXTPROC glImportSemaphoreFdEXT_ = NULL;
    static PFNGLSIGNALSEMAPHOREEXTPROC glSignalSemaphoreEXT_ = NULL;
    static PFNGLWAITSEMAPHOREEXTPROC glWaitSemaphoreEXT_ = NULL;
    static PFNGLGETUNSIGNEDBYTEI_VEXTPROC glGetUnsignedBytei_vEXT_ = NULL;
    static PFNGLBINDFRAMEBUFFERPROC glBindFramebufferreal = NULL;

    const char *resultname(VkResult r)
    {
        switch(r)
        {
            case VK_SUCCESS: return "success";
            case VK_NOT_READY: return "not ready";
            case VK_TIMEOUT: return "timeout";
            case VK_SUBOPTIMAL_KHR: return "suboptimal";
            case VK_ERROR_OUT_OF_HOST_MEMORY: return "out of host memory";
            case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "out of device memory";
            case VK_ERROR_INITIALIZATION_FAILED: return "initialization failed";
            case VK_ERROR_DEVICE_LOST: return "device lost";
            case VK_ERROR_LAYER_NOT_PRESENT: return "layer not present";
            case VK_ERROR_EXTENSION_NOT_PRESENT: return "extension not present";
            case VK_ERROR_FEATURE_NOT_PRESENT: return "feature not present";
            case VK_ERROR_INCOMPATIBLE_DRIVER: return "incompatible driver";
            case VK_ERROR_FORMAT_NOT_SUPPORTED: return "format not supported";
            case VK_ERROR_SURFACE_LOST_KHR: return "surface lost";
            case VK_ERROR_NATIVE_WINDOW_IN_USE_KHR: return "window in use";
            case VK_ERROR_OUT_OF_DATE_KHR: return "out of date";
            case VK_ERROR_INVALID_EXTERNAL_HANDLE: return "invalid external handle";
            default: return "error";
        }
    }

    #define VKCHECK(call) do { VkResult vkres_ = (call); if(vkres_ != VK_SUCCESS) { conoutf(colourred, "Vulkan: %s: %s (%d)", #call, resultname(vkres_), int(vkres_)); return false; } } while(0)

    int findmemorytype(uint32_t bits, VkMemoryPropertyFlags flags)
    {
        loopi(int(memprops.memoryTypeCount)) if(bits&(1u<<i) && (memprops.memoryTypes[i].propertyFlags&flags) == flags) return i;
        return -1;
    }

    static void addext(vector<const char *> &exts, const char *name)
    {
        loopv(exts) if(!strcmp(exts[i], name)) return;
        exts.add(name);
    }

    static bool hasext(const vector<VkExtensionProperties> &props, const char *name)
    {
        loopv(props) if(!strcmp(props[i].extensionName, name)) return true;
        return false;
    }

    static VKAPI_ATTR VkBool32 VKAPI_CALL debugcallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity, VkDebugUtilsMessageTypeFlagsEXT, const VkDebugUtilsMessengerCallbackDataEXT *data, void *)
    {
        // the driver can call this from its own threads, where the console isn't safe to use
        if(severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) fprintf(stderr, "Vulkan: %s\n", data->pMessage);
        return VK_FALSE;
    }

    void imagebarrier(VkCommandBuffer cmd, VkImage image, VkImageLayout oldlayout, VkImageLayout newlayout,
        VkPipelineStageFlags2 srcstage, VkAccessFlags2 srcaccess, VkPipelineStageFlags2 dststage, VkAccessFlags2 dstaccess,
        uint32_t srcqueue, uint32_t dstqueue, VkImageAspectFlags aspect)
    {
        VkImageMemoryBarrier2 b = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
        b.srcStageMask = srcstage;
        b.srcAccessMask = srcaccess;
        b.dstStageMask = dststage;
        b.dstAccessMask = dstaccess;
        b.oldLayout = oldlayout;
        b.newLayout = newlayout;
        b.srcQueueFamilyIndex = srcqueue;
        b.dstQueueFamilyIndex = dstqueue;
        b.image = image;
        b.subresourceRange.aspectMask = aspect;
        b.subresourceRange.levelCount = 1;
        b.subresourceRange.layerCount = 1;
        VkDependencyInfo dep = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers = &b;
        vkCmdPipelineBarrier2(cmd, &dep);
    }

    bool createsharedimage(sharedimage &s, int w, int h, VkFormat format, GLenum glformat, GLenum gltarget)
    {
        s.w = w;
        s.h = h;
        s.format = format;
        s.glformat = glformat;
        s.gltarget = gltarget;

        VkFormatProperties fp;
        vkGetPhysicalDeviceFormatProperties(physdev, format, &fp);
        VkImageUsageFlags usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT|VK_IMAGE_USAGE_TRANSFER_DST_BIT|VK_IMAGE_USAGE_SAMPLED_BIT;
        if(fp.optimalTilingFeatures&VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) usage |= VK_IMAGE_USAGE_STORAGE_BIT;
        if(fp.optimalTilingFeatures&VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT) usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

        VkExternalMemoryImageCreateInfo ext = { VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO };
        ext.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        VkImageCreateInfo ici = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
        ici.pNext = &ext;
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = format;
        ici.extent.width = w;
        ici.extent.height = h;
        ici.extent.depth = 1;
        ici.mipLevels = 1;
        ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = usage;
        ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VKCHECK(vkCreateImage(device, &ici, NULL, &s.image));

        VkMemoryRequirements req;
        vkGetImageMemoryRequirements(device, s.image, &req);
        VkMemoryDedicatedAllocateInfo dedicated = { VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO };
        dedicated.image = s.image;
        VkExportMemoryAllocateInfo exportinfo = { VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO };
        exportinfo.pNext = &dedicated;
        exportinfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        VkMemoryAllocateInfo mai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
        mai.pNext = &exportinfo;
        mai.allocationSize = req.size;
        int memtype = findmemorytype(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if(memtype < 0) return false;
        mai.memoryTypeIndex = memtype;
        VKCHECK(vkAllocateMemory(device, &mai, NULL, &s.memory));
        VKCHECK(vkBindImageMemory(device, s.image, s.memory, 0));

        VkImageViewCreateInfo vci = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
        vci.image = s.image;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = format;
        vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vci.subresourceRange.levelCount = 1;
        vci.subresourceRange.layerCount = 1;
        VKCHECK(vkCreateImageView(device, &vci, NULL, &s.view));

        int fd = -1;
        VkMemoryGetFdInfoKHR gfi = { VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR };
        gfi.memory = s.memory;
        gfi.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        VKCHECK(GetMemoryFdKHR(device, &gfi, &fd));

        while(glGetError() != GL_NO_ERROR);
        glCreateMemoryObjectsEXT_(1, &s.glmemory);
        GLint isdedicated = GL_TRUE;
        glMemoryObjectParameterivEXT_(s.glmemory, GL_DEDICATED_MEMORY_OBJECT_EXT, &isdedicated);
        glImportMemoryFdEXT_(s.glmemory, req.size, GL_HANDLE_TYPE_OPAQUE_FD_EXT, fd); // GL owns the fd from here on
        glGenTextures(1, &s.gltex);
        glBindTexture(gltarget, s.gltex);
        glTexParameteri(gltarget, GL_TEXTURE_TILING_EXT, GL_OPTIMAL_TILING_EXT);
        glTexStorageMem2DEXT_(gltarget, 1, glformat, w, h, s.glmemory, 0);
        glTexParameteri(gltarget, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(gltarget, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(gltarget, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(gltarget, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glBindTexture(gltarget, 0);
        GLenum err = glGetError();
        if(err != GL_NO_ERROR)
        {
            conoutf(colourred, "Vulkan: GL could not use a %dx%d shared image (GL error 0x%X)", w, h, err);
            return false;
        }
        return true;
    }

    void destroysharedimage(sharedimage &s)
    {
        if(s.gltex) glDeleteTextures(1, &s.gltex);
        if(s.glmemory) glDeleteMemoryObjectsEXT_(1, &s.glmemory);
        if(s.view) vkDestroyImageView(device, s.view, NULL);
        if(s.image) vkDestroyImage(device, s.image, NULL);
        if(s.memory) vkFreeMemory(device, s.memory, NULL);
        s = sharedimage();
    }

    bool createsharedsemaphore(sharedsemaphore &s)
    {
        VkExportSemaphoreCreateInfo esci = { VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO };
        esci.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
        VkSemaphoreCreateInfo sci = { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
        sci.pNext = &esci;
        VKCHECK(vkCreateSemaphore(device, &sci, NULL, &s.sem));
        int fd = -1;
        VkSemaphoreGetFdInfoKHR gfi = { VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR };
        gfi.semaphore = s.sem;
        gfi.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
        VKCHECK(GetSemaphoreFdKHR(device, &gfi, &fd));
        glGenSemaphoresEXT_(1, &s.glsem);
        glImportSemaphoreFdEXT_(s.glsem, GL_HANDLE_TYPE_OPAQUE_FD_EXT, fd);
        return true;
    }

    void destroysharedsemaphore(sharedsemaphore &s)
    {
        if(s.glsem) glDeleteSemaphoresEXT_(1, &s.glsem);
        if(s.sem) vkDestroySemaphore(device, s.sem, NULL);
        s = sharedsemaphore();
    }

    void glsignal(sharedsemaphore &s, int numtex, const GLuint *texs, const GLenum *layouts)
    {
        glSignalSemaphoreEXT_(s.glsem, 0, NULL, numtex, texs, layouts);
    }

    void glwait(sharedsemaphore &s, int numtex, const GLuint *texs, const GLenum *layouts)
    {
        glWaitSemaphoreEXT_(s.glsem, 0, NULL, numtex, texs, layouts);
    }

    static SDL_mutex *queuelock = NULL;
    void lockqueue() { if(queuelock) SDL_LockMutex(queuelock); }
    void unlockqueue() { if(queuelock) SDL_UnlockMutex(queuelock); }

    static bool haslowlatency = false, haspresentid = false;
    static uint64_t latencyframe = 0;

    // frameid tags the work with the frame's latency id, so the driver knows how long each frame keeps the GPU busy
    VkResult submit(const VkSubmitInfo2 &si, VkFence fence, uint64_t frameid)
    {
        VkSubmitInfo2 tagged = si;
        VkLatencySubmissionPresentIdNV lid = { VK_STRUCTURE_TYPE_LATENCY_SUBMISSION_PRESENT_ID_NV };
        lid.presentID = frameid;
        if(frameid && haslowlatency && !si.pNext) tagged.pNext = &lid;
        lockqueue();
        VkResult r = vkQueueSubmit2(vk::queue, 1, &tagged, fence);
        unlockqueue();
        return r;
    }

    uint64_t currentframeid() { return latencyframe; }

    void queueidle()
    {
        lockqueue();
        vkQueueWaitIdle(vk::queue);
        unlockqueue();
    }

    VkCommandBuffer begincommands()
    {
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        VkCommandBufferAllocateInfo cai = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
        cai.commandPool = cmdpool;
        cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai.commandBufferCount = 1;
        if(vkAllocateCommandBuffers(device, &cai, &cmd) != VK_SUCCESS) return VK_NULL_HANDLE;
        VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &bi);
        return cmd;
    }

    bool endcommands(VkCommandBuffer cmd)
    {
        vkEndCommandBuffer(cmd);
        VkCommandBufferSubmitInfo cbi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
        cbi.commandBuffer = cmd;
        VkSubmitInfo2 si = { VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
        si.commandBufferInfoCount = 1;
        si.pCommandBufferInfos = &cbi;
        VkResult r = submit(si, VK_NULL_HANDLE);
        queueidle();
        vkFreeCommandBuffers(device, cmdpool, 1, &cmd);
        return r == VK_SUCCESS;
    }

    bool createdeviceimage(deviceimage &d, int w, int h, VkFormat format, VkImageUsageFlags usage)
    {
        d.w = w;
        d.h = h;
        d.format = format;
        VkImageCreateInfo ici = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = format;
        ici.extent.width = w;
        ici.extent.height = h;
        ici.extent.depth = 1;
        ici.mipLevels = 1;
        ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = usage;
        ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VKCHECK(vkCreateImage(device, &ici, NULL, &d.image));
        VkMemoryRequirements req;
        vkGetImageMemoryRequirements(device, d.image, &req);
        VkMemoryAllocateInfo mai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
        mai.allocationSize = req.size;
        int memtype = findmemorytype(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if(memtype < 0) return false;
        mai.memoryTypeIndex = memtype;
        VKCHECK(vkAllocateMemory(device, &mai, NULL, &d.memory));
        VKCHECK(vkBindImageMemory(device, d.image, d.memory, 0));
        VkImageViewCreateInfo vci = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
        vci.image = d.image;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = format;
        vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vci.subresourceRange.levelCount = 1;
        vci.subresourceRange.layerCount = 1;
        VKCHECK(vkCreateImageView(device, &vci, NULL, &d.view));
        return true;
    }

    void destroydeviceimage(deviceimage &d)
    {
        if(d.view) vkDestroyImageView(device, d.view, NULL);
        if(d.image) vkDestroyImage(device, d.image, NULL);
        if(d.memory) vkFreeMemory(device, d.memory, NULL);
        d = deviceimage();
    }

    // a screen image: GL draws it, Vulkan presents it. With frame generation, GL also fills the frame
    // generation inputs and Vulkan makes the in-between frames that go out before it.
    struct screenframe
    {
        sharedimage img, hudless, mvec, depth;
        GLuint fbo = 0, hudlessfbo = 0, fgfbo = 0;
        deviceimage interp[MAXFGGEN];
        deviceimage hdr, hdrhudless; // with HDR output: the frame (and its copy without the HUD) in HDR10, what's presented
        VkDescriptorSet hdrset = VK_NULL_HANDLE;
        sharedsemaphore glready, vkdone; // GL finished drawing / Vulkan finished with every image of the frame
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        bool glwait = false; // GL waits for vkdone before drawing into it again
        bool hashudless = false, hasfg = false; // filled by GL this frame
        fgcamera camera;

        int textures(GLuint *texs) const
        {
            int n = 0;
            texs[n++] = img.gltex;
            if(hudless.gltex) texs[n++] = hudless.gltex;
            if(mvec.gltex) texs[n++] = mvec.gltex;
            if(depth.gltex) texs[n++] = depth.gltex;
            return n;
        }
    };
    static screenframe frames[MAXVKFRAMES];
    static int numframes = 0, curframe = 0;
    static bool framedone = true; // the current screen image went to Vulkan, so drawing moves on to the next one
    static bool framesdirty = false; // the screen images need making again at the next frame
    static GLuint screenfbo = 0;
    static bool fgimages = false; // the screen images carry frame generation inputs

    static VkSemaphore evaltimeline = VK_NULL_HANDLE; // counts frames Vulkan finished its own work on
    static uint64_t evalcounter = 0;

    // the HDR conversion, a compute pass over each frame's images
    static VkDescriptorSetLayout hdrsetlayout = VK_NULL_HANDLE;
    static VkPipelineLayout hdrlayout = VK_NULL_HANDLE;
    static VkPipeline hdrpipe = VK_NULL_HANDLE;
    static VkDescriptorPool hdrpool = VK_NULL_HANDLE;
    struct hdrparams { float curve[4], flags[4]; };

    static VkSwapchainKHR swapchain = VK_NULL_HANDLE;

    // low latency (VK_NV_low_latency2, what NVIDIA Reflex is on Vulkan): the main thread sleeps before taking input
    // for as long as the frame would otherwise wait in a queue, and marks where the frame's work starts and ends
    static PFN_vkSetLatencySleepModeNV SetLatencySleepModeNV = NULL;
    static PFN_vkLatencySleepNV LatencySleepNV = NULL;
    static PFN_vkSetLatencyMarkerNV SetLatencyMarkerNV = NULL;
    static PFN_vkGetLatencyTimingsNV GetLatencyTimingsNV = NULL;
    static VkSemaphore latencysem = VK_NULL_HANDLE;
    static uint64_t latencyvalue = 0;
    static SDL_mutex *swaplock = NULL; // the main thread uses the swapchain for latency, the presenting thread replaces it
    static SDL_atomic_t sleepmodedirty;
    static VkFormat swapformat = VK_FORMAT_UNDEFINED;
    static VkExtent2D swapextent = { 0, 0 };
    static VkPresentModeKHR presentmode = VK_PRESENT_MODE_FIFO_KHR;
    static vector<VkImage> swapimages;
    static vector<VkSemaphore> swapready;
    static SDL_atomic_t swapdirty;
    static bool hasrelaxed = false, cangrab = false;
    static int wantvsync = 0, wanttear = 0;

    // HDR: the swapchain is HDR10 when the image being presented is, and the surface says whether it can be
    static bool hascolorspace = false, hashdrmetadata = false, swaphdr = false, swapwanthdr = false;
    static bool hassurfacemaint = false; // what VK_EXT_swapchain_maintenance1 (wanted by NGX) needs from the instance
    static SDL_atomic_t hdrsurface;
    // what the presenting thread ran into, for the main thread to act on and report: the display wouldn't take an HDR
    // swapchain or stopped showing it (HDR output goes off), the last error remaking the swapchain, and a window that
    // stopped getting images while it had focus
    static SDL_atomic_t hdrfailed, swaperror, swapstalled, windowactive;
    static PFN_vkSetHdrMetadataEXT SetHdrMetadataEXT = NULL;
    #define HDRFORMAT VK_FORMAT_A2B10G10R10_UNORM_PACK32 // what the HDR images are, the swapchain's may be its BGR twin

    static bool ishdrformat(const VkSurfaceFormatKHR &f)
    {
        return f.colorSpace == VK_COLOR_SPACE_HDR10_ST2084_EXT && (f.format == VK_FORMAT_A2B10G10R10_UNORM_PACK32 || f.format == VK_FORMAT_A2R10G10B10_UNORM_PACK32);
    }

    static void sethdrmetadata()
    {
        SDL_AtomicSet(&hdrmetadirty, 0);
        if(!SetHdrMetadataEXT || !swapchain) return;
        // mastered on a Rec.2020 display with the chosen peak
        VkHdrMetadataEXT md = { VK_STRUCTURE_TYPE_HDR_METADATA_EXT };
        md.displayPrimaryRed = { 0.708f, 0.292f };
        md.displayPrimaryGreen = { 0.170f, 0.797f };
        md.displayPrimaryBlue = { 0.131f, 0.046f };
        md.whitePoint = { 0.3127f, 0.3290f };
        md.maxLuminance = hdrpeak;
        md.minLuminance = 0.0001f;
        md.maxContentLightLevel = hdrpeak;
        md.maxFrameAverageLightLevel = hdrpaperwhite;
        SetHdrMetadataEXT(device, 1, &swapchain, &md);
    }

    static void APIENTRY bindframebuffer(GLenum target, GLuint fbo)
    {
        // drawing to the screen after the last image was presented starts the next one
        if(!fbo && framedone && target != GL_READ_FRAMEBUFFER) vk_beginframe();
        glBindFramebufferreal(target, fbo ? fbo : screenfbo);
    }

    static const char *presentmodename(VkPresentModeKHR mode)
    {
        switch(mode)
        {
            case VK_PRESENT_MODE_IMMEDIATE_KHR: return "immediate";
            case VK_PRESENT_MODE_MAILBOX_KHR: return "mailbox";
            case VK_PRESENT_MODE_FIFO_KHR: return "fifo";
            case VK_PRESENT_MODE_FIFO_RELAXED_KHR: return "fifo relaxed";
            default: return "other";
        }
    }

    static double seconds() { return double(SDL_GetPerformanceCounter())/double(SDL_GetPerformanceFrequency()); }

    static void latencymarker(VkLatencyMarkerNV marker, uint64_t frameid)
    {
        if(!haslowlatency) return; // marked even with the sleep off, so the driver's timings compare
        SDL_LockMutex(swaplock);
        if(swapchain)
        {
            VkSetLatencyMarkerInfoNV mi = { VK_STRUCTURE_TYPE_SET_LATENCY_MARKER_INFO_NV };
            mi.presentID = frameid;
            mi.marker = marker;
            SetLatencyMarkerNV(device, swapchain, &mi);
        }
        SDL_UnlockMutex(swaplock);
    }

    static void sleepuntil(double t)
    {
        for(;;)
        {
            double left = t - seconds();
            if(left <= 0) return;
            if(left > 0.0005)
            {
                struct timespec ts;
                double sleep = left - 0.0003; // the last bit spins, sleeps overshoot
                ts.tv_sec = time_t(sleep);
                ts.tv_nsec = long((sleep - ts.tv_sec)*1e9);
                nanosleep(&ts, NULL);
            }
        }
    }

    static void destroyswapchain()
    {
        loopv(swapready) vkDestroySemaphore(device, swapready[i], NULL);
        swapready.setsize(0);
        swapimages.setsize(0);
        if(swapchain) vkDestroySwapchainKHR(device, swapchain, NULL);
        swapchain = VK_NULL_HANDLE;
    }

    // called by the presenting thread once it runs, before that by vk_init
    static bool createswapchain()
    {
        queueidle();
        SDL_AtomicSet(&swapdirty, 0);

        // the presenting thread calls this, so errors are left for the main thread to report rather than printed
        VkSurfaceCapabilitiesKHR caps;
        VkResult capsres = vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physdev, surface, &caps);
        if(capsres != VK_SUCCESS) { SDL_AtomicSet(&swaperror, int(capsres)); return false; }
        VkExtent2D extent = caps.currentExtent;
        if(extent.width == UINT32_MAX)
        {
            int w = 0, h = 0;
            SDL_Vulkan_GetDrawableSize(screen, &w, &h);
            extent.width = clamp(uint32_t(max(w, 0)), caps.minImageExtent.width, caps.maxImageExtent.width);
            extent.height = clamp(uint32_t(max(h, 0)), caps.minImageExtent.height, caps.maxImageExtent.height);
        }
        if(!extent.width || !extent.height)
        {
            // minimized: nothing to present to until the window comes back
            destroyswapchain();
            swapextent = extent;
            return true;
        }

        uint32_t numformats = 0;
        vkGetPhysicalDeviceSurfaceFormatsKHR(physdev, surface, &numformats, NULL);
        vector<VkSurfaceFormatKHR> formats;
        vkGetPhysicalDeviceSurfaceFormatsKHR(physdev, surface, &numformats, formats.pad(numformats));
        if(formats.empty()) return false;
        // GL wrote the values straight to its window, so a UNORM swapchain shows the same thing
        static const VkFormat preferred[] = { VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_A2B10G10R10_UNORM_PACK32, VK_FORMAT_A2R10G10B10_UNORM_PACK32 };
        VkSurfaceFormatKHR format = formats[0];
        bool found = false, hdrfound = false;
        loopv(formats) if(ishdrformat(formats[i]))
        {
            hdrfound = true;
            if(swapwanthdr && (!found || formats[i].format == HDRFORMAT))
            {
                format = formats[i];
                found = true;
            }
        }
        SDL_AtomicSet(&hdrsurface, hdrfound ? 1 : 0);
        loopj(int(sizeof(preferred)/sizeof(preferred[0])))
        {
            if(found) break;
            loopv(formats) if(formats[i].format == preferred[j] && formats[i].colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
            {
                format = formats[i];
                found = true;
                break;
            }
        }

        uint32_t nummodes = 0;
        vkGetPhysicalDeviceSurfacePresentModesKHR(physdev, surface, &nummodes, NULL);
        vector<VkPresentModeKHR> modes;
        vkGetPhysicalDeviceSurfacePresentModesKHR(physdev, surface, &nummodes, modes.pad(nummodes));
        bool hasimmediate = modes.find(VK_PRESENT_MODE_IMMEDIATE_KHR) >= 0, hasmailbox = modes.find(VK_PRESENT_MODE_MAILBOX_KHR) >= 0;
        hasrelaxed = modes.find(VK_PRESENT_MODE_FIFO_RELAXED_KHR) >= 0;
        VkPresentModeKHR mode = VK_PRESENT_MODE_FIFO_KHR;
        if(!wantvsync) mode = hasimmediate ? VK_PRESENT_MODE_IMMEDIATE_KHR : (hasmailbox ? VK_PRESENT_MODE_MAILBOX_KHR : VK_PRESENT_MODE_FIFO_KHR);
        else if(wanttear && hasrelaxed) mode = VK_PRESENT_MODE_FIFO_RELAXED_KHR;

        if(!(caps.supportedUsageFlags&VK_IMAGE_USAGE_TRANSFER_DST_BIT))
        {
            SDL_AtomicSet(&swaperror, int(VK_ERROR_FORMAT_NOT_SUPPORTED));
            return false;
        }
        // with vsync every image beyond the one shown and the one being drawn is a frame of latency, so low latency keeps
        // the fewest the surface allows
        uint32_t count = mode != VK_PRESENT_MODE_IMMEDIATE_KHR && mode != VK_PRESENT_MODE_MAILBOX_KHR && lowlatency ? max(caps.minImageCount, 2u) : max(caps.minImageCount + 1, 3u);
        if(caps.maxImageCount) count = min(count, caps.maxImageCount);
        VkCompositeAlphaFlagBitsKHR alpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        if(!(caps.supportedCompositeAlpha&alpha)) loopi(4) if(caps.supportedCompositeAlpha&(1<<i)) { alpha = VkCompositeAlphaFlagBitsKHR(1<<i); break; }

        VkSwapchainCreateInfoKHR sci = { VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR };
        sci.surface = surface;
        sci.minImageCount = count;
        sci.imageFormat = format.format;
        sci.imageColorSpace = format.colorSpace;
        sci.imageExtent = extent;
        sci.imageArrayLayers = 1;
        sci.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT|(caps.supportedUsageFlags&(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_STORAGE_BIT|VK_IMAGE_USAGE_TRANSFER_SRC_BIT));
        cangrab = (caps.supportedUsageFlags&VK_IMAGE_USAGE_TRANSFER_SRC_BIT) != 0;
        sci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        sci.preTransform = caps.currentTransform;
        sci.compositeAlpha = alpha;
        sci.presentMode = mode;
        sci.clipped = VK_TRUE;
        sci.oldSwapchain = swapchain;
        VkSwapchainLatencyCreateInfoNV lci = { VK_STRUCTURE_TYPE_SWAPCHAIN_LATENCY_CREATE_INFO_NV };
        lci.latencyModeEnable = VK_TRUE;
        if(haslowlatency) sci.pNext = &lci;
        VkSwapchainKHR newswapchain = VK_NULL_HANDLE;
        VkResult created = dbgvkhdrfail == 1 && ishdrformat(format) ? VK_ERROR_INITIALIZATION_FAILED : vkCreateSwapchainKHR(device, &sci, NULL, &newswapchain);
        if(created != VK_SUCCESS)
        {
            // the old swapchain is retired even so, and can't show anything more: it goes
            if(swaplock) SDL_LockMutex(swaplock);
            destroyswapchain();
            if(swaplock) SDL_UnlockMutex(swaplock);
            SDL_AtomicSet(&swaperror, int(created));
            if(ishdrformat(format))
            {
                // the display offered HDR but wouldn't take it: SDR, and HDR output goes off
                SDL_AtomicSet(&hdrfailed, 1);
                swapwanthdr = false;
                return createswapchain();
            }
            return false;
        }
        if(swaplock) SDL_LockMutex(swaplock);
        destroyswapchain();
        swapchain = newswapchain;
        if(swaplock) SDL_UnlockMutex(swaplock);
        SDL_AtomicSet(&sleepmodedirty, 1); // a new swapchain starts without its latency mode
        swaphdr = ishdrformat(format);
        if(swaphdr) sethdrmetadata();
        swapformat = format.format;
        swapextent = extent;
        presentmode = mode;

        uint32_t numimages = 0;
        vkGetSwapchainImagesKHR(device, swapchain, &numimages, NULL);
        vkGetSwapchainImagesKHR(device, swapchain, &numimages, swapimages.pad(numimages));
        loopi(int(numimages))
        {
            VkSemaphoreCreateInfo semci = { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
            VKCHECK(vkCreateSemaphore(device, &semci, NULL, &swapready.add()));
        }
        return true;
    }

    // the presenting thread: shows the generated frames of each rendered frame, then the rendered one, spaced evenly
    struct presentjob
    {
        int slot, numgen;
        uint64_t evalvalue, frameid;
        double interval;
    };
    #define PRESENTRING 8
    struct presentcmd
    {
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        VkSemaphore acquired = VK_NULL_HANDLE;
    };
    static presentcmd presentcmds[PRESENTRING];
    static int presentnext = 0;
    static VkCommandPool presentpool = VK_NULL_HANDLE;
    static SDL_Thread *presentthread = NULL;
    static SDL_mutex *presentlock = NULL;
    static SDL_cond *presentcond = NULL, *presentdone = NULL;
    static vector<presentjob> presentjobs;
    static bool presentquit = false;
    static bool slotbusy[MAXVKFRAMES];
    static double lastjobtime = 0, jobinterval = 0, lastcputime = 0, workinterval = 0;
    static int fgauto = 1, fgautonext = 1, fgautoframes = 0, fglastgen = 0;

    // automatic frame generation: the fewest generated frames that reach the refresh rate from how fast the game
    // could render, which is the processor time the main thread spends on a frame: blocking in the driver or waiting
    // for presenting costs none, so it holds when vsync and frame generation slow the frames down
    static int autogen(int maxgen)
    {
        if(workinterval <= 0) return min(fgauto, maxgen);
        double fps = 1.0/workinterval, target = 0.95*(refreshrate > 0 ? refreshrate : 60);
        int want = clamp(int(ceil(target/fps)) - 1, 1, maxgen);
        // a change has to hold for half a second, so the count doesn't flicker
        if(want == fgauto) fgautoframes = 0;
        else if(want != fgautonext) { fgautonext = want; fgautoframes = 0; }
        else if(++fgautoframes*workinterval >= 0.5) { fgauto = want; fgautoframes = 0; }
        return min(fgauto, maxgen);
    }
    static vector<float> presenttimes; // milliseconds between presents, for presenttimes
    static double lastpresent = 0;

    // vkgrab: the presenting thread copies presented images back, the main thread saves them
    static string grabname = "";
    static int grabwant = 0, grabtaken = 0;
    struct grabbedimage { ImageData *image; int index; string info; };
    static vector<grabbedimage> grabbed;
    static VkBuffer grabbuf = VK_NULL_HANDLE;
    static VkDeviceMemory grabmem = VK_NULL_HANDLE;

    static bool startgrab(VkCommandBuffer cmd, VkImage image)
    {
        VkDeviceSize size = VkDeviceSize(swapextent.width)*swapextent.height*4;
        VkBufferCreateInfo bci = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        bci.size = size;
        bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        if(vkCreateBuffer(device, &bci, NULL, &grabbuf) != VK_SUCCESS) return false;
        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(device, grabbuf, &req);
        VkMemoryAllocateInfo mai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
        mai.allocationSize = req.size;
        int memtype = findmemorytype(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if(memtype < 0 || (mai.memoryTypeIndex = memtype, vkAllocateMemory(device, &mai, NULL, &grabmem) != VK_SUCCESS))
        {
            vkDestroyBuffer(device, grabbuf, NULL);
            grabbuf = VK_NULL_HANDLE;
            return false;
        }
        vkBindBufferMemory(device, grabbuf, grabmem, 0);
        imagebarrier(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
        VkBufferImageCopy region = {};
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.layerCount = 1;
        region.imageExtent.width = swapextent.width;
        region.imageExtent.height = swapextent.height;
        region.imageExtent.depth = 1;
        vkCmdCopyImageToBuffer(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, grabbuf, 1, &region);
        imagebarrier(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
        return true;
    }

    static void finishgrab(VkFence fence)
    {
        vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
        void *data = NULL;
        if(vkMapMemory(device, grabmem, 0, VK_WHOLE_SIZE, 0, &data) == VK_SUCCESS)
        {
            string info = "";
            int w = swapextent.width, h = swapextent.height;
            ImageData *image = new ImageData(w, h, 3);
            const uchar *src = (const uchar *)data;
            bool bgr = swapformat == VK_FORMAT_B8G8R8A8_UNORM || swapformat == VK_FORMAT_B8G8R8A8_SRGB;
            if(swaphdr)
            {
                // HDR10: saved with the peak as white, and its light reported
                float peak = max(float(hdrpeak), 1.0f), maxnits = 0, above = 0;
                double sum = 0;
                bool rfirst = swapformat == VK_FORMAT_A2B10G10R10_UNORM_PACK32;
                loopi(w*h)
                {
                    uint v = ((const uint *)src)[i];
                    float c[3] = { float(v&0x3FF), float((v>>10)&0x3FF), float((v>>20)&0x3FF) }, rgb[3];
                    loopj(3)
                    {
                        // ST.2084 back to nits
                        float e = powf(c[j]/1023.0f, 1/78.84375f);
                        c[j] = 10000*powf(max(e - 0.8359375f, 0.0f)/(18.8515625f - 18.6875f*e), 1/0.1593017578125f);
                    }
                    if(!rfirst) swap(c[0], c[2]);
                    // Rec.2020 to Rec.709
                    rgb[0] = 1.6605f*c[0] - 0.5876f*c[1] - 0.0728f*c[2];
                    rgb[1] = -0.1246f*c[0] + 1.1329f*c[1] - 0.0083f*c[2];
                    rgb[2] = -0.0182f*c[0] - 0.1006f*c[1] + 1.1187f*c[2];
                    float nits = 0.2126f*rgb[0] + 0.7152f*rgb[1] + 0.0722f*rgb[2];
                    maxnits = max(maxnits, nits);
                    sum += nits;
                    if(nits > hdrpaperwhite*1.05f) above++;
                    loopj(3) image->data[i*3+j] = uchar(clamp(powf(max(rgb[j], 0.0f)/peak, 1/2.2f)*255 + 0.5f, 0.0f, 255.0f));
                }
                formatstring(info, " (HDR: brightest %.0f nits, mean %.0f nits, %.1f%% above paper white at %.0f nits)",
                    maxnits, sum/(w*h), 100*above/(w*h), float(hdrpaperwhite));
            }
            else loopi(w*h)
            {
                image->data[i*3+0] = src[i*4+(bgr ? 2 : 0)];
                image->data[i*3+1] = src[i*4+1];
                image->data[i*3+2] = src[i*4+(bgr ? 0 : 2)];
            }
            vkUnmapMemory(device, grabmem);
            SDL_LockMutex(presentlock);
            grabbedimage &g = grabbed.add();
            g.image = image;
            g.index = grabtaken++;
            copystring(g.info, info);
            SDL_UnlockMutex(presentlock);
        }
        vkDestroyBuffer(device, grabbuf, NULL);
        vkFreeMemory(device, grabmem, NULL);
        grabbuf = VK_NULL_HANDLE;
        grabmem = VK_NULL_HANDLE;
    }

    static void savegrabs()
    {
        SDL_LockMutex(presentlock);
        vector<grabbedimage> done;
        loopv(grabbed) done.add(grabbed[i]);
        grabbed.setsize(0);
        SDL_UnlockMutex(presentlock);
        loopv(done)
        {
            defformatstring(name, "%s_%d.png", grabname, done[i].index);
            saveimage(name, *done[i].image, IFMT_PNG, 9, false);
            conoutf(colourwhite, "Vulkan: saved presented image %s%s", name, done[i].info);
            delete done[i].image;
        }
    }

    // copies src (GL oriented, in TRANSFER_SRC layout unless real) into the next swapchain image and presents it;
    // for the rendered image it also hands the frame's screen image back to GL, unless it was HDR and that went already
    static void latencymarker(VkLatencyMarkerNV marker, uint64_t frameid);

    // the swapchain is the main thread's too (latency sleep and markers), so it's only used under swaplock, and waiting
    // for an image polls instead of holding the lock while it blocks
    static VkResult acquireimage(VkSemaphore acquired, uint32_t &index)
    {
        // a hidden window (another workspace) can hold its images forever with vsync on, then frames are dropped
        double giveup = seconds() + 0.1;
        for(;;)
        {
            SDL_LockMutex(swaplock);
            VkResult r = vkAcquireNextImageKHR(device, swapchain, 0, acquired, VK_NULL_HANDLE, &index);
            SDL_UnlockMutex(swaplock);
            if(r != VK_NOT_READY && r != VK_TIMEOUT) return r;
            if(seconds() >= giveup) return VK_TIMEOUT;
            struct timespec ts = { 0, 100000 };
            nanosleep(&ts, NULL);
        }
    }

    static void presentimage(VkImage src, int srcw, int srch, bool hdrimage, screenframe *real, uint64_t frameid = 0)
    {
        presentcmd &p = presentcmds[presentnext];
        presentnext = (presentnext+1)%PRESENTRING;
        vkWaitForFences(device, 1, &p.fence, VK_TRUE, UINT64_MAX);

        if(hdrimage != swapwanthdr && !(hdrimage && SDL_AtomicGet(&hdrfailed)))
        {
            // the swapchain follows what's presented
            swapwanthdr = hdrimage;
            SDL_AtomicSet(&swapdirty, 1);
        }
        if(SDL_AtomicGet(&swapdirty)) createswapchain();
        if(swaphdr && SDL_AtomicGet(&hdrmetadirty)) sethdrmetadata();
        uint32_t index = UINT32_MAX;
        if(swapchain && !(dbgvkhdrfail == 2 && swaphdr)) // the test of a display that never gives an HDR image back
        {
            VkResult r = acquireimage(p.acquired, index);
            if(r == VK_ERROR_OUT_OF_DATE_KHR)
            {
                index = UINT32_MAX;
                if(createswapchain() && swapchain) r = acquireimage(p.acquired, index);
            }
            if(r == VK_SUBOPTIMAL_KHR) SDL_AtomicSet(&swapdirty, 1);
            else if(r != VK_SUCCESS) index = UINT32_MAX;
        }
        // an HDR frame left over from before HDR went off would show washed out on an SDR swapchain
        if(hdrimage && !swaphdr) index = UINT32_MAX;

        // a window with focus that stops getting images (a hidden one may, with vsync) has a swapchain the display let
        // go of: after a second it is made again, after three in HDR, HDR is given up
        static double stallstart = 0;
        static bool stallremade = false;
        if(index != UINT32_MAX || !SDL_AtomicGet(&windowactive)) { stallstart = 0; stallremade = false; }
        else
        {
            double now = seconds();
            if(!stallstart) stallstart = now;
            else if(now - stallstart > 1 && !stallremade)
            {
                stallremade = true;
                SDL_AtomicSet(&swapstalled, 1);
                SDL_AtomicSet(&swapdirty, 1);
            }
            else if(now - stallstart > 3 && swaphdr)
            {
                SDL_AtomicSet(&hdrfailed, 1);
                swapwanthdr = false;
                SDL_AtomicSet(&swapdirty, 1);
                stallstart = 0;
            }
        }
        if(index == UINT32_MAX && !real) return; // nowhere to show it

        vkResetFences(device, 1, &p.fence);
        VkCommandBuffer cmd = p.cmd;
        vkResetCommandBuffer(cmd, 0);
        VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &bi);
        // the frame's own submission finished before this thread got here, so only layouts change
        if(real) imagebarrier(cmd, src, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE, VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
        bool grab = false;
        if(index != UINT32_MAX)
        {
            VkImage dst = swapimages[index];
            imagebarrier(cmd, dst, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_NONE, VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
            // GL's rows start at the bottom
            VkImageBlit region = {};
            region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.srcSubresource.layerCount = 1;
            region.srcOffsets[0].y = srch;
            region.srcOffsets[1].x = srcw;
            region.srcOffsets[1].z = 1;
            region.dstSubresource = region.srcSubresource;
            region.dstOffsets[1].x = swapextent.width;
            region.dstOffsets[1].y = swapextent.height;
            region.dstOffsets[1].z = 1;
            bool samesize = int(swapextent.width) == srcw && int(swapextent.height) == srch;
            vkCmdBlitImage(cmd, src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region, samesize ? VK_FILTER_NEAREST : VK_FILTER_LINEAR);
            if(grabtaken < grabwant && cangrab) grab = startgrab(cmd, dst);
            imagebarrier(cmd, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE);
        }
        if(real && src == real->img.image)
        {
            // the screen image goes back to GL, in the layout GL draws in
            imagebarrier(cmd, real->img.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_NONE, VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
                queuefamily, VK_QUEUE_FAMILY_EXTERNAL);
        }
        vkEndCommandBuffer(cmd);

        VkSemaphoreSubmitInfo waits[1], signals[2];
        int numwaits = 0, numsignals = 0;
        if(index != UINT32_MAX)
        {
            VkSemaphoreSubmitInfo &w = waits[numwaits++];
            w = VkSemaphoreSubmitInfo{ VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO };
            w.semaphore = p.acquired;
            w.stageMask = VK_PIPELINE_STAGE_2_BLIT_BIT;
            VkSemaphoreSubmitInfo &s = signals[numsignals++];
            s = VkSemaphoreSubmitInfo{ VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO };
            s.semaphore = swapready[index];
            s.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        }
        if(real)
        {
            VkSemaphoreSubmitInfo &s = signals[numsignals++];
            s = VkSemaphoreSubmitInfo{ VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO };
            s.semaphore = real->vkdone.sem;
            s.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        }
        VkCommandBufferSubmitInfo cbi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
        cbi.commandBuffer = cmd;
        VkSubmitInfo2 si = { VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
        si.waitSemaphoreInfoCount = numwaits;
        si.pWaitSemaphoreInfos = waits;
        si.commandBufferInfoCount = 1;
        si.pCommandBufferInfos = &cbi;
        si.signalSemaphoreInfoCount = numsignals;
        si.pSignalSemaphoreInfos = signals;
        VkLatencySubmissionPresentIdNV lid = { VK_STRUCTURE_TYPE_LATENCY_SUBMISSION_PRESENT_ID_NV };
        lid.presentID = frameid;
        if(real && frameid && haslowlatency) si.pNext = &lid;
        lockqueue();
        VkResult r = vkQueueSubmit2(vk::queue, 1, &si, p.fence);
        if(r == VK_SUCCESS && index != UINT32_MAX)
        {
            VkPresentInfoKHR pi = { VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
            pi.waitSemaphoreCount = 1;
            pi.pWaitSemaphores = &swapready[index];
            pi.swapchainCount = 1;
            pi.pSwapchains = &swapchain;
            pi.pImageIndices = &index;
            // the rendered frame's present carries the id its latency markers use
            VkPresentIdKHR pid = { VK_STRUCTURE_TYPE_PRESENT_ID_KHR };
            pid.swapchainCount = 1;
            pid.pPresentIds = &frameid;
            if(real && frameid && haspresentid) pi.pNext = &pid;
            SDL_LockMutex(swaplock); // the main thread's latency calls use the swapchain too
            if(real && frameid) latencymarker(VK_LATENCY_MARKER_PRESENT_START_NV, frameid);
            VkResult pr = vkQueuePresentKHR(vk::queue, &pi);
            if(real && frameid) latencymarker(VK_LATENCY_MARKER_PRESENT_END_NV, frameid);
            SDL_UnlockMutex(swaplock);
            if(pr == VK_ERROR_OUT_OF_DATE_KHR || pr == VK_SUBOPTIMAL_KHR) SDL_AtomicSet(&swapdirty, 1);
        }
        unlockqueue();
        if(r == VK_SUCCESS && index != UINT32_MAX)
        {
            double now = seconds();
            SDL_LockMutex(presentlock);
            if(lastpresent > 0 && presenttimes.length() < 200000) presenttimes.add(float((now - lastpresent)*1000));
            lastpresent = now;
            SDL_UnlockMutex(presentlock);
        }
        if(r != VK_SUCCESS) fprintf(stderr, "Vulkan: presenting failed: %s\n", resultname(r));
        if(grab) finishgrab(p.fence);
    }

    static int presentloop(void *)
    {
        for(;;)
        {
            SDL_LockMutex(presentlock);
            while(presentjobs.empty() && !presentquit) SDL_CondWait(presentcond, presentlock);
            if(presentjobs.empty() && presentquit)
            {
                SDL_UnlockMutex(presentlock);
                break;
            }
            presentjob job = presentjobs[0];
            presentjobs.remove(0);
            SDL_UnlockMutex(presentlock);

            // the frame's generated images exist once its submission finished
            VkSemaphoreWaitInfo wi = { VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO };
            wi.semaphoreCount = 1;
            wi.pSemaphores = &evaltimeline;
            wi.pValues = &job.evalvalue;
            vkWaitSemaphores(device, &wi, UINT64_MAX);

            screenframe &f = frames[job.slot];
            double start = seconds(), step = job.interval/(job.numgen+1);
            loopi(job.numgen)
            {
                if(i) sleepuntil(start + i*step);
                presentimage(f.interp[i].image, f.interp[i].w, f.interp[i].h, f.interp[i].format == HDRFORMAT, NULL);
            }
            if(job.numgen) sleepuntil(start + job.numgen*step);
            if(f.hdr.image) presentimage(f.hdr.image, f.hdr.w, f.hdr.h, true, &f, job.frameid);
            else presentimage(f.img.image, f.img.w, f.img.h, false, &f, job.frameid);

            SDL_LockMutex(presentlock);
            slotbusy[job.slot] = false;
            SDL_CondBroadcast(presentdone);
            SDL_UnlockMutex(presentlock);
        }
        return 0;
    }

    static void waitslot(int slot)
    {
        if(!presentlock) return;
        SDL_LockMutex(presentlock);
        while(slotbusy[slot]) SDL_CondWait(presentdone, presentlock);
        SDL_UnlockMutex(presentlock);
    }

    static void drainpresent()
    {
        loopi(MAXVKFRAMES) waitslot(i);
        queueidle();
    }

    static bool startpresent()
    {
        VkCommandPoolCreateInfo pci = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
        pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pci.queueFamilyIndex = queuefamily;
        VKCHECK(vkCreateCommandPool(device, &pci, NULL, &presentpool));
        loopi(PRESENTRING)
        {
            presentcmd &p = presentcmds[i];
            VkCommandBufferAllocateInfo cai = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
            cai.commandPool = presentpool;
            cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            cai.commandBufferCount = 1;
            VKCHECK(vkAllocateCommandBuffers(device, &cai, &p.cmd));
            VkFenceCreateInfo fci = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
            fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
            VKCHECK(vkCreateFence(device, &fci, NULL, &p.fence));
            VkSemaphoreCreateInfo semci = { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
            VKCHECK(vkCreateSemaphore(device, &semci, NULL, &p.acquired));
        }
        VkSemaphoreTypeCreateInfo tci = { VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO };
        tci.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        tci.initialValue = 0;
        VkSemaphoreCreateInfo semci = { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
        semci.pNext = &tci;
        VKCHECK(vkCreateSemaphore(device, &semci, NULL, &evaltimeline));
        evalcounter = 0;
        presentquit = false;
        loopi(MAXVKFRAMES) slotbusy[i] = false;
        lastjobtime = jobinterval = 0;
        presentthread = SDL_CreateThread(presentloop, "present", NULL);
        return presentthread != NULL;
    }

    static void stoppresent()
    {
        if(presentthread)
        {
            SDL_LockMutex(presentlock);
            presentquit = true;
            SDL_CondBroadcast(presentcond);
            SDL_UnlockMutex(presentlock);
            SDL_WaitThread(presentthread, NULL);
            presentthread = NULL;
        }
        presentjobs.setsize(0);
        if(device) queueidle();
        loopi(PRESENTRING)
        {
            presentcmd &p = presentcmds[i];
            if(p.fence) vkDestroyFence(device, p.fence, NULL);
            if(p.acquired) vkDestroySemaphore(device, p.acquired, NULL);
            p = presentcmd();
        }
        if(presentpool) vkDestroyCommandPool(device, presentpool, NULL);
        presentpool = VK_NULL_HANDLE;
        if(evaltimeline) vkDestroySemaphore(device, evaltimeline, NULL);
        evaltimeline = VK_NULL_HANDLE;
        presentnext = 0;
    }

    static void destroyframes()
    {
        if(!numframes) return;
        glFinish();
        drainpresent();
        ngx_fgrelease();
        if(glBindFramebufferreal) glBindFramebufferreal(GL_FRAMEBUFFER, 0);
        screenfbo = 0;
        loopi(numframes)
        {
            screenframe &f = frames[i];
            if(f.fbo) glDeleteFramebuffers_(1, &f.fbo);
            if(f.hudlessfbo) glDeleteFramebuffers_(1, &f.hudlessfbo);
            if(f.fgfbo) glDeleteFramebuffers_(1, &f.fgfbo);
            destroysharedimage(f.img);
            destroysharedimage(f.hudless);
            destroysharedimage(f.mvec);
            destroysharedimage(f.depth);
            loopj(MAXFGGEN) destroydeviceimage(f.interp[j]);
            destroydeviceimage(f.hdr);
            destroydeviceimage(f.hdrhudless);
            destroysharedsemaphore(f.glready);
            destroysharedsemaphore(f.vkdone);
            if(f.cmd) vkFreeCommandBuffers(device, cmdpool, 1, &f.cmd);
            if(f.fence) vkDestroyFence(device, f.fence, NULL);
            f = screenframe();
        }
        if(hdrpool) vkResetDescriptorPool(device, hdrpool, 0);
        numframes = 0;
        fgimages = false;
    }

    static void destroyhdrpipeline()
    {
        if(hdrpipe) vkDestroyPipeline(device, hdrpipe, NULL);
        if(hdrlayout) vkDestroyPipelineLayout(device, hdrlayout, NULL);
        if(hdrsetlayout) vkDestroyDescriptorSetLayout(device, hdrsetlayout, NULL);
        if(hdrpool) vkDestroyDescriptorPool(device, hdrpool, NULL);
        hdrpipe = VK_NULL_HANDLE;
        hdrlayout = VK_NULL_HANDLE;
        hdrsetlayout = VK_NULL_HANDLE;
        hdrpool = VK_NULL_HANDLE;
    }

    static bool createhdrpipeline()
    {
        if(hdrpipe) return true;
        VkDescriptorSetLayoutBinding bindings[4];
        loopi(4)
        {
            bindings[i] = VkDescriptorSetLayoutBinding{};
            bindings[i].binding = i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo dslci = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        dslci.bindingCount = 4;
        dslci.pBindings = bindings;
        VKCHECK(vkCreateDescriptorSetLayout(device, &dslci, NULL, &hdrsetlayout));
        VkPushConstantRange range = { VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(hdrparams) };
        VkPipelineLayoutCreateInfo plci = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        plci.setLayoutCount = 1;
        plci.pSetLayouts = &hdrsetlayout;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges = &range;
        VKCHECK(vkCreatePipelineLayout(device, &plci, NULL, &hdrlayout));
        VkShaderModuleCreateInfo smci = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
        smci.codeSize = sizeof(hdrencodespv);
        smci.pCode = hdrencodespv;
        VkShaderModule module = VK_NULL_HANDLE;
        VKCHECK(vkCreateShaderModule(device, &smci, NULL, &module));
        VkComputePipelineCreateInfo cpci = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
        cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cpci.stage.module = module;
        cpci.stage.pName = "main";
        cpci.layout = hdrlayout;
        VkResult r = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, NULL, &hdrpipe);
        vkDestroyShaderModule(device, module, NULL);
        VKCHECK(r);
        VkDescriptorPoolSize size = { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 4*MAXVKFRAMES };
        VkDescriptorPoolCreateInfo dpci = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        dpci.maxSets = MAXVKFRAMES;
        dpci.poolSizeCount = 1;
        dpci.pPoolSizes = &size;
        VKCHECK(vkCreateDescriptorPool(device, &dpci, NULL, &hdrpool));
        return true;
    }

    // the HDR curve for hdrencode.comp: the frame's light k (the tonemap showed 1 - 2^-k) is shown as it was up to the
    // knee, then rises to the peak at the light hdrrange names. The knee moves down when the peak is too far to reach
    // from it without the curve getting steeper than the tonemap's own.
    static void hdrcurve(float &knee, float &soft)
    {
        const float maxlight = hdrrange;
        float peak = max(hdrpeak/hdrpaperwhite, 1.0f); // in paper whites
        // the most a straight line leaving the knee as steeply as the tonemap reaches
        #define REACH(t) ((t) + M_LN2*(1 - (t))*(maxlight + log2f(1 - (t))))
        knee = 0.5f;
        if(REACH(knee) < peak*1.01f)
        {
            float lo = 0, hi = 0.5f;
            loopi(30)
            {
                float mid = 0.5f*(lo + hi);
                if(REACH(mid) < peak*1.01f) hi = mid; else lo = mid;
            }
            knee = lo;
        }
        #undef REACH
        // soft s from s(1 - 2^(-range/s)) = (peak - knee)/(1 - knee), which grows with s up to range*ln2
        float range = maxlight + log2f(1 - knee), want = (peak - knee)/(1 - knee);
        if(want >= range*M_LN2*0.999f) { soft = 1e4f; return; }
        float lo = 1e-3f, hi = 1e4f;
        loopi(60)
        {
            float mid = sqrtf(lo*hi);
            if(mid*(1 - exp2f(-range/mid)) < want) lo = mid; else hi = mid;
        }
        soft = sqrtf(lo*hi);
    }

    // the HDR conversion reads the screen image and its copy without the HUD and writes their HDR10 versions
    static bool createhdrset(screenframe &f)
    {
        VkDescriptorSetAllocateInfo dsai = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        dsai.descriptorPool = hdrpool;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts = &hdrsetlayout;
        VKCHECK(vkAllocateDescriptorSets(device, &dsai, &f.hdrset));
        // bindings without an image of their own get one the shader leaves alone
        VkImageView views[4] = { f.img.view, f.hudless.view ? f.hudless.view : f.img.view, f.hdr.view, f.hdrhudless.view ? f.hdrhudless.view : f.hdr.view };
        VkDescriptorImageInfo infos[4];
        VkWriteDescriptorSet writes[4];
        loopi(4)
        {
            infos[i] = VkDescriptorImageInfo{ VK_NULL_HANDLE, views[i], VK_IMAGE_LAYOUT_GENERAL };
            writes[i] = VkWriteDescriptorSet{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
            writes[i].dstSet = f.hdrset;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            writes[i].pImageInfo = &infos[i];
        }
        vkUpdateDescriptorSets(device, 4, writes, 0, NULL);
        return true;
    }

    static bool createframebuffer(GLuint &fbo, sharedimage &color0, sharedimage *color1 = NULL)
    {
        glGenFramebuffers_(1, &fbo);
        glBindFramebufferreal(GL_FRAMEBUFFER, fbo);
        glFramebufferTexture2D_(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, color0.gltarget, color0.gltex, 0);
        if(color1) glFramebufferTexture2D_(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, color1->gltarget, color1->gltex, 0);
        if(glCheckFramebufferStatus_(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        {
            conoutf(colourred, "Vulkan: GL can't draw into a shared image");
            return false;
        }
        return true;
    }

    // gw and gh are the frame generation input size, the G-buffer's
    static bool createframes(int w, int h, int fgw, int fgh)
    {
        bool wantfg = dlssg && ngx_fgmaxgen() > 0 && fgw > 0 && fgh > 0, wanthdr = hdrscreen;
        int numinterp = min(dlssg < 0 ? ngx_fgmaxgen() : int(dlssg), MAXFGGEN); // automatic makes room for the most
        if(wanthdr && !createhdrpipeline()) return false;
        // with HDR the frame comes in 16 bits, for the light in its highlights, and gets an HDR10 copy that's presented
        VkFormat screenvk = wanthdr ? VK_FORMAT_R16G16B16A16_UNORM : VK_FORMAT_R8G8B8A8_UNORM, outvk = wanthdr ? HDRFORMAT : VK_FORMAT_R8G8B8A8_UNORM;
        GLenum screengl = wanthdr ? GL_RGBA16 : GL_RGBA8;
        const VkImageUsageFlags outusage = VK_IMAGE_USAGE_STORAGE_BIT|VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        loopi(MAXVKFRAMES)
        {
            screenframe &f = frames[i];
            numframes = i+1;
            if(!createsharedimage(f.img, w, h, screenvk, screengl) || !createframebuffer(f.fbo, f.img)) return false;
            // the frame without the HUD, for frame generation and for HDR to leave the HUD alone
            if((wantfg || wanthdr) && (!createsharedimage(f.hudless, w, h, screenvk, screengl) || !createframebuffer(f.hudlessfbo, f.hudless))) return false;
            if(wanthdr)
            {
                if(!createdeviceimage(f.hdr, w, h, HDRFORMAT, outusage)) return false;
                if(wantfg && !createdeviceimage(f.hdrhudless, w, h, HDRFORMAT, outusage)) return false;
                if(!createhdrset(f)) return false;
            }
            if(wantfg)
            {
                if(!createsharedimage(f.mvec, fgw, fgh, VK_FORMAT_R16G16_SFLOAT, GL_RG16F) ||
                   !createsharedimage(f.depth, fgw, fgh, VK_FORMAT_R32_SFLOAT, GL_R32F) ||
                   !createframebuffer(f.fgfbo, f.mvec, &f.depth)) return false;
                loopj(numinterp)
                    if(!createdeviceimage(f.interp[j], w, h, outvk, outusage))
                        return false;
            }
            if(!createsharedsemaphore(f.glready) || !createsharedsemaphore(f.vkdone)) return false;
            VkCommandBufferAllocateInfo cai = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
            cai.commandPool = cmdpool;
            cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            cai.commandBufferCount = 1;
            VKCHECK(vkAllocateCommandBuffers(device, &cai, &f.cmd));
            VkFenceCreateInfo fci = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
            fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
            VKCHECK(vkCreateFence(device, &fci, NULL, &f.fence));
        }
        fgimages = wantfg;

        // clear the screen images and hand every image to GL in the layout it draws in
        VkCommandBuffer cmd = begincommands();
        if(!cmd) return false;
        VkSemaphoreSubmitInfo signals[MAXVKFRAMES];
        loopi(numframes)
        {
            screenframe &f = frames[i];
            imagebarrier(cmd, f.img.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE, VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
            VkClearColorValue black = {{ 0, 0, 0, 1 }};
            VkImageSubresourceRange range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            vkCmdClearColorImage(cmd, f.img.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black, 1, &range);
            imagebarrier(cmd, f.img.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
                queuefamily, VK_QUEUE_FAMILY_EXTERNAL);
            sharedimage *aux[3] = { &f.hudless, &f.mvec, &f.depth };
            loopj(3) if(aux[j]->image)
                imagebarrier(cmd, aux[j]->image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE, VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
                    queuefamily, VK_QUEUE_FAMILY_EXTERNAL);
            VkSemaphoreSubmitInfo &s = signals[i];
            s = VkSemaphoreSubmitInfo{ VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO };
            s.semaphore = f.vkdone.sem;
            s.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            f.glwait = true;
        }
        vkEndCommandBuffer(cmd);
        VkCommandBufferSubmitInfo cbi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
        cbi.commandBuffer = cmd;
        VkSubmitInfo2 si = { VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
        si.commandBufferInfoCount = 1;
        si.pCommandBufferInfos = &cbi;
        si.signalSemaphoreInfoCount = numframes;
        si.pSignalSemaphoreInfos = signals;
        VKCHECK(submit(si, VK_NULL_HANDLE));
        queueidle();
        vkFreeCommandBuffers(device, cmdpool, 1, &cmd);

        curframe = numframes-1;
        framedone = true;
        if(dbgvk) conoutf(colourwhite, "Vulkan: screen images %dx%d%s%s", w, h, wanthdr ? " (HDR)" : "", fgimages ? tempformatstring(", frame generation inputs %dx%d", fgw, fgh) : "");
        return true;
    }

    static bool loadglinterop()
    {
        static const char * const glexts[] = { "GL_EXT_memory_object", "GL_EXT_memory_object_fd", "GL_EXT_semaphore", "GL_EXT_semaphore_fd" };
        loopi(int(sizeof(glexts)/sizeof(glexts[0]))) if(!SDL_GL_ExtensionSupported(glexts[i]))
        {
            conoutf(colourred, "Vulkan: GL is missing %s", glexts[i]);
            return false;
        }
        #define GLPROC(name) if(!(name##_ = (decltype(name##_))SDL_GL_GetProcAddress(#name))) { conoutf(colourred, "Vulkan: GL is missing %s", #name); return false; }
        GLPROC(glCreateMemoryObjectsEXT);
        GLPROC(glDeleteMemoryObjectsEXT);
        GLPROC(glMemoryObjectParameterivEXT);
        GLPROC(glImportMemoryFdEXT);
        GLPROC(glTexStorageMem2DEXT);
        GLPROC(glGenSemaphoresEXT);
        GLPROC(glDeleteSemaphoresEXT);
        GLPROC(glImportSemaphoreFdEXT);
        GLPROC(glSignalSemaphoreEXT);
        GLPROC(glWaitSemaphoreEXT);
        GLPROC(glGetUnsignedBytei_vEXT);
        #undef GLPROC
        return true;
    }

    static bool createinstance()
    {
        vector<const char *> exts;
        unsigned int numsdl = 0;
        if(!SDL_Vulkan_GetInstanceExtensions(screen, &numsdl, NULL))
        {
            conoutf(colourred, "Vulkan: %s", SDL_GetError());
            return false;
        }
        const char **sdlexts = exts.pad(numsdl);
        SDL_Vulkan_GetInstanceExtensions(screen, &numsdl, sdlexts);
        addext(exts, VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
        ngx_instanceextensions(exts);

        uint32_t numavail = 0;
        vkEnumerateInstanceExtensionProperties(NULL, &numavail, NULL);
        vector<VkExtensionProperties> avail;
        vkEnumerateInstanceExtensionProperties(NULL, &numavail, avail.pad(numavail));
        vector<const char *> layers;
        bool debugutils = false;
        if(vkvalidation)
        {
            uint32_t numlayers = 0;
            vkEnumerateInstanceLayerProperties(&numlayers, NULL);
            vector<VkLayerProperties> availlayers;
            vkEnumerateInstanceLayerProperties(&numlayers, availlayers.pad(numlayers));
            loopv(availlayers) if(!strcmp(availlayers[i].layerName, "VK_LAYER_KHRONOS_validation")) layers.add("VK_LAYER_KHRONOS_validation");
            if(layers.empty()) conoutf(colouryellow, "Vulkan: the validation layer isn't installed");
            if(hasext(avail, VK_EXT_DEBUG_UTILS_EXTENSION_NAME)) { addext(exts, VK_EXT_DEBUG_UTILS_EXTENSION_NAME); debugutils = true; }
        }
        // HDR10 colour spaces for the swapchain
        hascolorspace = hasext(avail, VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME);
        if(hascolorspace) addext(exts, VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME);
        hassurfacemaint = hasext(avail, VK_EXT_SURFACE_MAINTENANCE_1_EXTENSION_NAME) && hasext(avail, VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME);
        if(hassurfacemaint)
        {
            addext(exts, VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME);
            addext(exts, VK_EXT_SURFACE_MAINTENANCE_1_EXTENSION_NAME);
        }
        loopv(exts) if(!hasext(avail, exts[i]))
        {
            conoutf(colourred, "Vulkan: the instance is missing %s", exts[i]);
            return false;
        }

        VkApplicationInfo ai = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
        ai.pApplicationName = versionname;
        ai.applicationVersion = VK_MAKE_VERSION(versionmajor, versionminor, versionpatch);
        ai.pEngineName = versionname;
        ai.engineVersion = ai.applicationVersion;
        ai.apiVersion = VK_API_VERSION_1_3;
        VkInstanceCreateInfo ici = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
        ici.pApplicationInfo = &ai;
        ici.enabledExtensionCount = exts.length();
        ici.ppEnabledExtensionNames = exts.getbuf();
        ici.enabledLayerCount = layers.length();
        ici.ppEnabledLayerNames = layers.getbuf();
        VKCHECK(vkCreateInstance(&ici, NULL, &instance));

        if(debugutils)
        {
            PFN_vkCreateDebugUtilsMessengerEXT create = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
            VkDebugUtilsMessengerCreateInfoEXT dci = { VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT };
            dci.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT|VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
            dci.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT|VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT|VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
            dci.pfnUserCallback = debugcallback;
            if(create) create(instance, &dci, NULL, &messenger);
            SetDebugUtilsObjectNameEXT = (PFN_vkSetDebugUtilsObjectNameEXT)vkGetInstanceProcAddr(instance, "vkSetDebugUtilsObjectNameEXT");
        }
        return true;
    }

    static bool pickdevice()
    {
        // GL and Vulkan must share the GPU, so pick the device GL runs on
        GLubyte gluuid[GL_UUID_SIZE_EXT];
        GLint numuuids = 0;
        glGetIntegerv(GL_NUM_DEVICE_UUIDS_EXT, &numuuids);
        bool hasuuid = numuuids > 0;
        if(hasuuid) glGetUnsignedBytei_vEXT_(GL_DEVICE_UUID_EXT, 0, gluuid);

        uint32_t numdevs = 0;
        vkEnumeratePhysicalDevices(instance, &numdevs, NULL);
        vector<VkPhysicalDevice> devs;
        vkEnumeratePhysicalDevices(instance, &numdevs, devs.pad(numdevs));
        loopv(devs)
        {
            VkPhysicalDeviceIDProperties idprops = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES };
            VkPhysicalDeviceProperties2 props = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
            props.pNext = &idprops;
            vkGetPhysicalDeviceProperties2(devs[i], &props);
            if(hasuuid ? memcmp(idprops.deviceUUID, gluuid, VK_UUID_SIZE) != 0 : props.properties.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) continue;
            if(props.properties.apiVersion < VK_API_VERSION_1_3) continue;

            uint32_t numfamilies = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(devs[i], &numfamilies, NULL);
            vector<VkQueueFamilyProperties> families;
            vkGetPhysicalDeviceQueueFamilyProperties(devs[i], &numfamilies, families.pad(numfamilies));
            loopvj(families)
            {
                VkBool32 present = VK_FALSE;
                vkGetPhysicalDeviceSurfaceSupportKHR(devs[i], j, surface, &present);
                if(present && (families[j].queueFlags&(VK_QUEUE_GRAPHICS_BIT|VK_QUEUE_COMPUTE_BIT)) == (VK_QUEUE_GRAPHICS_BIT|VK_QUEUE_COMPUTE_BIT))
                {
                    physdev = devs[i];
                    queuefamily = j;
                    devprops = props.properties;
                    return true;
                }
            }
        }
        conoutf(colourred, "Vulkan: no Vulkan 1.3 device that GL runs on can present to the window");
        return false;
    }

    static bool createdevice()
    {
        uint32_t numavail = 0;
        vkEnumerateDeviceExtensionProperties(physdev, NULL, &numavail, NULL);
        vector<VkExtensionProperties> avail;
        vkEnumerateDeviceExtensionProperties(physdev, NULL, &numavail, avail.pad(numavail));

        vector<const char *> exts;
        exts.add(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
        exts.add(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
        exts.add(VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);
        loopv(exts) if(!hasext(avail, exts[i]))
        {
            conoutf(colourred, "Vulkan: the device is missing %s", exts[i]);
            return false;
        }
        vector<const char *> ngxexts;
        ngx_deviceextensions(ngxexts);
        loopv(ngxexts) if(hasext(avail, ngxexts[i]))
        {
            if(!hassurfacemaint && !strcmp(ngxexts[i], VK_EXT_SWAPCHAIN_MAINTENANCE_1_EXTENSION_NAME)) continue;
            addext(exts, ngxexts[i]);
        }
        hasrt = hasext(avail, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME) && hasext(avail, VK_KHR_RAY_QUERY_EXTENSION_NAME) && hasext(avail, VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
        haslowlatency = hasext(avail, VK_NV_LOW_LATENCY_2_EXTENSION_NAME);
        haspresentid = hasext(avail, VK_KHR_PRESENT_ID_EXTENSION_NAME);
        VkPhysicalDevicePresentIdFeaturesKHR pidfeats = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_ID_FEATURES_KHR };

        VkPhysicalDeviceAccelerationStructureFeaturesKHR asfeats = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR };
        VkPhysicalDeviceRayQueryFeaturesKHR rqfeats = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR };
        VkPhysicalDeviceVulkan13Features f13 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES };
        VkPhysicalDeviceVulkan12Features f12 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
        VkPhysicalDeviceVulkan11Features f11 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES };
        VkPhysicalDeviceFeatures2 feats = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
        feats.pNext = &f11;
        f11.pNext = &f12;
        f12.pNext = &f13;
        if(hasrt)
        {
            f13.pNext = &asfeats;
            asfeats.pNext = &rqfeats;
        }
        if(haspresentid)
        {
            pidfeats.pNext = feats.pNext;
            feats.pNext = &pidfeats;
        }
        vkGetPhysicalDeviceFeatures2(physdev, &feats);
        if(haspresentid && !pidfeats.presentId)
        {
            haspresentid = false;
            feats.pNext = pidfeats.pNext;
        }
        if(haspresentid) exts.add(VK_KHR_PRESENT_ID_EXTENSION_NAME);
        if(!haspresentid) haslowlatency = false; // its markers name frames by present id
        if(haslowlatency) exts.add(VK_NV_LOW_LATENCY_2_EXTENSION_NAME);
        hashdrmetadata = hasext(avail, VK_EXT_HDR_METADATA_EXTENSION_NAME);
        if(hashdrmetadata) exts.add(VK_EXT_HDR_METADATA_EXTENSION_NAME);
        if(!f13.synchronization2 || !f12.timelineSemaphore)
        {
            conoutf(colourred, "Vulkan: the device lacks synchronization2 or timeline semaphores");
            return false;
        }
        if(hasrt && (!asfeats.accelerationStructure || !rqfeats.rayQuery || !f12.bufferDeviceAddress)) hasrt = false;
        if(hasrt)
        {
            exts.add(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
            exts.add(VK_KHR_RAY_QUERY_EXTENSION_NAME);
            exts.add(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
            // only what's used: no capture/replay, multi-device or indirect builds
            asfeats.accelerationStructureCaptureReplay = VK_FALSE;
            asfeats.accelerationStructureIndirectBuild = VK_FALSE;
            asfeats.accelerationStructureHostCommands = VK_FALSE;
        }
        else f13.pNext = NULL;
        // the core features stay as the device reports them, apart from ones that cost something when on
        feats.features.robustBufferAccess = VK_FALSE;
        f12.bufferDeviceAddressCaptureReplay = VK_FALSE;
        f12.bufferDeviceAddressMultiDevice = VK_FALSE;
        f13.robustImageAccess = VK_FALSE;

        float priority = 1;
        VkDeviceQueueCreateInfo qci = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
        qci.queueFamilyIndex = queuefamily;
        qci.queueCount = 1;
        qci.pQueuePriorities = &priority;
        VkDeviceCreateInfo dci = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
        dci.pNext = &feats;
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;
        dci.enabledExtensionCount = exts.length();
        dci.ppEnabledExtensionNames = exts.getbuf();
        VKCHECK(vkCreateDevice(physdev, &dci, NULL, &device));
        vkGetDeviceQueue(device, queuefamily, 0, &vk::queue);
        vkGetPhysicalDeviceMemoryProperties(physdev, &memprops);

        if(haslowlatency)
        {
            SetLatencySleepModeNV = (PFN_vkSetLatencySleepModeNV)vkGetDeviceProcAddr(device, "vkSetLatencySleepModeNV");
            LatencySleepNV = (PFN_vkLatencySleepNV)vkGetDeviceProcAddr(device, "vkLatencySleepNV");
            SetLatencyMarkerNV = (PFN_vkSetLatencyMarkerNV)vkGetDeviceProcAddr(device, "vkSetLatencyMarkerNV");
            GetLatencyTimingsNV = (PFN_vkGetLatencyTimingsNV)vkGetDeviceProcAddr(device, "vkGetLatencyTimingsNV");
            VkSemaphoreTypeCreateInfo tci = { VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO };
            tci.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
            VkSemaphoreCreateInfo semci = { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
            semci.pNext = &tci;
            if(!SetLatencySleepModeNV || !LatencySleepNV || !SetLatencyMarkerNV || !GetLatencyTimingsNV || vkCreateSemaphore(device, &semci, NULL, &latencysem) != VK_SUCCESS)
                haslowlatency = false;
            latencyvalue = 0;
        }

        SetHdrMetadataEXT = hashdrmetadata ? (PFN_vkSetHdrMetadataEXT)vkGetDeviceProcAddr(device, "vkSetHdrMetadataEXT") : NULL;
        GetMemoryFdKHR = (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(device, "vkGetMemoryFdKHR");
        GetSemaphoreFdKHR = (PFN_vkGetSemaphoreFdKHR)vkGetDeviceProcAddr(device, "vkGetSemaphoreFdKHR");
        if(!GetMemoryFdKHR || !GetSemaphoreFdKHR) return false;

        VkCommandPoolCreateInfo pci = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
        pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pci.queueFamilyIndex = queuefamily;
        VKCHECK(vkCreateCommandPool(device, &pci, NULL, &cmdpool));
        return true;
    }

    // frame generation inputs GL fills in each frame
    static matrix4 fgprevvp, fgprevscreen;
    static vec fgprevpos(0, 0, 0);
    static float fgprevfov = 0;
    static bool fgprevvalid = false;

    static void storematrix(float *dst, const matrix4 &m) { memcpy(dst, m.a.v, 16*sizeof(float)); }

    // models' own motion: each moving entity's parts as they were placed the frame before
    struct partmotion
    {
        const void *owner, *part;
        matrix4 cur, prev;
        int frame;
    };
    static vector<partmotion> partmotions;
    static int motionframe = 0;
    static matrix4 velcurvp, velprevvp;
    static bool velvpvalid = false;

    // the main view as the G-buffer was drawn with it
    matrix4 viewcam, viewproj, viewworld;
    vec viewpos(0, 0, 0);
    static vec viewdir(0, 1, 0), viewright(1, 0, 0), viewup(0, 0, 1);
    static bool viewfresh = false; // snapshotted since the last capture
}

using namespace vk;

// DLSS frame generation: how many frames to make between rendered frames, 0 is off
// -1 picks the count itself: the fewest that bring the presented rate up to the display's refresh rate
VARF(IDF_PERSIST, dlssg, -1, 0, MAXFGGEN, { if(vkactive) framesdirty = true; });
ICOMMANDV(0, vkpresenting, vkactive ? 1 : 0);
VAR(0, dbgvkfg, 0, 0, 2); // prints the range of the frame generation inputs each frame, 2 also saves motion.png once

SDL_Window *vk_createwindow(const char *caption, int x, int y, int w, int h, int flags)
{
    return SDL_CreateWindow(caption, x, y, w, h, (flags&~SDL_WINDOW_OPENGL)|SDL_WINDOW_VULKAN);
}

SDL_Window *vk_createglwindow()
{
    // GL only ever draws into framebuffer objects, its window is never shown
    glwindow = SDL_CreateWindow("", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, 64, 64, SDL_WINDOW_OPENGL|SDL_WINDOW_HIDDEN);
    return glwindow;
}

void vk_destroyglwindow()
{
    if(glwindow) SDL_DestroyWindow(glwindow);
    glwindow = NULL;
}

bool vk_init()
{
    extern int vsync, vsynctear;
    wantvsync = vsync;
    wanttear = vsynctear;
    SDL_AtomicSet(&swapdirty, 1);
    if(!queuelock) queuelock = SDL_CreateMutex();
    if(!swaplock) swaplock = SDL_CreateMutex();
    if(!presentlock)
    {
        presentlock = SDL_CreateMutex();
        presentcond = SDL_CreateCond();
        presentdone = SDL_CreateCond();
    }
    if(!loadglinterop() || !createinstance()) { vk_shutdown(); return false; }
    if(!SDL_Vulkan_CreateSurface(screen, instance, &surface))
    {
        conoutf(colourred, "Vulkan: no surface for the window: %s", SDL_GetError());
        vk_shutdown();
        return false;
    }
    if(!pickdevice() || !createdevice() || !createswapchain() || !startpresent()) { vk_shutdown(); return false; }
    vkactive = true;
    gvelocity = 1;
    conoutf(colourwhite, "Vulkan: %s (Vulkan %u.%u.%u), presenting %ux%u %s%s", devprops.deviceName,
        VK_VERSION_MAJOR(devprops.apiVersion), VK_VERSION_MINOR(devprops.apiVersion), VK_VERSION_PATCH(devprops.apiVersion),
        swapextent.width, swapextent.height, presentmodename(presentmode), hasrt ? ", ray queries on" : "");
    ngx_init();
    return true;
}

void vk_cleanupgl()
{
    if(!device) return;
    vkrt_cleanupgl();
    vk_dlsscleanupgl();
    destroyframes();
    if(glBindFramebufferreal && glBindFramebuffer_ == bindframebuffer) glBindFramebuffer_ = glBindFramebufferreal;
}

void vk_shutdown()
{
    if(device)
    {
        stoppresent();
        vkDeviceWaitIdle(device);
        vkrt_shutdown();
        vk_dlssshutdown();
        ngx_shutdown();
        destroyhdrpipeline();
        if(latencysem) vkDestroySemaphore(device, latencysem, NULL);
        latencysem = VK_NULL_HANDLE;
        haslowlatency = haspresentid = false;
        destroyswapchain();
        if(cmdpool) vkDestroyCommandPool(device, cmdpool, NULL);
        cmdpool = VK_NULL_HANDLE;
        vkDestroyDevice(device, NULL);
        device = VK_NULL_HANDLE;
    }
    if(surface) vkDestroySurfaceKHR(instance, surface, NULL);
    surface = VK_NULL_HANDLE;
    if(messenger)
    {
        PFN_vkDestroyDebugUtilsMessengerEXT destroy = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
        if(destroy) destroy(instance, messenger, NULL);
        messenger = VK_NULL_HANDLE;
    }
    if(instance) vkDestroyInstance(instance, NULL);
    instance = VK_NULL_HANDLE;
    physdev = VK_NULL_HANDLE;
    vkactive = false;
    hdrscreen = swaphdr = swapwanthdr = false;
    SDL_AtomicSet(&hdrsurface, 0);
    SDL_AtomicSet(&hdrfailed, 0);
    SetHdrMetadataEXT = NULL;
    gvelocity = 0;
    velvpvalid = false;
    partmotions.setsize(0);
    hasrt = false;
}

void vk_getdrawablesize(int &w, int &h)
{
    SDL_Vulkan_GetDrawableSize(screen, &w, &h);
}

static int vkscreenw = 0, vkscreenh = 0;

static void makeframes()
{
    destroyframes();
    if(vkscreenw <= 0 || vkscreenh <= 0) return;
    if(!createframes(vkscreenw, vkscreenh, gw, gh)) fatal("Vulkan: could not make the %dx%d screen images", vkscreenw, vkscreenh);
    fgprevvalid = false;
}

void vk_resize(int w, int h)
{
    if(!vkactive || !glBindFramebuffer_ || !glGenFramebuffers_) return; // GL's framebuffer functions load after the first window setup
    if(glBindFramebuffer_ != bindframebuffer)
    {
        glBindFramebufferreal = glBindFramebuffer_;
        glBindFramebuffer_ = bindframebuffer;
    }
    SDL_AtomicSet(&swapdirty, 1);
    if(numframes && w == vkscreenw && h == vkscreenh) return;
    vkscreenw = w;
    vkscreenh = h;
    makeframes();
    vk_beginframe();
}

void vk_beginframe()
{
    if(!numframes || !framedone) return;
    if(framesdirty)
    {
        framesdirty = false;
        makeframes();
        if(!numframes) return;
    }
    if(grabname[0]) savegrabs();
    curframe = (curframe+1)%numframes;
    waitslot(curframe);
    screenframe &f = frames[curframe];
    if(f.glwait)
    {
        GLuint texs[4];
        GLenum layouts[4];
        int n = f.textures(texs);
        loopi(n) layouts[i] = GL_LAYOUT_COLOR_ATTACHMENT_EXT;
        glwait(f.vkdone, n, texs, layouts);
        f.glwait = false;
    }
    f.hashudless = f.hasfg = false;
    screenfbo = f.fbo;
    framedone = false;
    glBindFramebufferreal(GL_FRAMEBUFFER, screenfbo);
}

void vk_capturehudless()
{
    if(!vkactive || framedone) return;
    screenframe &f = frames[curframe];
    if(!f.hudlessfbo) return;
    GLboolean scissor = glIsEnabled(GL_SCISSOR_TEST);
    if(scissor) glDisable(GL_SCISSOR_TEST);
    glBindFramebufferreal(GL_READ_FRAMEBUFFER, screenfbo);
    glBindFramebufferreal(GL_DRAW_FRAMEBUFFER, f.hudlessfbo);
    glBlitFramebuffer_(0, 0, f.img.w, f.img.h, 0, 0, f.hudless.w, f.hudless.h, GL_COLOR_BUFFER_BIT, GL_NEAREST);
    glBindFramebufferreal(GL_FRAMEBUFFER, screenfbo);
    if(scissor) glEnable(GL_SCISSOR_TEST);
    f.hashudless = true;
}

void vk_snapshotview()
{
    if(!vkactive) return;
    extern matrix4 nojittermatrix;
    viewcam = cammatrix;
    viewproj = nojittermatrix;
    viewworld = worldmatrix;
    viewpos = camera1->o;
    viewdir = camdir;
    viewright = camright;
    viewup = camup;
    viewfresh = true;

    // a new frame for the models' motion
    motionframe++;
    matrix4 vp;
    vp.muld(viewproj, viewcam);
    velprevvp = velvpvalid ? velcurvp : vp;
    velcurvp = vp;
    velvpvalid = true;
    loopvrev(partmotions) if(partmotions[i].frame < motionframe-2) partmotions.removeunordered(i);
}

// a model part's placement this frame and last, as clip space matrices for the velocity target; parts of no entity
// only move with the camera
void vk_partmotion(const void *owner, const void *part, const matrix4 &world, matrix4 &cur, matrix4 &prev)
{
    cur.mul(velcurvp, world);
    if(!owner) { prev.mul(velprevvp, world); return; }
    loopv(partmotions)
    {
        partmotion &p = partmotions[i];
        if(p.owner != owner || p.part != part) continue;
        if(p.frame != motionframe)
        {
            p.prev = p.frame == motionframe-1 ? p.cur : world;
            p.cur = world;
            p.frame = motionframe;
        }
        prev.mul(velprevvp, p.prev);
        return;
    }
    partmotion &p = partmotions.add();
    p.owner = owner;
    p.part = part;
    p.cur = p.prev = world;
    p.frame = motionframe;
    prev.mul(velprevvp, world);
}

// motion to where each pixel was the frame before, in pixels, and depth as the depth buffer holds it, from the main
// view's G-buffer into fbo's two attachments; screenm and prevscreen take world positions to this and last frame's pixels
bool vk_drawmotion(GLuint fbo, int w, int h, const matrix4 &screenm, const matrix4 &prevscreen)
{
    // lazy shaders only exist once something asks for them by name
    Shader *vectorshader = useshaderbyname("dlssgvectors"), *stillshader = useshaderbyname("dlssgstill");
    if(!vectorshader || !stillshader) return false;
    matrix4 curm, prevm;
    curm.muld(screenm, viewworld);
    prevm.muld(prevscreen, viewworld);
    GLint oldfbo = 0, oldviewport[4];
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &oldfbo);
    glGetIntegerv(GL_VIEWPORT, oldviewport);
    glBindFramebufferreal(GL_FRAMEBUFFER, fbo);
    bool stencil = gdepthformat && useavatarmask();
    // full screen passes: the scene's depth must stay as it is
    GLboolean depthtest = glIsEnabled(GL_DEPTH_TEST), depthwrite = GL_TRUE;
    glGetBooleanv(GL_DEPTH_WRITEMASK, &depthwrite);
    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);
    glViewport(0, 0, w, h);
    static const GLenum drawbufs[2] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
    glDrawBuffers_(2, drawbufs);
    vectorshader->set();
    LOCALPARAM(curmatrix, curm);
    LOCALPARAM(prevmatrix, prevm);
    glActiveTexture_(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_RECTANGLE, gvelocitytex);
    glActiveTexture_(GL_TEXTURE0);
    if(msaalight) glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, msdepthtex);
    else glBindTexture(GL_TEXTURE_RECTANGLE, gdepthtex);
    screenquad(w, h);
    glstencilstate oldstencil;
    if(stencil)
    {
        // the first person model moves with the camera, so on screen it holds still
        oldstencil.save();
        bindgdepth(); // only for its stencil, which marks the first person model
        glDrawBuffers_(1, drawbufs);
        glEnable(GL_STENCIL_TEST);
        glStencilFunc(GL_EQUAL, 0x40, 0x40);
        glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);
        stillshader->set();
        screenquad(w, h);
        oldstencil.restore();
        glFramebufferRenderbuffer_(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, 0);
        glFramebufferRenderbuffer_(GL_FRAMEBUFFER, GL_STENCIL_ATTACHMENT, GL_RENDERBUFFER, 0);
    }
    glDrawBuffers_(1, drawbufs);
    if(depthtest) glEnable(GL_DEPTH_TEST);
    glDepthMask(depthwrite);
    glBindFramebufferreal(GL_FRAMEBUFFER, oldfbo);
    glViewport(oldviewport[0], oldviewport[1], oldviewport[2], oldviewport[3]);
    return true;
}

void vk_capturefg()
{
    if(!vkactive || framedone || drawtex) return;
    bool want = dlssg != 0 && ngx_fgmaxgen() > 0;
    if(want != fgimages)
    {
        framesdirty = true; // the screen images get made again with or without the inputs at the next frame
        return;
    }
    if(!fgimages) return;
    screenframe &f = frames[curframe];
    if(f.mvec.w != gw || f.mvec.h != gh)
    {
        framesdirty = true; // the G-buffer changed size, the inputs follow at the next frame
        return;
    }
    if(!viewfresh) return; // no main view this frame
    viewfresh = false;
    // unjittered: frame generation wants motion without the anti-aliasing jitter
    matrix4 vp, screenm;
    vp.muld(viewproj, viewcam);
    screenm.identity();
    screenm.settranslation(0.5f*gw, 0.5f*gh, 0.5f);
    screenm.setscale(0.5f*gw, 0.5f*gh, 0.5f);
    screenm.muld(vp);
    bool reset = !fgprevvalid || fgprevpos.dist(viewpos) > 64 || fabs(fgprevfov - curfov) > 0.01f;
    if(reset)
    {
        fgprevvp = vp;
        fgprevscreen = screenm;
    }

    if(!vk_drawmotion(f.fgfbo, gw, gh, screenm, fgprevscreen)) return;
    glBindFramebufferreal(GL_FRAMEBUFFER, f.fgfbo);
    static const GLenum drawbufs[2] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
    if(dbgvkfg)
    {
        float *motion = new float[gw*gh*2], *depth = new float[gw*gh];
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glReadBuffer(GL_COLOR_ATTACHMENT0);
        glReadPixels(0, 0, gw, gh, GL_RG, GL_FLOAT, motion);
        glReadBuffer(GL_COLOR_ATTACHMENT1);
        glReadPixels(0, 0, gw, gh, GL_RED, GL_FLOAT, depth);
        glReadBuffer(GL_COLOR_ATTACHMENT0);
        float mx[2] = { 1e9f, 1e9f }, Mx[2] = { -1e9f, -1e9f }, dmin = 1e9f, dmax = -1e9f;
        double sum[2] = { 0, 0 };
        loopi(gw*gh)
        {
            loopj(2) { mx[j] = min(mx[j], motion[i*2+j]); Mx[j] = max(Mx[j], motion[i*2+j]); sum[j] += motion[i*2+j]; }
            dmin = min(dmin, depth[i]);
            dmax = max(dmax, depth[i]);
        }
        int c = (gh/2)*gw + gw/2;
        conoutf(colourwhite, "fg: motion x %.2f..%.2f (mean %.2f) y %.2f..%.2f (mean %.2f), centre %.2f %.2f, depth %.5f..%.5f centre %.5f%s",
            mx[0], Mx[0], sum[0]/(gw*gh), mx[1], Mx[1], sum[1]/(gw*gh), motion[c*2], motion[c*2+1], dmin, dmax, depth[c], reset ? " reset" : "");
        if(dbgvkfg > 1)
        {
            // the motion as a picture, grey where nothing moves
            ImageData img(gw, gh, 3);
            loopi(gw*gh)
            {
                img.data[i*3+0] = uchar(clamp(128 + motion[i*2]*4, 0.0f, 255.0f));
                img.data[i*3+1] = uchar(clamp(128 + motion[i*2+1]*4, 0.0f, 255.0f));
                img.data[i*3+2] = 128;
            }
            saveimage("motion.png", img, IFMT_PNG, 9, true);
            dbgvkfg = 0;
        }
        delete[] motion;
        delete[] depth;
    }
    glBindFramebufferreal(GL_FRAMEBUFFER, screenfbo);

    // camera data, in Vulkan's clip space: depth from 0 to 1, and with GL's rows from the bottom the y axis already
    // runs the way Vulkan's does
    matrix4 zfix(vec4(1, 0, 0, 0), vec4(0, 1, 0, 0), vec4(0, 0, 0.5f, 0), vec4(0, 0, 0.5f, 1)), invzfix, viewtoclip, cliptoview, invvp, cliptoprev, prevtoclip;
    invzfix.invert(zfix);
    viewtoclip.muld(zfix, viewproj);
    cliptoview.invert(viewtoclip);
    invvp.invert(vp);
    cliptoprev.muld(zfix, fgprevvp);
    cliptoprev.muld(invvp);
    cliptoprev.muld(invzfix);
    prevtoclip.invert(cliptoprev);
    fgcamera &c = f.camera;
    storematrix(c.viewtoclip, viewtoclip);
    storematrix(c.cliptoview, cliptoview);
    storematrix(c.cliptoprevclip, cliptoprev);
    storematrix(c.prevcliptoclip, prevtoclip);
    c.pos[0] = viewpos.x; c.pos[1] = viewpos.y; c.pos[2] = viewpos.z;
    c.up[0] = viewup.x; c.up[1] = viewup.y; c.up[2] = viewup.z;
    c.right[0] = viewright.x; c.right[1] = viewright.y; c.right[2] = viewright.z;
    c.fwd[0] = viewdir.x; c.fwd[1] = viewdir.y; c.fwd[2] = viewdir.z;
    c.nearplane = nearplane;
    c.farplane = farplane;
    c.fov = fovy*RAD;
    c.aspect = aspect;
    c.reset = reset;

    fgprevvp = vp;
    fgprevscreen = screenm;
    fgprevpos = viewpos;
    fgprevfov = curfov;
    fgprevvalid = true;
    f.hasfg = true;
}

void vk_present()
{
    if(!numframes || framedone) return;
    screenframe &f = frames[curframe];
    int maxgen = 0;
    while(maxgen < MAXFGGEN && f.interp[maxgen].image) maxgen++;
    maxgen = min(maxgen, ngx_fgmaxgen());
    int numgen = f.hasfg ? (dlssg < 0 ? autogen(maxgen) : min(int(dlssg), maxgen)) : 0;
    if(!f.hasfg) fgprevvalid = false; // the next frame generation starts over

    GLuint texs[4];
    GLenum layouts[4];
    int n = f.textures(texs);
    loopi(n) layouts[i] = GL_LAYOUT_SHADER_READ_ONLY_EXT;
    glsignal(f.glready, n, texs, layouts);
    glFlush(); // Vulkan can only wait on the signal once GL has submitted it
    framedone = true;
    f.glwait = true;

    bool hdrframe = f.hdr.image != VK_NULL_HANDLE;
    if(numgen && !ngx_fgready(f.img.w, f.img.h, f.mvec.w, f.mvec.h, hdrframe ? HDRFORMAT : VK_FORMAT_R8G8B8A8_UNORM)) numgen = 0;

    vkWaitForFences(device, 1, &f.fence, VK_TRUE, UINT64_MAX);
    vkResetFences(device, 1, &f.fence);
    VkCommandBuffer cmd = f.cmd;
    vkResetCommandBuffer(cmd, 0);
    VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);
    sharedimage *imgs[4] = { &f.img, &f.hudless, &f.mvec, &f.depth };
    // the HDR conversion reads the frame and its copy without the HUD as storage images
    VkImageLayout readlayouts[4] = { VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    if(hdrframe) readlayouts[0] = readlayouts[1] = VK_IMAGE_LAYOUT_GENERAL;
    loopi(4) if(imgs[i]->image)
        imagebarrier(cmd, imgs[i]->image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, readlayouts[i],
            VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_SHADER_READ_BIT|VK_ACCESS_2_TRANSFER_READ_BIT,
            VK_QUEUE_FAMILY_EXTERNAL, queuefamily);
    if(hdrframe)
    {
        bool hudlesshdr = f.hashudless && numgen && f.hdrhudless.image; // frame generation wants it in HDR too
        // the copy without the HUD is in the descriptors whether written this frame or not, so it is in their layout
        deviceimage *outs[2] = { &f.hdr, f.hdrhudless.image ? &f.hdrhudless : NULL };
        // the presenting thread's last copy from these images came before
        loopi(2) if(outs[i]) imagebarrier(cmd, outs[i]->image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_NONE, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        float knee, soft;
        hdrcurve(knee, soft);
        hdrparams p = {
            { min(float(hdrpaperwhite), float(hdrpeak)), float(hdrrange), knee, soft },
            { hdrgamma, f.hashudless ? 1.0f : 0.0f, hudlesshdr ? 1.0f : 0.0f, 0 }
        };
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, hdrpipe);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, hdrlayout, 0, 1, &f.hdrset, 0, NULL);
        vkCmdPushConstants(cmd, hdrlayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(p), &p);
        vkCmdDispatch(cmd, (f.hdr.w + 7)/8, (f.hdr.h + 7)/8, 1);
        // ready to read, for frame generation and the presenting thread
        loopi(2) if(outs[i]) imagebarrier(cmd, outs[i]->image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_SHADER_READ_BIT|VK_ACCESS_2_TRANSFER_READ_BIT);
    }
    if(numgen)
    {
        loopi(numgen) imagebarrier(cmd, f.interp[i].image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_NONE, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_SHADER_WRITE_BIT|VK_ACCESS_2_SHADER_READ_BIT);
        fginputs in;
        in.backbuffer = hdrframe ? f.hdr : f.img.device();
        in.hashudless = f.hashudless;
        if(f.hashudless) in.hudless = hdrframe ? f.hdrhudless : f.hudless.device();
        in.hdr = hdrframe;
        in.mvec = &f.mvec;
        in.depth = &f.depth;
        in.interp = f.interp;
        in.camera = f.camera;
        if(!fglastgen) in.camera.reset = true; // its last pair of frames is stale after a frame without it
        if(!ngx_fgevaluate(cmd, in, numgen)) numgen = 0;
        loopi(numgen) imagebarrier(cmd, f.interp[i].image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
    }
    // the inputs go straight back to GL, the screen image after the presenting thread copied it, or now when what's
    // presented is its HDR version
    loopi(4) if(imgs[i]->image && (i || hdrframe))
        imagebarrier(cmd, imgs[i]->image, readlayouts[i], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_NONE, VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
            queuefamily, VK_QUEUE_FAMILY_EXTERNAL);
    vkEndCommandBuffer(cmd);

    VkSemaphoreSubmitInfo waitgl = { VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO };
    waitgl.semaphore = f.glready.sem;
    waitgl.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    VkSemaphoreSubmitInfo signaldone = { VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO };
    signaldone.semaphore = evaltimeline;
    signaldone.value = ++evalcounter;
    signaldone.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    VkCommandBufferSubmitInfo cbi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
    cbi.commandBuffer = cmd;
    VkSubmitInfo2 si = { VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
    si.waitSemaphoreInfoCount = 1;
    si.pWaitSemaphoreInfos = &waitgl;
    si.commandBufferInfoCount = 1;
    si.pCommandBufferInfos = &cbi;
    si.signalSemaphoreInfoCount = 1;
    si.pSignalSemaphoreInfos = &signaldone;
    VkResult r = submit(si, f.fence, latencyframe);
    if(r != VK_SUCCESS) fatal("Vulkan: submitting the frame failed: %s", resultname(r));

    fglastgen = numgen;

    // rendered frames come this far apart, the generated ones go out in between
    double now = seconds();
    struct timespec cpu;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &cpu);
    double cputime = cpu.tv_sec + cpu.tv_nsec*1e-9;
    if(lastjobtime > 0)
    {
        double dt = clamp(now - lastjobtime, 0.0, 0.1), work = clamp(cputime - lastcputime, 0.001, 0.1);
        jobinterval = jobinterval > 0 ? jobinterval + (dt - jobinterval)*0.1 : dt;
        workinterval = workinterval > 0 ? workinterval + (work - workinterval)*0.05 : work;
    }
    lastjobtime = now;
    lastcputime = cputime;

    latencymarker(VK_LATENCY_MARKER_RENDERSUBMIT_END_NV, latencyframe);

    presentjob job;
    job.frameid = latencyframe;
    job.slot = curframe;
    job.numgen = numgen;
    job.evalvalue = evalcounter;
    job.interval = jobinterval;
    SDL_LockMutex(presentlock);
    slotbusy[curframe] = true;
    presentjobs.add(job);
    SDL_CondSignal(presentcond);
    SDL_UnlockMutex(presentlock);
}

void vk_setvsync(int vsync, int tear)
{
    if(vsync == wantvsync && tear == wanttear) return;
    wantvsync = vsync;
    wanttear = tear;
    SDL_AtomicSet(&swapdirty, 1);
}

bool vk_hastear() { return hasrelaxed; }

// 1 when the display takes HDR10 (the window's surface offers it), 2 when it's on
ICOMMANDV(0, hdravailable, vkactive && SDL_AtomicGet(&hdrsurface) && !SDL_AtomicGet(&hdrfailed) ? (hdrscreen ? 2 : 1) : 0);

void vk_updatehdr()
{
    if(!vkactive) return;
    SDL_AtomicSet(&windowactive, !minimized && screen && (SDL_GetWindowFlags(screen)&SDL_WINDOW_INPUT_FOCUS) ? 1 : 0);
    int err = SDL_AtomicSet(&swaperror, 0);
    if(err) conoutf(colourred, "Vulkan: could not make the swapchain: %s (%d)", resultname(VkResult(err)), err);
    if(SDL_AtomicSet(&swapstalled, 0)) conoutf(colouryellow, "Vulkan: the display stopped taking frames, the swapchain was made again");
    if(SDL_AtomicGet(&hdrfailed) && hdroutput)
    {
        // off for good, so the next start doesn't run into it again
        setvar("hdroutput", 0, true);
        conoutf(colourred, "Vulkan: the display wouldn't show HDR output, it is off now");
    }
    bool want = hdroutput && SDL_AtomicGet(&hdrsurface) && !SDL_AtomicGet(&hdrfailed);
    if(want == hdrscreen) return;
    hdrscreen = want;
    // the screen images and every buffer the tonemapped frame passes through change format
    framesdirty = true;
    cleanupscale(); // the next setup makes the post-processing buffers again
    vk_dlsscleanupgl();
    if(dbgvk) conoutf(colourwhite, "Vulkan: HDR output %s", hdrscreen ? "on" : "off");
}

// 0 off, 1 on, 2 on with the GPU's clocks kept up (Reflex's boost)
VARF(IDF_PERSIST, lowlatency, 0, 1, 2, { SDL_AtomicSet(&sleepmodedirty, 1); SDL_AtomicSet(&swapdirty, 1); });
ICOMMANDV(0, lowlatencyavailable, vkactive ? (haslowlatency ? 2 : 1) : 0);

// at the start of a frame, before input: sleeps for as long as the frame would otherwise wait to be shown
static double latencysleeptime = 0, latencywaittime = 0; // smoothed, in seconds

void vk_latencysleep()
{
    if(!vkactive) return;
    latencyframe++;
    double sleepstart = seconds();
    if(haslowlatency)
    {
        SDL_LockMutex(swaplock);
        VkSwapchainKHR sc = swapchain;
        if(sc && SDL_AtomicGet(&sleepmodedirty))
        {
            SDL_AtomicSet(&sleepmodedirty, 0);
            VkLatencySleepModeInfoNV mi = { VK_STRUCTURE_TYPE_LATENCY_SLEEP_MODE_INFO_NV };
            mi.lowLatencyMode = lowlatency ? VK_TRUE : VK_FALSE;
            mi.lowLatencyBoost = lowlatency > 1 ? VK_TRUE : VK_FALSE;
            mi.minimumIntervalUs = 0;
            SetLatencySleepModeNV(device, sc, &mi);
        }
        bool slept = false;
        if(sc && lowlatency)
        {
            VkLatencySleepInfoNV si = { VK_STRUCTURE_TYPE_LATENCY_SLEEP_INFO_NV };
            si.signalSemaphore = latencysem;
            si.value = ++latencyvalue;
            slept = LatencySleepNV(device, sc, &si) == VK_SUCCESS;
        }
        SDL_UnlockMutex(swaplock);
        if(slept)
        {
            VkSemaphoreWaitInfo wi = { VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO };
            wi.semaphoreCount = 1;
            wi.pSemaphores = &latencysem;
            wi.pValues = &latencyvalue;
            vkWaitSemaphores(device, &wi, 100000000); // never more than 100ms, whatever the driver thinks
        }
    }
    double waitstart = seconds();
    latencysleeptime += (waitstart - sleepstart - latencysleeptime)*0.05;
    // the wait for a free screen image happens here rather than once the frame is simulated, so it isn't latency; without
    // the driver's sleep a frame also waits for the last one to go out, so frames don't queue up
    if(lowlatency && numframes)
    {
        if(haslowlatency) waitslot((curframe+1)%numframes);
        else loopi(numframes) waitslot(i);
    }
    latencywaittime += (seconds() - waitstart - latencywaittime)*0.05;
    latencymarker(VK_LATENCY_MARKER_SIMULATION_START_NV, latencyframe);
}

void vk_latencymark(int marker) { latencymarker(VkLatencyMarkerNV(marker), latencyframe); }

// what the driver measured over the last frames, from the start of the frame to it being shown
static void vklatency()
{
    if(!vkactive || !haslowlatency) { conoutf(colourwhite, "Latency: no driver timings%s", vkactive && lowlatency ? ", frames wait for the last one instead" : ""); return; }
    VkLatencyTimingsFrameReportNV reports[64];
    loopi(64) { memset(&reports[i], 0, sizeof(reports[i])); reports[i].sType = VK_STRUCTURE_TYPE_LATENCY_TIMINGS_FRAME_REPORT_NV; }
    VkGetLatencyMarkerInfoNV gi = { VK_STRUCTURE_TYPE_GET_LATENCY_MARKER_INFO_NV };
    gi.timingCount = 64;
    gi.pTimings = reports;
    SDL_LockMutex(swaplock);
    if(swapchain) GetLatencyTimingsNV(device, swapchain, &gi);
    else gi.timingCount = 0;
    SDL_UnlockMutex(swaplock);
    // frames from the last second or so with every mark in order
    double todone = 0, topresent = 0, sim = 0;
    int n = 0;
    loopi(int(gi.timingCount))
    {
        const VkLatencyTimingsFrameReportNV &r = reports[i];
        if(r.presentID + 200 < latencyframe || r.presentID > latencyframe) continue;
        if(!r.simStartTimeUs || !r.simEndTimeUs || !r.presentEndTimeUs || !r.gpuRenderEndTimeUs) continue;
        if(r.simEndTimeUs < r.simStartTimeUs || r.presentEndTimeUs < r.simStartTimeUs || r.gpuRenderEndTimeUs < r.simStartTimeUs) continue;
        if(r.presentEndTimeUs - r.simStartTimeUs > 500000) continue;
        todone += r.gpuRenderEndTimeUs - r.simStartTimeUs;
        topresent += r.presentEndTimeUs - r.simStartTimeUs;
        sim += r.simEndTimeUs - r.simStartTimeUs;
        n++;
    }
    if(!n) { conoutf(colourwhite, "Latency: no complete frames yet"); result("0 0 0 0"); return; }
    defformatstring(str, "%.2f %.2f %.2f %d", todone/n/1000, topresent/n/1000, sim/n/1000, n);
    conoutf(colourwhite, "Latency: %.2f ms from a frame starting to the GPU finishing it, %.2f ms to it being presented, simulation %.2f ms, slept %.2f ms and waited %.2f ms before it (%d frames, low latency %s)",
        todone/n/1000, topresent/n/1000, sim/n/1000, latencysleeptime*1000, latencywaittime*1000, n, lowlatency ? (lowlatency > 1 ? "on with boost" : "on") : "off");
    result(str);
}
COMMAND(0, vklatency, "");

// development aid like frametimes, for what was presented: "presents average-ms worst-ms 1%-low-fps" since the last call
static void getpresenttimes()
{
    if(!vkactive) { result("0 0 0 0"); return; }
    SDL_LockMutex(presentlock);
    vector<float> times;
    loopv(presenttimes) times.add(presenttimes[i]);
    presenttimes.setsize(0);
    SDL_UnlockMutex(presentlock);
    int n = times.length();
    if(!n) { result("0 0 0 0"); return; }
    double total = 0;
    loopi(n) total += times[i];
    times.sort();
    float worst = times.last(), low = times[clamp(int(n*0.99f), 0, n-1)];
    defformatstring(str, "%d %.3f %.3f %.1f", n, total/n, worst, low > 0 ? 1000.0f/low : 0.0f);
    result(str);
}
COMMANDN(0, presenttimes, getpresenttimes, "");

ICOMMAND(0, vkgrab, "si", (char *name, int *num),
{
    if(!vkactive) { conoutf(colourred, "Vulkan: not presenting"); return; }
    SDL_LockMutex(presentlock);
    copystring(grabname, *name ? name : "vkgrab");
    grabtaken = 0;
    grabwant = max(*num, 1);
    SDL_UnlockMutex(presentlock);
});

ICOMMAND(0, vkinfo, "", (),
{
    if(!vkactive) { conoutf(colourwhite, "Vulkan: off, GL presents"); return; }
    conoutf(colourwhite, "Vulkan: %s, swapchain %ux%u %s, screen images %dx%d, ray queries %s, frame generation %s, HDR %s", devprops.deviceName,
        swapextent.width, swapextent.height, presentmodename(presentmode), numframes ? frames[0].img.w : 0, numframes ? frames[0].img.h : 0,
        hasrt ? "on" : "off", fgimages ? tempformatstring("%s%dx (inputs %dx%d, the game could render %.0f fps)", dlssg < 0 ? "auto " : "", (dlssg < 0 ? fglastgen : int(dlssg))+1, frames[0].mvec.w, frames[0].mvec.h, workinterval > 0 ? 1/workinterval : 0.0) : "off",
        swaphdr ? tempformatstring("on (HDR10, peak %.0f nits, paper white %.0f nits)", float(hdrpeak), float(hdrpaperwhite)) : (SDL_AtomicGet(&hdrsurface) ? "off" : "not offered by the display"));
});
