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

VARF(IDF_INIT, vulkan, 0, 1, 1, initwarning("Vulkan presenting", INIT_RESET));
VAR(IDF_INIT, vkvalidation, 0, 0, 1); // Khronos validation layer, for development
VAR(0, dbgvk, 0, 0, 1);

bool vkactive = false;
extern int dlssg;

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

    VkResult submit(const VkSubmitInfo2 &si, VkFence fence)
    {
        lockqueue();
        VkResult r = vkQueueSubmit2(vk::queue, 1, &si, fence);
        unlockqueue();
        return r;
    }

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

    static VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    static VkFormat swapformat = VK_FORMAT_UNDEFINED;
    static VkExtent2D swapextent = { 0, 0 };
    static VkPresentModeKHR presentmode = VK_PRESENT_MODE_FIFO_KHR;
    static vector<VkImage> swapimages;
    static vector<VkSemaphore> swapready;
    static SDL_atomic_t swapdirty;
    static bool hasrelaxed = false, cangrab = false;
    static int wantvsync = 0, wanttear = 0;

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

        VkSurfaceCapabilitiesKHR caps;
        VKCHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physdev, surface, &caps));
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
        bool found = false;
        loopj(int(sizeof(preferred)/sizeof(preferred[0])))
        {
            loopv(formats) if(formats[i].format == preferred[j] && formats[i].colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
            {
                format = formats[i];
                found = true;
                break;
            }
            if(found) break;
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
            conoutf(colourred, "Vulkan: the swapchain can't be copied into");
            return false;
        }
        uint32_t count = max(caps.minImageCount + 1, 3u);
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
        VkSwapchainKHR newswapchain = VK_NULL_HANDLE;
        VKCHECK(vkCreateSwapchainKHR(device, &sci, NULL, &newswapchain));
        destroyswapchain();
        swapchain = newswapchain;
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
        uint64_t evalvalue;
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
    static double lastjobtime = 0, jobinterval = 0;
    static vector<float> presenttimes; // milliseconds between presents, for presenttimes
    static double lastpresent = 0;

    // vkgrab: the presenting thread copies presented images back, the main thread saves them
    static string grabname = "";
    static int grabwant = 0, grabtaken = 0;
    struct grabbedimage { ImageData *image; int index; };
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
            int w = swapextent.width, h = swapextent.height;
            ImageData *image = new ImageData(w, h, 3);
            const uchar *src = (const uchar *)data;
            bool bgr = swapformat == VK_FORMAT_B8G8R8A8_UNORM || swapformat == VK_FORMAT_B8G8R8A8_SRGB;
            loopi(w*h)
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
            conoutf(colourwhite, "Vulkan: saved presented image %s", name);
            delete done[i].image;
        }
    }

    // copies src (GL oriented, in TRANSFER_SRC layout unless real) into the next swapchain image and presents it;
    // for the rendered image it also hands the frame's images back to GL
    static void presentimage(VkImage src, int srcw, int srch, screenframe *real)
    {
        presentcmd &p = presentcmds[presentnext];
        presentnext = (presentnext+1)%PRESENTRING;
        vkWaitForFences(device, 1, &p.fence, VK_TRUE, UINT64_MAX);

        if(SDL_AtomicGet(&swapdirty) && !createswapchain()) fprintf(stderr, "Vulkan: could not remake the swapchain\n");
        uint32_t index = UINT32_MAX;
        if(swapchain)
        {
            VkResult r = vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, p.acquired, VK_NULL_HANDLE, &index);
            if(r == VK_ERROR_OUT_OF_DATE_KHR)
            {
                index = UINT32_MAX;
                if(createswapchain() && swapchain) r = vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, p.acquired, VK_NULL_HANDLE, &index);
            }
            if(r == VK_SUBOPTIMAL_KHR) SDL_AtomicSet(&swapdirty, 1);
            else if(r != VK_SUCCESS) index = UINT32_MAX;
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
        if(real)
        {
            // every image of the frame goes back to GL, in the layout GL draws in
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
            VkResult pr = vkQueuePresentKHR(vk::queue, &pi);
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
                presentimage(f.interp[i].image, f.interp[i].w, f.interp[i].h, NULL);
            }
            if(job.numgen) sleepuntil(start + job.numgen*step);
            presentimage(f.img.image, f.img.w, f.img.h, &f);

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
            destroysharedsemaphore(f.glready);
            destroysharedsemaphore(f.vkdone);
            if(f.cmd) vkFreeCommandBuffers(device, cmdpool, 1, &f.cmd);
            if(f.fence) vkDestroyFence(device, f.fence, NULL);
            f = screenframe();
        }
        numframes = 0;
        fgimages = false;
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
        bool wantfg = dlssg && ngx_fgmaxgen() > 0 && fgw > 0 && fgh > 0;
        loopi(MAXVKFRAMES)
        {
            screenframe &f = frames[i];
            numframes = i+1;
            if(!createsharedimage(f.img, w, h, VK_FORMAT_R8G8B8A8_UNORM, GL_RGBA8) || !createframebuffer(f.fbo, f.img)) return false;
            if(wantfg)
            {
                if(!createsharedimage(f.hudless, w, h, VK_FORMAT_R8G8B8A8_UNORM, GL_RGBA8) || !createframebuffer(f.hudlessfbo, f.hudless)) return false;
                if(!createsharedimage(f.mvec, fgw, fgh, VK_FORMAT_R16G16_SFLOAT, GL_RG16F) ||
                   !createsharedimage(f.depth, fgw, fgh, VK_FORMAT_R32_SFLOAT, GL_R32F) ||
                   !createframebuffer(f.fgfbo, f.mvec, &f.depth)) return false;
                loopj(min(int(dlssg), MAXFGGEN))
                    if(!createdeviceimage(f.interp[j], w, h, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_STORAGE_BIT|VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_TRANSFER_SRC_BIT))
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
        if(dbgvk) conoutf(colourwhite, "Vulkan: screen images %dx%d%s", w, h, fgimages ? tempformatstring(", frame generation inputs %dx%d", fgw, fgh) : "");
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
        loopv(ngxexts) if(hasext(avail, ngxexts[i])) addext(exts, ngxexts[i]);
        hasrt = hasext(avail, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME) && hasext(avail, VK_KHR_RAY_QUERY_EXTENSION_NAME) && hasext(avail, VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);

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
        vkGetPhysicalDeviceFeatures2(physdev, &feats);
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

    // the main view as the G-buffer was drawn with it
    matrix4 viewcam, viewproj, viewworld;
    vec viewpos(0, 0, 0);
    static vec viewdir(0, 1, 0), viewright(1, 0, 0), viewup(0, 0, 1);
    static bool viewfresh = false; // snapshotted since the last capture
}

using namespace vk;

// DLSS frame generation: how many frames to make between rendered frames, 0 is off
VARF(IDF_PERSIST, dlssg, 0, 0, MAXFGGEN, { if(vkactive) framesdirty = true; });
VAR(0, dbgvkfg, 0, 0, 1); // prints the range of the frame generation inputs each frame

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
        ngx_shutdown();
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
    if(!vkactive || !fgimages || framedone) return;
    screenframe &f = frames[curframe];
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
}

void vk_capturefg()
{
    if(!vkactive || framedone || drawtex) return;
    bool want = dlssg && ngx_fgmaxgen() > 0;
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

    // lazy shaders only exist once something asks for them by name
    Shader *vectorshader = useshaderbyname("dlssgvectors"), *stillshader = useshaderbyname("dlssgstill");
    if(!vectorshader || !stillshader) return;

    // motion to where each pixel was last frame, in pixels, and depth as the depth buffer holds it
    matrix4 curm, prevm;
    curm.muld(screenm, viewworld);
    prevm.muld(fgprevscreen, viewworld);
    GLint oldviewport[4];
    glGetIntegerv(GL_VIEWPORT, oldviewport);
    glBindFramebufferreal(GL_FRAMEBUFFER, f.fgfbo);
    bool stencil = gdepthformat && useavatarmask();
    // full screen passes: the scene's depth must stay as it is
    GLboolean depthtest = glIsEnabled(GL_DEPTH_TEST), depthwrite = GL_TRUE;
    glGetBooleanv(GL_DEPTH_WRITEMASK, &depthwrite);
    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);
    glViewport(0, 0, gw, gh);
    static const GLenum drawbufs[2] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
    glDrawBuffers_(2, drawbufs);
    vectorshader->set();
    LOCALPARAM(curmatrix, curm);
    LOCALPARAM(prevmatrix, prevm);
    glActiveTexture_(GL_TEXTURE0);
    if(msaalight) glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, msdepthtex);
    else glBindTexture(GL_TEXTURE_RECTANGLE, gdepthtex);
    screenquad(gw, gh);
    glstencilstate oldstencil;
    if(stencil)
    {
        oldstencil.save();
        bindgdepth(); // only for its stencil, which marks the first person model
    }
    if(stencil)
    {
        // the first person model moves with the camera, so on screen it holds still
        glDrawBuffers_(1, drawbufs);
        glEnable(GL_STENCIL_TEST);
        glStencilFunc(GL_EQUAL, 0x40, 0x40);
        glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);
        stillshader->set();
        screenquad(gw, gh);
        oldstencil.restore();
        glFramebufferRenderbuffer_(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, 0);
        glFramebufferRenderbuffer_(GL_FRAMEBUFFER, GL_STENCIL_ATTACHMENT, GL_RENDERBUFFER, 0);
    }
    if(depthtest) glEnable(GL_DEPTH_TEST);
    glDepthMask(depthwrite);
    glDrawBuffers_(1, drawbufs);
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
        delete[] motion;
        delete[] depth;
    }
    glBindFramebufferreal(GL_FRAMEBUFFER, screenfbo);
    glViewport(oldviewport[0], oldviewport[1], oldviewport[2], oldviewport[3]);

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
    int numgen = f.hasfg ? min(int(dlssg), ngx_fgmaxgen()) : 0;
    if(!f.hasfg) fgprevvalid = false; // the next frame generation starts over

    GLuint texs[4];
    GLenum layouts[4];
    int n = f.textures(texs);
    loopi(n) layouts[i] = GL_LAYOUT_SHADER_READ_ONLY_EXT;
    glsignal(f.glready, n, texs, layouts);
    glFlush(); // Vulkan can only wait on the signal once GL has submitted it
    framedone = true;
    f.glwait = true;

    if(numgen && !ngx_fgready(f.img.w, f.img.h, f.mvec.w, f.mvec.h)) numgen = 0;

    vkWaitForFences(device, 1, &f.fence, VK_TRUE, UINT64_MAX);
    vkResetFences(device, 1, &f.fence);
    VkCommandBuffer cmd = f.cmd;
    vkResetCommandBuffer(cmd, 0);
    VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);
    sharedimage *imgs[4] = { &f.img, &f.hudless, &f.mvec, &f.depth };
    loopi(4) if(imgs[i]->image)
        imagebarrier(cmd, imgs[i]->image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_SHADER_READ_BIT|VK_ACCESS_2_TRANSFER_READ_BIT,
            VK_QUEUE_FAMILY_EXTERNAL, queuefamily);
    if(numgen)
    {
        loopi(numgen) imagebarrier(cmd, f.interp[i].image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_NONE, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_SHADER_WRITE_BIT|VK_ACCESS_2_SHADER_READ_BIT);
        fginputs in;
        in.backbuffer = &f.img;
        in.hudless = f.hashudless ? &f.hudless : NULL;
        in.mvec = &f.mvec;
        in.depth = &f.depth;
        in.interp = f.interp;
        in.camera = f.camera;
        if(!ngx_fgevaluate(cmd, in, numgen)) numgen = 0;
        loopi(numgen) imagebarrier(cmd, f.interp[i].image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
    }
    // the inputs go straight back to GL, the screen image after the presenting thread copied it
    loopi(3) if(imgs[i+1]->image)
        imagebarrier(cmd, imgs[i+1]->image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
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
    VkResult r = submit(si, f.fence);
    if(r != VK_SUCCESS) fatal("Vulkan: submitting the frame failed: %s", resultname(r));

    // rendered frames come this far apart, the generated ones go out in between
    double now = seconds();
    if(lastjobtime > 0)
    {
        double dt = clamp(now - lastjobtime, 0.0, 0.1);
        jobinterval = jobinterval > 0 ? jobinterval + (dt - jobinterval)*0.1 : dt;
    }
    lastjobtime = now;

    presentjob job;
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
    conoutf(colourwhite, "Vulkan: %s, swapchain %ux%u %s, screen images %dx%d, ray queries %s, frame generation %s", devprops.deviceName,
        swapextent.width, swapextent.height, presentmodename(presentmode), numframes ? frames[0].img.w : 0, numframes ? frames[0].img.h : 0,
        hasrt ? "on" : "off", fgimages ? tempformatstring("%dx (inputs %dx%d)", int(dlssg)+1, frames[0].mvec.w, frames[0].mvec.h) : "off");
});
