// Vulkan objects shared by the Vulkan side of the engine: presenting (vk.cpp), DLSS (ngx.cpp), ray tracing
#ifndef ENGINE_VKINT_H
#define ENGINE_VKINT_H

#include <vulkan/vulkan.h>

namespace vk
{
    extern VkInstance instance;
    extern VkPhysicalDevice physdev;
    extern VkDevice device;
    extern uint32_t queuefamily;
    extern VkQueue queue;
    extern VkCommandPool cmdpool;
    extern VkPhysicalDeviceProperties devprops;
    extern bool hasrt; // acceleration structures and ray queries are enabled

    extern int findmemorytype(uint32_t bits, VkMemoryPropertyFlags flags);
    // the presenting thread submits too, so every queue submission goes through this lock
    extern void lockqueue();
    extern void unlockqueue();
    extern VkResult submit(const VkSubmitInfo2 &si, VkFence fence, uint64_t frameid = 0);
    extern uint64_t currentframeid(); // the frame's latency id, for submit
    extern void queueidle();
    extern void imagebarrier(VkCommandBuffer cmd, VkImage image, VkImageLayout oldlayout, VkImageLayout newlayout,
        VkPipelineStageFlags2 srcstage, VkAccessFlags2 srcaccess, VkPipelineStageFlags2 dststage, VkAccessFlags2 dstaccess,
        uint32_t srcqueue = VK_QUEUE_FAMILY_IGNORED, uint32_t dstqueue = VK_QUEUE_FAMILY_IGNORED, VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT);
    extern VkCommandBuffer begincommands();          // one-off commands on the main thread
    extern bool endcommands(VkCommandBuffer cmd);    // submits them and waits
    extern const char *resultname(VkResult r);

    // an image Vulkan allocated that GL draws into or reads as a texture
    struct deviceimage;
    struct sharedimage
    {
        VkImage image = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkFormat format = VK_FORMAT_UNDEFINED;
        int w = 0, h = 0;
        GLuint glmemory = 0, gltex = 0;
        GLenum gltarget = 0, glformat = 0;

        inline deviceimage device() const; // its Vulkan side, as a device image (not owning it)
    };
    extern bool createsharedimage(sharedimage &s, int w, int h, VkFormat format, GLenum glformat, GLenum gltarget = GL_TEXTURE_2D);
    extern void destroysharedimage(sharedimage &s);

    // a semaphore GL signals and Vulkan waits on, or the other way around
    struct sharedsemaphore
    {
        VkSemaphore sem = VK_NULL_HANDLE;
        GLuint glsem = 0;
    };
    extern bool createsharedsemaphore(sharedsemaphore &s);
    extern void destroysharedsemaphore(sharedsemaphore &s);
    extern void glsignal(sharedsemaphore &s, int numtex, const GLuint *texs, const GLenum *layouts);
    extern void glwait(sharedsemaphore &s, int numtex, const GLuint *texs, const GLenum *layouts);

    // NGX (DLSS) hooks around instance and device creation, see ngx.cpp
    extern void ngx_instanceextensions(vector<const char *> &exts);
    extern void ngx_deviceextensions(vector<const char *> &exts);
    extern void ngx_init();
    extern void ngx_shutdown();

    // a Vulkan-only image, for frame generation output
    struct deviceimage
    {
        VkImage image = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkFormat format = VK_FORMAT_UNDEFINED;
        int w = 0, h = 0;
    };
    inline deviceimage sharedimage::device() const
    {
        deviceimage d;
        d.image = image;
        d.view = view;
        d.format = format;
        d.w = w;
        d.h = h;
        return d;
    }
    extern bool createdeviceimage(deviceimage &d, int w, int h, VkFormat format, VkImageUsageFlags usage);
    extern void destroydeviceimage(deviceimage &d);

    // camera data for DLSS frame generation, matrices as NGX wants them: Vulkan clip space (depth 0 to 1),
    // stored the way matrix4 stores them, which reads as row-major with the vector on the left
    struct fgcamera
    {
        float viewtoclip[16], cliptoview[16], cliptoprevclip[16], prevcliptoclip[16];
        float pos[3], up[3], right[3], fwd[3];
        float nearplane, farplane, fov, aspect;
        bool reset;
    };

    // GL stencil state, kept around passes that test the avatar mask in the middle of the frame
    struct glstencilstate
    {
        GLboolean enabled;
        GLint func, ref, mask, fail, zfail, zpass;
        void save()
        {
            enabled = glIsEnabled(GL_STENCIL_TEST);
            glGetIntegerv(GL_STENCIL_FUNC, &func);
            glGetIntegerv(GL_STENCIL_REF, &ref);
            glGetIntegerv(GL_STENCIL_VALUE_MASK, &mask);
            glGetIntegerv(GL_STENCIL_FAIL, &fail);
            glGetIntegerv(GL_STENCIL_PASS_DEPTH_FAIL, &zfail);
            glGetIntegerv(GL_STENCIL_PASS_DEPTH_PASS, &zpass);
        }
        void restore()
        {
            glStencilFunc(func, ref, mask);
            glStencilOp(fail, zfail, zpass);
            if(enabled) glEnable(GL_STENCIL_TEST); else glDisable(GL_STENCIL_TEST);
        }
    };

    // the main view as its G-buffer was drawn (vk_snapshotview)
    extern matrix4 viewcam, viewproj, viewworld;
    extern vec viewpos;

    #define MAXFGGEN 5 // DLSS multi frame generation makes at most this many frames per rendered frame

    struct fginputs
    {
        deviceimage backbuffer, hudless; // in HDR10 with hdr, hudless only with hashudless
        bool hashudless = false, hdr = false;
        sharedimage *mvec, *depth;
        deviceimage *interp; // one per generated frame
        fgcamera camera;
    };
    // DLSS super resolution: color, motion and depth at the render size, output at the display size
    struct srinputs
    {
        sharedimage *color, *mvec, *depth, *output;
        float jitterx, jittery; // render pixels, the way the motion vectors run
        bool reset;
    };
    extern bool ngx_srready(int w, int h, int outw, int outh, int quality);
    extern bool ngx_srevaluate(VkCommandBuffer cmd, const srinputs &in);
    extern void ngx_srrelease();
    // DLSS ray reconstruction: the lit frame before tonemapping, noisy ray traced light and all, with what it is made
    // of, denoised and anti-aliased at the same size
    struct rrinputs
    {
        sharedimage *color, *diffuse, *specular, *normals, *mvec, *depth, *output; // normals carry roughness in w
        float jitterx, jittery;
        float worldtoview[16], viewtoclip[16];
        bool reset;
    };
    extern bool ngx_rrready(int w, int h);
    extern bool ngx_rrevaluate(VkCommandBuffer cmd, const rrinputs &in);
    extern void ngx_rrrelease();

    extern int ngx_fgmaxgen(); // 0 without frame generation
    extern bool ngx_fgready(int w, int h, int renderw, int renderh, VkFormat format); // creates the feature for these sizes and frame format if needed
    extern bool ngx_fgevaluate(VkCommandBuffer cmd, const fginputs &in, int numgen);
    extern void ngx_fgrelease();
}

#endif
