// DLSS super resolution as the anti-aliasing step of the GL renderer: GL tonemaps the frame into an image shared with
// Vulkan, along with motion vectors and depth from the G-buffer, NGX writes the anti-aliased frame into another shared
// image, and GL carries on from it with the post effects and the HUD. The projection follows a Halton sequence of
// sub-pixel offsets that DLSS gathers over frames.
#include "engine.h"
#include "vkint.h"

extern int dlssavailable;
VAR(IDF_PERSIST, dlss, 0, 0, 1); // DLSS anti-aliasing, 1 is DLAA: at the full resolution
FVAR(0, dlssjitterscale, -2, 1, 2); // development aid: scales the jitter offsets DLSS is told, to check their direction

namespace vk
{
    static sharedimage srcolor, srmvec, srdepth, sroutput;
    static GLuint srcolorfbo = 0, srvecfbo = 0, sroutfbo = 0;
    static sharedsemaphore srready, srdone;
    static VkCommandBuffer srcmds[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    static VkFence srfences[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    static int srcmdindex = 0, srimagew = 0, srimageh = 0;
    static matrix4 srprevscreen;
    static vec srprevpos(0, 0, 0);
    static bool srprevvalid = false, srfailed = false;
    static vec2 srjitter(0, 0);
    static uint srframe = 0;

    static float halton(int i, int base)
    {
        float f = 1, r = 0;
        for(; i > 0; i /= base)
        {
            f /= base;
            r += f*(i%base);
        }
        return r;
    }

    static void destroysrimages()
    {
        if(!srimagew) return;
        glFinish();
        queueidle();
        ngx_srrelease();
        GLuint *fbos[3] = { &srcolorfbo, &srvecfbo, &sroutfbo };
        loopi(3) if(*fbos[i]) { glDeleteFramebuffers_(1, fbos[i]); *fbos[i] = 0; }
        destroysharedimage(srcolor);
        destroysharedimage(srmvec);
        destroysharedimage(srdepth);
        destroysharedimage(sroutput);
        destroysharedsemaphore(srready);
        destroysharedsemaphore(srdone);
        srimagew = srimageh = 0;
        srprevvalid = false;
    }

    static bool makefbo(GLuint &fbo, sharedimage &a, sharedimage *b = NULL)
    {
        glGenFramebuffers_(1, &fbo);
        glBindFramebuffer_(GL_FRAMEBUFFER, fbo);
        glFramebufferTexture2D_(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, a.gltarget, a.gltex, 0);
        if(b) glFramebufferTexture2D_(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, b->gltarget, b->gltex, 0);
        return glCheckFramebufferStatus_(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
    }

    static bool createsrimages(int w, int h)
    {
        srimagew = w;
        srimageh = h;
        // the tonemapped frame, in 16 bits with HDR output
        VkFormat colorvk = hdrscreen ? VK_FORMAT_R16G16B16A16_UNORM : VK_FORMAT_R8G8B8A8_UNORM;
        GLenum colorgl = screenformat(GL_RGBA8);
        if(!createsharedimage(srcolor, w, h, colorvk, colorgl) ||
           !createsharedimage(srmvec, w, h, VK_FORMAT_R16G16_SFLOAT, GL_RG16F) ||
           !createsharedimage(srdepth, w, h, VK_FORMAT_R32_SFLOAT, GL_R32F) ||
           !createsharedimage(sroutput, w, h, colorvk, colorgl) ||
           !createsharedsemaphore(srready) || !createsharedsemaphore(srdone))
            return false;
        bool ok = makefbo(srcolorfbo, srcolor) && makefbo(srvecfbo, srmvec, &srdepth) && makefbo(sroutfbo, sroutput);
        glBindFramebuffer_(GL_FRAMEBUFFER, renderfbo);
        if(!ok) return false;
        if(!srcmds[0]) loopi(2)
        {
            VkCommandBufferAllocateInfo cai = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
            cai.commandPool = cmdpool;
            cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            cai.commandBufferCount = 1;
            if(vkAllocateCommandBuffers(device, &cai, &srcmds[i]) != VK_SUCCESS) return false;
            VkFenceCreateInfo fci = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
            fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
            if(vkCreateFence(device, &fci, NULL, &srfences[i]) != VK_SUCCESS) return false;
        }
        // every image starts on GL's side
        VkCommandBuffer cmd = begincommands();
        if(!cmd) return false;
        sharedimage *imgs[4] = { &srcolor, &srmvec, &srdepth, &sroutput };
        loopi(4) imagebarrier(cmd, imgs[i]->image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE, VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE, queuefamily, VK_QUEUE_FAMILY_EXTERNAL);
        return endcommands(cmd);
    }
}

using namespace vk;

bool vk_dlssactive() { return dlss && vkactive && dlssavailable && !srfailed && !msaasamples && !drawtex; }

// once a frame: the next sub-pixel offset, 16 of them for DLAA
void vk_dlssnextjitter()
{
    srframe++;
    int i = srframe%16 + 1;
    srjitter = vec2(halton(i, 2) - 0.5f, halton(i, 3) - 0.5f);
}

vec2 vk_dlssjitter() { return srjitter; }

GLuint vk_dlssinput()
{
    if(srimagew != vieww || srimageh != viewh)
    {
        destroysrimages();
        if(!createsrimages(vieww, viewh))
        {
            conoutf(colourred, "DLSS: could not make the %dx%d super resolution images", vieww, viewh);
            destroysrimages();
            srfailed = true;
            return renderfbo;
        }
    }
    return srcolorfbo;
}

void vk_dlssapply(GLuint outfbo)
{
    if(!srimagew || !ngx_srready(srimagew, srimageh, srimagew, srimageh, 5)) // 5 is NGX's DLAA quality
    {
        // no DLSS this frame: the tonemapped frame goes out as it is
        if(srimagew)
        {
            glBindFramebuffer_(GL_READ_FRAMEBUFFER, srcolorfbo);
            glBindFramebuffer_(GL_DRAW_FRAMEBUFFER, outfbo);
            glBlitFramebuffer_(0, 0, srimagew, srimageh, 0, 0, srimagew, srimageh, GL_COLOR_BUFFER_BIT, GL_NEAREST);
            glBindFramebuffer_(GL_FRAMEBUFFER, outfbo);
        }
        srfailed = true;
        return;
    }

    // motion to the last frame and depth, unjittered like DLSS wants them
    matrix4 vp, screenm;
    vp.muld(viewproj, viewcam);
    screenm.identity();
    screenm.settranslation(0.5f*srimagew, 0.5f*srimageh, 0.5f);
    screenm.setscale(0.5f*srimagew, 0.5f*srimageh, 0.5f);
    screenm.muld(vp);
    bool reset = !srprevvalid || srprevpos.dist(viewpos) > 64;
    if(reset) srprevscreen = screenm;
    vk_drawmotion(srvecfbo, srimagew, srimageh, screenm, srprevscreen);
    srprevscreen = screenm;
    srprevpos = viewpos;
    srprevvalid = true;

    GLuint texs[4] = { srcolor.gltex, srmvec.gltex, srdepth.gltex, sroutput.gltex };
    GLenum layouts[4] = { GL_LAYOUT_SHADER_READ_ONLY_EXT, GL_LAYOUT_SHADER_READ_ONLY_EXT, GL_LAYOUT_SHADER_READ_ONLY_EXT, GL_LAYOUT_GENERAL_EXT };
    glsignal(srready, 4, texs, layouts);
    glFlush();

    VkCommandBuffer cmd = srcmds[srcmdindex];
    VkFence fence = srfences[srcmdindex];
    srcmdindex = (srcmdindex+1)%2;
    vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
    vkResetFences(device, 1, &fence);
    vkResetCommandBuffer(cmd, 0);
    VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);
    sharedimage *inputs[3] = { &srcolor, &srmvec, &srdepth };
    loopi(3) imagebarrier(cmd, inputs[i]->image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_SHADER_READ_BIT,
        VK_QUEUE_FAMILY_EXTERNAL, queuefamily);
    imagebarrier(cmd, sroutput.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_SHADER_WRITE_BIT|VK_ACCESS_2_SHADER_READ_BIT,
        VK_QUEUE_FAMILY_EXTERNAL, queuefamily);
    srinputs in;
    in.color = &srcolor;
    in.mvec = &srmvec;
    in.depth = &srdepth;
    in.output = &sroutput;
    in.jitterx = srjitter.x*dlssjitterscale;
    in.jittery = srjitter.y*dlssjitterscale;
    in.reset = reset;
    bool ok = ngx_srevaluate(cmd, in);
    loopi(3) imagebarrier(cmd, inputs[i]->image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_NONE, VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
        queuefamily, VK_QUEUE_FAMILY_EXTERNAL);
    imagebarrier(cmd, sroutput.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
        queuefamily, VK_QUEUE_FAMILY_EXTERNAL);
    vkEndCommandBuffer(cmd);

    VkSemaphoreSubmitInfo wait = { VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO };
    wait.semaphore = srready.sem;
    wait.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    VkSemaphoreSubmitInfo signal = { VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO };
    signal.semaphore = srdone.sem;
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
    if(submit(si, fence, currentframeid()) != VK_SUCCESS) ok = false;
    GLenum backlayouts[4] = { GL_LAYOUT_COLOR_ATTACHMENT_EXT, GL_LAYOUT_COLOR_ATTACHMENT_EXT, GL_LAYOUT_COLOR_ATTACHMENT_EXT, GL_LAYOUT_SHADER_READ_ONLY_EXT };
    glwait(srdone, 4, texs, backlayouts);

    // the anti-aliased frame, or the plain one if DLSS let it down
    glBindFramebuffer_(GL_READ_FRAMEBUFFER, ok ? sroutfbo : srcolorfbo);
    glBindFramebuffer_(GL_DRAW_FRAMEBUFFER, outfbo);
    glBlitFramebuffer_(0, 0, srimagew, srimageh, 0, 0, srimagew, srimageh, GL_COLOR_BUFFER_BIT, GL_NEAREST);
    glBindFramebuffer_(GL_FRAMEBUFFER, outfbo);
    if(!ok) srfailed = true;
}

void vk_dlsscleanupgl()
{
    if(!device) return;
    destroysrimages();
}

void vk_dlssshutdown()
{
    if(!device) return;
    loopi(2)
    {
        if(srfences[i]) vkDestroyFence(device, srfences[i], NULL);
        if(srcmds[i]) vkFreeCommandBuffers(device, cmdpool, 1, &srcmds[i]);
        srfences[i] = VK_NULL_HANDLE;
        srcmds[i] = VK_NULL_HANDLE;
    }
    srfailed = false;
}
