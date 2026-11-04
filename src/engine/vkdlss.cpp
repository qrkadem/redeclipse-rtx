// DLSS super resolution as the anti-aliasing step of the GL renderer: GL tonemaps the frame into an image shared with
// Vulkan, along with motion vectors and depth from the G-buffer, NGX writes the anti-aliased frame into another shared
// image, and GL carries on from it with the post effects and the HUD. The projection follows a Halton sequence of
// sub-pixel offsets that DLSS gathers over frames.
#include "engine.h"
#include "vkint.h"

extern int dlssavailable;
VAR(IDF_PERSIST, dlss, 0, 0, 1); // DLSS anti-aliasing, 1 is DLAA: at the full resolution
extern int dlssrravailable;
VAR(IDF_PERSIST, dlssrr, 0, 0, 1); // DLSS ray reconstruction: DLSS denoises the ray traced light and anti-aliases, in place of DLAA and the rays' own filters
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

namespace vk
{
    // ray reconstruction: the lit frame and its guides go to NGX before tonemapping, the result comes back in its place
    static sharedimage rrcolor, rrdiffuse, rrspecular, rrnormals, rroutput;
    static GLuint rrcolorfbo = 0, rrguidefbo = 0, rroutfbo = 0;
    static sharedsemaphore rrready, rrdone;
    static VkCommandBuffer rrcmds[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    static VkFence rrfences[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    static int rrcmdindex = 0, rrimagew = 0, rrimageh = 0;
    static bool rrfailed = false, rrqueried[2] = { false, false };
    static VkQueryPool rrqueries = VK_NULL_HANDLE;
    static float rrms = 0;

    static void destroyrrimages()
    {
        if(!rrimagew) return;
        glFinish();
        queueidle();
        ngx_rrrelease();
        GLuint *fbos[3] = { &rrcolorfbo, &rrguidefbo, &rroutfbo };
        loopi(3) if(*fbos[i]) { glDeleteFramebuffers_(1, fbos[i]); *fbos[i] = 0; }
        sharedimage *imgs[5] = { &rrcolor, &rrdiffuse, &rrspecular, &rrnormals, &rroutput };
        loopi(5) destroysharedimage(*imgs[i]);
        destroysharedsemaphore(rrready);
        destroysharedsemaphore(rrdone);
        rrimagew = rrimageh = 0;
    }

    static bool createrrimages(int w, int h)
    {
        rrimagew = w;
        rrimageh = h;
        if(!createsharedimage(rrcolor, w, h, VK_FORMAT_R16G16B16A16_SFLOAT, GL_RGBA16F) ||
           !createsharedimage(rrdiffuse, w, h, VK_FORMAT_R8G8B8A8_UNORM, GL_RGBA8) ||
           !createsharedimage(rrspecular, w, h, VK_FORMAT_R8G8B8A8_UNORM, GL_RGBA8) ||
           !createsharedimage(rrnormals, w, h, VK_FORMAT_R16G16B16A16_SFLOAT, GL_RGBA16F) ||
           !createsharedimage(rroutput, w, h, VK_FORMAT_R16G16B16A16_SFLOAT, GL_RGBA16F) ||
           !createsharedsemaphore(rrready) || !createsharedsemaphore(rrdone))
            return false;
        bool ok = makefbo(rrcolorfbo, rrcolor) && makefbo(rroutfbo, rroutput);
        if(ok)
        {
            glGenFramebuffers_(1, &rrguidefbo);
            glBindFramebuffer_(GL_FRAMEBUFFER, rrguidefbo);
            sharedimage *guides[3] = { &rrdiffuse, &rrspecular, &rrnormals };
            loopi(3) glFramebufferTexture2D_(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, guides[i]->gltarget, guides[i]->gltex, 0);
            ok = glCheckFramebufferStatus_(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
        }
        glBindFramebuffer_(GL_FRAMEBUFFER, renderfbo);
        if(!ok) return false;
        if(!rrcmds[0]) loopi(2)
        {
            VkCommandBufferAllocateInfo cai = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
            cai.commandPool = cmdpool;
            cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            cai.commandBufferCount = 1;
            if(vkAllocateCommandBuffers(device, &cai, &rrcmds[i]) != VK_SUCCESS) return false;
            VkFenceCreateInfo fci = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
            fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
            if(vkCreateFence(device, &fci, NULL, &rrfences[i]) != VK_SUCCESS) return false;
        }
        VkCommandBuffer cmd = begincommands();
        if(!cmd) return false;
        sharedimage *imgs[5] = { &rrcolor, &rrdiffuse, &rrspecular, &rrnormals, &rroutput };
        loopi(5) imagebarrier(cmd, imgs[i]->image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE, VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE, queuefamily, VK_QUEUE_FAMILY_EXTERNAL);
        return endcommands(cmd);
    }

    // the super resolution images, for their motion vectors and depth
    static bool ensuresrimages(int w, int h)
    {
        if(srimagew == w && srimageh == h) return true;
        destroysrimages();
        if(createsrimages(w, h)) return true;
        conoutf(colourred, "DLSS: could not make the %dx%d images", w, h);
        destroysrimages();
        srfailed = true;
        return false;
    }
}

using namespace vk;
bool vk_dlssrractive() { return dlssrr && vkactive && dlssrravailable && !rrfailed && !msaasamples && !drawtex; }

// once the frame is lit, transparent things and all: DLSS denoises and anti-aliases it in place
bool vk_dlssrrapply()
{
    int w = vieww, h = viewh;
    if(!ensuresrimages(w, h)) return false;
    if(rrimagew != w || rrimageh != h)
    {
        destroyrrimages();
        if(!createrrimages(w, h))
        {
            conoutf(colourred, "DLSS: could not make the %dx%d ray reconstruction images", w, h);
            destroyrrimages();
            rrfailed = true;
            return false;
        }
    }
    Shader *guideshader = useshaderbyname("dlssrrguides");
    if(!guideshader || !ngx_rrready(w, h)) { rrfailed = true; return false; }

    // the lit frame, and the guides from the G-buffer
    glBindFramebuffer_(GL_READ_FRAMEBUFFER, hdrfbo);
    glBindFramebuffer_(GL_DRAW_FRAMEBUFFER, rrcolorfbo);
    glBlitFramebuffer_(0, 0, w, h, 0, 0, w, h, GL_COLOR_BUFFER_BIT, GL_NEAREST);
    glBindFramebuffer_(GL_FRAMEBUFFER, rrguidefbo);
    glViewport(0, 0, w, h);
    static const GLenum drawbufs[3] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2 };
    glDrawBuffers_(3, drawbufs);
    GLboolean depthtest = glIsEnabled(GL_DEPTH_TEST), blend = glIsEnabled(GL_BLEND);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);
    guideshader->set();
    LOCALPARAM(worldmat, viewworld);
    glActiveTexture_(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_RECTANGLE, gcolortex);
    glActiveTexture_(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_RECTANGLE, gnormaltex);
    glActiveTexture_(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_RECTANGLE, gdepthtex);
    glActiveTexture_(GL_TEXTURE0);
    screenquad(w, h);
    glDrawBuffers_(1, drawbufs);
    if(depthtest) glEnable(GL_DEPTH_TEST);
    if(blend) glEnable(GL_BLEND);

    // motion to the last frame and depth, as for super resolution
    matrix4 vp, screenm;
    vp.muld(viewproj, viewcam);
    screenm.identity();
    screenm.settranslation(0.5f*w, 0.5f*h, 0.5f);
    screenm.setscale(0.5f*w, 0.5f*h, 0.5f);
    screenm.muld(vp);
    bool reset = !srprevvalid || srprevpos.dist(viewpos) > 64;
    if(reset) srprevscreen = screenm;
    vk_drawmotion(srvecfbo, w, h, screenm, srprevscreen);
    srprevscreen = screenm;
    srprevpos = viewpos;
    srprevvalid = true;

    GLuint texs[7] = { rrcolor.gltex, rrdiffuse.gltex, rrspecular.gltex, rrnormals.gltex, srmvec.gltex, srdepth.gltex, rroutput.gltex };
    GLenum layouts[7];
    loopi(6) layouts[i] = GL_LAYOUT_SHADER_READ_ONLY_EXT;
    layouts[6] = GL_LAYOUT_GENERAL_EXT;
    glsignal(rrready, 7, texs, layouts);
    glFlush();

    int slot = rrcmdindex;
    VkCommandBuffer cmd = rrcmds[rrcmdindex];
    VkFence fence = rrfences[rrcmdindex];
    rrcmdindex = (rrcmdindex+1)%2;
    vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
    if(!rrqueries)
    {
        VkQueryPoolCreateInfo qpci = { VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
        qpci.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qpci.queryCount = 4;
        vkCreateQueryPool(device, &qpci, NULL, &rrqueries);
    }
    if(rrqueries && rrqueried[slot])
    {
        uint64_t stamps[2];
        if(vkGetQueryPoolResults(device, rrqueries, slot*2, 2, sizeof(stamps), stamps, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT) == VK_SUCCESS)
            rrms += ((stamps[1]-stamps[0])*devprops.limits.timestampPeriod*1e-6f - rrms)*0.05f;
    }
    vkResetFences(device, 1, &fence);
    vkResetCommandBuffer(cmd, 0);
    VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);
    if(rrqueries)
    {
        vkCmdResetQueryPool(cmd, rrqueries, slot*2, 2);
        vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, rrqueries, slot*2);
    }
    sharedimage *inputs[6] = { &rrcolor, &rrdiffuse, &rrspecular, &rrnormals, &srmvec, &srdepth };
    loopi(6) imagebarrier(cmd, inputs[i]->image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_SHADER_READ_BIT,
        VK_QUEUE_FAMILY_EXTERNAL, queuefamily);
    imagebarrier(cmd, rroutput.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_SHADER_WRITE_BIT|VK_ACCESS_2_SHADER_READ_BIT,
        VK_QUEUE_FAMILY_EXTERNAL, queuefamily);
    rrinputs in;
    in.color = &rrcolor;
    in.diffuse = &rrdiffuse;
    in.specular = &rrspecular;
    in.normals = &rrnormals;
    in.mvec = &srmvec;
    in.depth = &srdepth;
    in.output = &rroutput;
    in.jitterx = srjitter.x*dlssjitterscale;
    in.jittery = srjitter.y*dlssjitterscale;
    in.reset = reset;
    // the camera as NGX takes it: Vulkan's clip space, depth from 0 to 1, stored the way matrix4 stores them
    matrix4 zfix(vec4(1, 0, 0, 0), vec4(0, 1, 0, 0), vec4(0, 0, 0.5f, 0), vec4(0, 0, 0.5f, 1)), viewtoclip;
    viewtoclip.muld(zfix, viewproj);
    memcpy(in.worldtoview, viewcam.a.v, sizeof(in.worldtoview));
    memcpy(in.viewtoclip, viewtoclip.a.v, sizeof(in.viewtoclip));
    bool ok = ngx_rrevaluate(cmd, in);
    if(rrqueries)
    {
        vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, rrqueries, slot*2+1);
        rrqueried[slot] = true;
    }
    loopi(6) imagebarrier(cmd, inputs[i]->image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_NONE, VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
        queuefamily, VK_QUEUE_FAMILY_EXTERNAL);
    imagebarrier(cmd, rroutput.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
        queuefamily, VK_QUEUE_FAMILY_EXTERNAL);
    vkEndCommandBuffer(cmd);

    VkSemaphoreSubmitInfo wait = { VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO };
    wait.semaphore = rrready.sem;
    wait.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    VkSemaphoreSubmitInfo signal = { VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO };
    signal.semaphore = rrdone.sem;
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
    GLenum backlayouts[7];
    loopi(6) backlayouts[i] = GL_LAYOUT_COLOR_ATTACHMENT_EXT;
    backlayouts[6] = GL_LAYOUT_SHADER_READ_ONLY_EXT;
    glwait(rrdone, 7, texs, backlayouts);

    // the denoised frame takes the lit one's place, which stays as it was if DLSS let it down
    if(ok)
    {
        glBindFramebuffer_(GL_READ_FRAMEBUFFER, rroutfbo);
        glBindFramebuffer_(GL_DRAW_FRAMEBUFFER, hdrfbo);
        glBlitFramebuffer_(0, 0, w, h, 0, 0, w, h, GL_COLOR_BUFFER_BIT, GL_NEAREST);
    }
    else rrfailed = true;
    glBindFramebuffer_(GL_FRAMEBUFFER, hdrfbo);
    return ok;
}

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
    destroyrrimages();
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
        if(rrfences[i]) vkDestroyFence(device, rrfences[i], NULL);
        if(rrcmds[i]) vkFreeCommandBuffers(device, cmdpool, 1, &rrcmds[i]);
        rrfences[i] = VK_NULL_HANDLE;
        rrcmds[i] = VK_NULL_HANDLE;
    }
    srfailed = rrfailed = false;
    if(rrqueries) vkDestroyQueryPool(device, rrqueries, NULL);
    rrqueries = VK_NULL_HANDLE;
    rrqueried[0] = rrqueried[1] = false;
}

ICOMMAND(0, dlssrrinfo, "", (),
{
    conoutf(colourwhite, "DLSS: ray reconstruction %s, GPU %.2f ms%s", vk_dlssrractive() ? "on" : "off", rrms, rrfailed ? ", failed" : "");
});
