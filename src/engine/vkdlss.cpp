// DLSS for the GL renderer, through NGX on the Vulkan device. GL hands its images over to Vulkan through shared images
// and semaphores, NGX writes its result into another shared image, and GL carries on from there.
//  - DLAA anti-aliases the tonemapped frame at the screen's size, as the anti-aliasing step.
//  - Super resolution draws the view smaller, by the mode's share of the screen, and brings it up to the screen where
//    the engine would otherwise stretch it (doscale).
//  - Ray reconstruction takes the lit frame before tonemapping, with guides from the G-buffer, and denoises the ray
//    traced light in it as it anti-aliases it, at the screen's size or bringing it up to it; the frame is tonemapped
//    from its result.
// The projection follows a Halton sequence of sub-pixel offsets that DLSS gathers over frames, longer the smaller the
// view is drawn, and textures read sharper mips to match the screen's size.
#include "engine.h"
#include "vkint.h"

extern int dlssavailable, dlssrravailable;
// 0 off, 1 DLAA, 2 Quality, 3 Balanced, 4 Performance, 5 Ultra Performance: the last four draw the view at 67%, 58%,
// 50% and 33% of the screen's size and bring it up with DLSS
VARF(IDF_PERSIST, dlss, 0, 0, 5, cleanupgbuffer());
VARF(IDF_PERSIST, dlssrr, 0, 0, 1, cleanupgbuffer()); // DLSS ray reconstruction: DLSS denoises the ray traced light and anti-aliases, in place of DLAA and the rays' own filters
FVAR(0, dlssjitterscale, -2, 1, 2); // development aid: scales the jitter offsets DLSS is told, to check their direction

namespace vk
{
    static const float dlssscales[6] = { 1, 1, 2/3.0f, 0.58f, 0.5f, 1/3.0f };
    static const int dlssqualities[6] = { 5, 5, 2, 1, 0, 3 }; // NGX's DLAA, MaxQuality, Balanced, MaxPerf, UltraPerformance

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

    // one way through NGX: the semaphores GL and Vulkan pass the images with, two command buffers in turn, and GPU time
    struct ngxpass
    {
        sharedsemaphore ready, done;
        VkCommandBuffer cmds[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
        VkFence fences[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
        int index = 0;
        VkQueryPool queries = VK_NULL_HANDLE;
        bool queried[2] = { false, false };
        float ms = 0;

        bool setup()
        {
            if(!ready.sem && !createsharedsemaphore(ready)) return false;
            if(!done.sem && !createsharedsemaphore(done)) return false;
            if(!cmds[0]) loopi(2)
            {
                VkCommandBufferAllocateInfo cai = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
                cai.commandPool = cmdpool;
                cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
                cai.commandBufferCount = 1;
                if(vkAllocateCommandBuffers(device, &cai, &cmds[i]) != VK_SUCCESS) return false;
                VkFenceCreateInfo fci = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
                fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
                if(vkCreateFence(device, &fci, NULL, &fences[i]) != VK_SUCCESS) return false;
            }
            if(!queries)
            {
                VkQueryPoolCreateInfo qpci = { VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
                qpci.queryType = VK_QUERY_TYPE_TIMESTAMP;
                qpci.queryCount = 4;
                if(vkCreateQueryPool(device, &qpci, NULL, &queries) != VK_SUCCESS) queries = VK_NULL_HANDLE;
            }
            return true;
        }

        void cleanupgl()
        {
            destroysharedsemaphore(ready);
            destroysharedsemaphore(done);
        }

        void shutdown()
        {
            loopi(2)
            {
                if(fences[i]) vkDestroyFence(device, fences[i], NULL);
                if(cmds[i]) vkFreeCommandBuffers(device, cmdpool, 1, &cmds[i]);
                fences[i] = VK_NULL_HANDLE;
                cmds[i] = VK_NULL_HANDLE;
            }
            if(queries) vkDestroyQueryPool(device, queries, NULL);
            queries = VK_NULL_HANDLE;
            queried[0] = queried[1] = false;
        }

        // GL hands the inputs and the output over, eval records NGX's work, and GL waits for them back
        template<class F> bool run(sharedimage **inputs, int numinputs, sharedimage *output, F eval)
        {
            GLuint texs[8];
            GLenum layouts[8];
            loopi(numinputs) { texs[i] = inputs[i]->gltex; layouts[i] = GL_LAYOUT_SHADER_READ_ONLY_EXT; }
            texs[numinputs] = output->gltex;
            layouts[numinputs] = GL_LAYOUT_GENERAL_EXT;
            glsignal(ready, numinputs + 1, texs, layouts);
            glFlush();

            int slot = index;
            VkCommandBuffer cmd = cmds[slot];
            VkFence fence = fences[slot];
            index = (index+1)%2;
            vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
            if(queries && queried[slot])
            {
                uint64_t stamps[2];
                if(vkGetQueryPoolResults(device, queries, slot*2, 2, sizeof(stamps), stamps, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT) == VK_SUCCESS)
                    ms += ((stamps[1]-stamps[0])*devprops.limits.timestampPeriod*1e-6f - ms)*0.05f;
            }
            vkResetFences(device, 1, &fence);
            vkResetCommandBuffer(cmd, 0);
            VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
            bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            vkBeginCommandBuffer(cmd, &bi);
            if(queries)
            {
                vkCmdResetQueryPool(cmd, queries, slot*2, 2);
                vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, queries, slot*2);
            }
            loopi(numinputs) imagebarrier(cmd, inputs[i]->image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_SHADER_READ_BIT,
                VK_QUEUE_FAMILY_EXTERNAL, queuefamily);
            imagebarrier(cmd, output->image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_SHADER_WRITE_BIT|VK_ACCESS_2_SHADER_READ_BIT,
                VK_QUEUE_FAMILY_EXTERNAL, queuefamily);
            bool ok = eval(cmd);
            if(queries)
            {
                vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, queries, slot*2+1);
                queried[slot] = true;
            }
            loopi(numinputs) imagebarrier(cmd, inputs[i]->image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_NONE, VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
                queuefamily, VK_QUEUE_FAMILY_EXTERNAL);
            imagebarrier(cmd, output->image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
                queuefamily, VK_QUEUE_FAMILY_EXTERNAL);
            vkEndCommandBuffer(cmd);

            VkSemaphoreSubmitInfo wait = { VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO };
            wait.semaphore = ready.sem;
            wait.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            VkSemaphoreSubmitInfo signal = { VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO };
            signal.semaphore = done.sem;
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
            loopi(numinputs) layouts[i] = GL_LAYOUT_COLOR_ATTACHMENT_EXT;
            layouts[numinputs] = GL_LAYOUT_SHADER_READ_ONLY_EXT;
            glwait(done, numinputs + 1, texs, layouts);
            return ok;
        }
    };
    static ngxpass srpass, rrpass;

    static bool makefbo(GLuint &fbo, sharedimage &a, sharedimage *b = NULL, sharedimage *c = NULL)
    {
        glGenFramebuffers_(1, &fbo);
        glBindFramebuffer_(GL_FRAMEBUFFER, fbo);
        sharedimage *imgs[3] = { &a, b, c };
        loopi(3) if(imgs[i]) glFramebufferTexture2D_(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, imgs[i]->gltarget, imgs[i]->gltex, 0);
        bool ok = glCheckFramebufferStatus_(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
        glBindFramebuffer_(GL_FRAMEBUFFER, renderfbo);
        return ok;
    }

    static void deletefbo(GLuint &fbo)
    {
        if(fbo) glDeleteFramebuffers_(1, &fbo);
        fbo = 0;
    }

    // every shared image starts on GL's side, as GL draws into it
    static bool togl(sharedimage **imgs, int n)
    {
        VkCommandBuffer cmd = begincommands();
        if(!cmd) return false;
        loopi(n) imagebarrier(cmd, imgs[i]->image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE, VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE, queuefamily, VK_QUEUE_FAMILY_EXTERNAL);
        return endcommands(cmd);
    }

    // super resolution: the frame, motion and depth at the view's size, the output at the screen's
    static sharedimage srcolor, srmvec, srdepth, sroutput;
    static GLuint srcolorfbo = 0, srvecfbo = 0, sroutfbo = 0;
    static int srw = 0, srh = 0, sroutw = 0, srouth = 0;
    static bool srhdr = false;
    static matrix4 srprevscreen;
    static vec srprevpos(0, 0, 0);
    static bool srprevvalid = false, srfailed = false, srprepared = false;
    static vec2 srjitter(0, 0);
    static uint srframe = 0;

    static void destroysrimages()
    {
        if(!srw) return;
        glFinish();
        queueidle();
        ngx_srrelease();
        deletefbo(srcolorfbo);
        deletefbo(srvecfbo);
        deletefbo(sroutfbo);
        destroysharedimage(srcolor);
        destroysharedimage(srmvec);
        destroysharedimage(srdepth);
        destroysharedimage(sroutput);
        srpass.cleanupgl();
        srw = srh = sroutw = srouth = 0;
        srprevvalid = false;
    }

    static bool createsrimages(int w, int h, int outw, int outh)
    {
        srw = w;
        srh = h;
        sroutw = outw;
        srouth = outh;
        srhdr = hdrscreen;
        // the tonemapped frame, in 16 bits with HDR output
        VkFormat colorvk = hdrscreen ? VK_FORMAT_R16G16B16A16_UNORM : VK_FORMAT_R8G8B8A8_UNORM;
        GLenum colorgl = screenformat(GL_RGBA8);
        if(!createsharedimage(srcolor, w, h, colorvk, colorgl) ||
           !createsharedimage(srmvec, w, h, VK_FORMAT_R16G16_SFLOAT, GL_RG16F) ||
           !createsharedimage(srdepth, w, h, VK_FORMAT_R32_SFLOAT, GL_R32F) ||
           !createsharedimage(sroutput, outw, outh, colorvk, colorgl) ||
           !srpass.setup())
            return false;
        if(!makefbo(srcolorfbo, srcolor) || !makefbo(srvecfbo, srmvec, &srdepth) || !makefbo(sroutfbo, sroutput)) return false;
        NAMEIMAGE(srcolor, "srcolor");
        NAMEIMAGE(srmvec, "srmvec");
        NAMEIMAGE(srdepth, "srdepth");
        NAMEIMAGE(sroutput, "sroutput");
        NAMECMDS(srpass.cmds, "dlss sr");
        sharedimage *imgs[4] = { &srcolor, &srmvec, &srdepth, &sroutput };
        return togl(imgs, 4);
    }

    static bool ensuresrimages(int w, int h, int outw, int outh)
    {
        if(srw == w && srh == h && sroutw == outw && srouth == outh && srhdr == hdrscreen) return true;
        destroysrimages();
        if(createsrimages(w, h, outw, outh)) return true;
        conoutf(colourred, "DLSS: could not make the %dx%d to %dx%d super resolution images", w, h, outw, outh);
        destroysrimages();
        srfailed = true;
        return false;
    }

    // motion to the last frame and depth from the G-buffer, unjittered like DLSS wants them, at the view's size
    static bool drawmotion(int w, int h)
    {
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
        return reset;
    }
    static bool srreset = false;

    static bool evaluatesr(int quality)
    {
        if(!ngx_srready(srw, srh, sroutw, srouth, quality)) { srfailed = true; return false; }
        sharedimage *inputs[3] = { &srcolor, &srmvec, &srdepth };
        bool ok = srpass.run(inputs, 3, &sroutput, [&](VkCommandBuffer cmd)
        {
            srinputs in;
            in.color = &srcolor;
            in.mvec = &srmvec;
            in.depth = &srdepth;
            in.output = &sroutput;
            in.jitterx = srjitter.x*dlssjitterscale;
            in.jittery = srjitter.y*dlssjitterscale;
            in.reset = srreset;
            return ngx_srevaluate(cmd, in);
        });
        if(!ok) srfailed = true;
        return ok;
    }

    // ray reconstruction: the lit frame and its guides at the view's size, the output at the screen's, and there the
    // frame tonemapped from it when it was brought up
    static sharedimage rrcolor, rrdiffuse, rrspecular, rrnormals, rroutput;
    static GLuint rrcolorfbo = 0, rrguidefbo = 0, rroutfbo = 0, rrfinaltex = 0, rrfinalfbo = 0;
    static int rrw = 0, rrh = 0, rroutw = 0, rrouth = 0;
    static bool rrfailed = false, rrupscaled = false, rrhdr = false;

    static void destroyrrimages()
    {
        if(!rrw) return;
        glFinish();
        queueidle();
        ngx_rrrelease();
        deletefbo(rrcolorfbo);
        deletefbo(rrguidefbo);
        deletefbo(rroutfbo);
        deletefbo(rrfinalfbo);
        if(rrfinaltex) glDeleteTextures(1, &rrfinaltex);
        rrfinaltex = 0;
        sharedimage *imgs[5] = { &rrcolor, &rrdiffuse, &rrspecular, &rrnormals, &rroutput };
        loopi(5) destroysharedimage(*imgs[i]);
        rrpass.cleanupgl();
        rrw = rrh = rroutw = rrouth = 0;
    }

    static bool createrrimages(int w, int h, int outw, int outh)
    {
        rrw = w;
        rrh = h;
        rroutw = outw;
        rrouth = outh;
        rrhdr = hdrscreen;
        if(!createsharedimage(rrcolor, w, h, VK_FORMAT_R16G16B16A16_SFLOAT, GL_RGBA16F) ||
           !createsharedimage(rrdiffuse, w, h, VK_FORMAT_R8G8B8A8_UNORM, GL_RGBA8) ||
           !createsharedimage(rrspecular, w, h, VK_FORMAT_R8G8B8A8_UNORM, GL_RGBA8) ||
           !createsharedimage(rrnormals, w, h, VK_FORMAT_R16G16B16A16_SFLOAT, GL_RGBA16F) ||
           // the tonemap reads it as the lit frame, from a rectangle texture
           !createsharedimage(rroutput, outw, outh, VK_FORMAT_R16G16B16A16_SFLOAT, GL_RGBA16F, GL_TEXTURE_RECTANGLE) ||
           !rrpass.setup())
            return false;
        if(!makefbo(rrcolorfbo, rrcolor) || !makefbo(rrguidefbo, rrdiffuse, &rrspecular, &rrnormals) || !makefbo(rroutfbo, rroutput)) return false;
        if(outw != w || outh != h)
        {
            glGenTextures(1, &rrfinaltex);
            createtexture(rrfinaltex, outw, outh, NULL, 3, 0, screenformat(GL_RGB8), GL_TEXTURE_RECTANGLE);
            glGenFramebuffers_(1, &rrfinalfbo);
            glBindFramebuffer_(GL_FRAMEBUFFER, rrfinalfbo);
            glFramebufferTexture2D_(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_RECTANGLE, rrfinaltex, 0);
            bool ok = glCheckFramebufferStatus_(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
            glBindFramebuffer_(GL_FRAMEBUFFER, renderfbo);
            if(!ok) return false;
        }
        NAMEIMAGE(rrcolor, "rrcolor");
        NAMEIMAGE(rroutput, "rroutput");
        NAMECMDS(rrpass.cmds, "dlss rr");
        sharedimage *imgs[5] = { &rrcolor, &rrdiffuse, &rrspecular, &rrnormals, &rroutput };
        return togl(imgs, 5);
    }

    // the mode that runs: 0 when DLSS can't
    static int dlssmode()
    {
        if(!dlss || !vkactive || !dlssavailable || srfailed || msaasamples) return 0;
        return dlss;
    }
}

using namespace vk;

bool vk_dlssrractive() { return dlssrr && vkactive && dlssrravailable && !rrfailed && !msaasamples && !drawtex; }
bool vk_dlssactive() { return dlssmode() == 1 && !drawtex && !vk_dlssrractive(); } // DLAA, as the anti-aliasing step
bool vk_dlssjittering() { return !drawtex && (dlssmode() || vk_dlssrractive()); }
bool vk_dlssupscaling() { return dlssmode() >= 2 && !drawtex; }

// the share of the screen's size the view is drawn at
float vk_dlssrenderscale()
{
    int mode = dlssmode();
    return mode >= 2 ? dlssscales[mode] : 1.0f;
}

// once a frame: the next sub-pixel offset, enough of them that every screen pixel is covered by the view's pixels
void vk_dlssnextjitter()
{
    srframe++;
    float scale = vk_dlssrenderscale();
    int phases = max(16, int(ceilf(8/(scale*scale))));
    int i = srframe%phases + 1;
    srjitter = vec2(halton(i, 2) - 0.5f, halton(i, 3) - 0.5f);
    settexlodbias(scale < 1 ? log2f(scale) : 0.0f);
    srprepared = rrupscaled = false;
}

vec2 vk_dlssjitter() { return srjitter; }

// DLAA: GL tonemaps into srcolor, at the view's size, which is the screen's
GLuint vk_dlssinput()
{
    if(!ensuresrimages(vieww, viewh, vieww, viewh)) return renderfbo;
    return srcolorfbo;
}

void vk_dlssapply(GLuint outfbo)
{
    if(!srw) return;
    srreset = drawmotion(srw, srh);
    bool ok = evaluatesr(dlssqualities[1]);
    // the anti-aliased frame, or the plain one if DLSS let it down
    glBindFramebuffer_(GL_READ_FRAMEBUFFER, ok ? sroutfbo : srcolorfbo);
    glBindFramebuffer_(GL_DRAW_FRAMEBUFFER, outfbo);
    glBlitFramebuffer_(0, 0, srw, srh, 0, 0, srw, srh, GL_COLOR_BUFFER_BIT, GL_NEAREST);
    glBindFramebuffer_(GL_FRAMEBUFFER, outfbo);
}

// once the view is drawn, while its G-buffer is there: the motion super resolution needs to bring it up later
void vk_dlssprepare()
{
    if(dlssmode() < 2 || rrupscaled || drawtex) return;
    if(!ensuresrimages(gw, gh, renderw, renderh)) return;
    srreset = drawmotion(srw, srh);
    srprepared = true;
}

// in place of stretching the view up to the screen: DLSS does it, or ray reconstruction already has
bool vk_dlssupscale(GLuint srcfbo, int w, int h, GLuint outfbo, int outw, int outh)
{
    GLuint from = 0;
    int fromw = 0, fromh = 0;
    if(rrupscaled && rrfinalfbo)
    {
        from = rrfinalfbo;
        fromw = rroutw;
        fromh = rrouth;
    }
    else if(srprepared && srw == w && srh == h)
    {
        srprepared = false;
        glBindFramebuffer_(GL_READ_FRAMEBUFFER, srcfbo);
        glBindFramebuffer_(GL_DRAW_FRAMEBUFFER, srcolorfbo);
        glBlitFramebuffer_(0, 0, w, h, 0, 0, w, h, GL_COLOR_BUFFER_BIT, GL_NEAREST);
        if(!evaluatesr(dlssqualities[dlssmode()])) return false;
        from = sroutfbo;
        fromw = sroutw;
        fromh = srouth;
    }
    else return false;
    glBindFramebuffer_(GL_READ_FRAMEBUFFER, from);
    glBindFramebuffer_(GL_DRAW_FRAMEBUFFER, outfbo);
    glBlitFramebuffer_(0, 0, fromw, fromh, 0, 0, outw, outh, GL_COLOR_BUFFER_BIT, fromw == outw && fromh == outh ? GL_NEAREST : GL_LINEAR);
    glBindFramebuffer_(GL_FRAMEBUFFER, outfbo);
    return true;
}

// once the frame is lit, transparent things and all: DLSS denoises and anti-aliases it, in place at the view's size
// (1), or brought up to the screen's (2), for vk_dlssrrfinal; 0 when it couldn't
int vk_dlssrrapply()
{
    int w = vieww, h = viewh, mode = dlssmode(), outw = mode >= 2 ? renderw : w, outh = mode >= 2 ? renderh : h;
    if(!ensuresrimages(w, h, w, h)) return 0;
    if(rrw != w || rrh != h || rroutw != outw || rrouth != outh || rrhdr != hdrscreen)
    {
        destroyrrimages();
        if(!createrrimages(w, h, outw, outh))
        {
            conoutf(colourred, "DLSS: could not make the %dx%d to %dx%d ray reconstruction images", w, h, outw, outh);
            destroyrrimages();
            rrfailed = true;
            return 0;
        }
    }
    Shader *guideshader = useshaderbyname("dlssrrguides");
    if(!guideshader || !ngx_rrready(w, h, outw, outh, dlssqualities[mode])) { rrfailed = true; return 0; }

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

    bool reset = drawmotion(w, h);
    sharedimage *inputs[6] = { &rrcolor, &rrdiffuse, &rrspecular, &rrnormals, &srmvec, &srdepth };
    bool ok = rrpass.run(inputs, 6, &rroutput, [&](VkCommandBuffer cmd)
    {
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
        return ngx_rrevaluate(cmd, in);
    });
    if(!ok)
    {
        rrfailed = true;
        glBindFramebuffer_(GL_FRAMEBUFFER, hdrfbo);
        return 0;
    }
    if(outw != w || outh != h)
    {
        rrupscaled = true;
        return 2;
    }
    // the denoised frame takes the lit one's place
    glBindFramebuffer_(GL_READ_FRAMEBUFFER, rroutfbo);
    glBindFramebuffer_(GL_DRAW_FRAMEBUFFER, hdrfbo);
    glBlitFramebuffer_(0, 0, w, h, 0, 0, w, h, GL_COLOR_BUFFER_BIT, GL_NEAREST);
    glBindFramebuffer_(GL_FRAMEBUFFER, hdrfbo);
    return 1;
}

// where the frame ray reconstruction brought up is tonemapped to, and what it is tonemapped from
GLuint vk_dlssrrfinal(GLuint &tex, int &w, int &h)
{
    tex = rroutput.gltex;
    w = rroutw;
    h = rrouth;
    return rrfinalfbo;
}

void vk_dlsscleanupgl()
{
    if(!device) return;
    destroyrrimages();
    destroysrimages();
    settexlodbias(0);
}

void vk_dlssshutdown()
{
    if(!device) return;
    srpass.shutdown();
    rrpass.shutdown();
    srfailed = rrfailed = false;
}

static const char * const dlssmodenames[6] = { "off", "DLAA", "Quality", "Balanced", "Performance", "Ultra Performance" };
ICOMMAND(0, dlssinfo, "", (),
{
    int mode = dlssmode();
    conoutf(colourwhite, "DLSS: %s, drawing at %dx%d for %dx%d, super resolution GPU %.2f ms%s", dlssmodenames[mode], gw, gh, renderw, renderh, srpass.ms, srfailed ? ", failed" : "");
    conoutf(colourwhite, "DLSS: ray reconstruction %s, GPU %.2f ms%s", vk_dlssrractive() ? "on" : "off", rrpass.ms, rrfailed ? ", failed" : "");
});
