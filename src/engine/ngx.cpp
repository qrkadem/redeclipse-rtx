// DLSS through NVIDIA NGX on the Vulkan device from vk.cpp. NGX is optional: without the SDK libraries
// (src/ngx/fetch.sh) the build leaves it out and DLSS reports unavailable.
// This software contains source code provided by NVIDIA Corporation (the NGX SDK headers and helpers).
#include "engine.h"
#include "vkint.h"

VARR(dlssavailable, 0);   // DLSS super resolution
VARR(dlssgavailable, 0);  // DLSS frame generation: how many frames it can make per rendered frame, 0 without it
VARR(dlssrravailable, 0); // DLSS ray reconstruction
VAR(IDF_INIT, dbgngx, 0, 0, 2); // NGX logs to stderr: 1 normal, 2 verbose
FVAR(0, dlssgmvecscale, -1e3f, 1, 1e3f); // what NGX multiplies the motion vectors by, they are in render pixels

#ifdef HAVE_NGX
#include <wchar.h>
#include "nvsdk_ngx_vk.h"
#include "nvsdk_ngx_helpers_vk.h"
#include "nvsdk_ngx_defs_dlssg.h"
#include "nvsdk_ngx_params_dlssg.h"
#include "nvsdk_ngx_defs_dlssd.h"
#include "nvsdk_ngx_helpers_dlssg_vk.h"
#include "nvsdk_ngx_helpers_dlssd_vk.h"

namespace vk
{
    // NGX wants an id for the app: NVIDIA hands those out, this is a fixed custom one
    static const char *ngxprojectid = "a0f57b54-1daf-4934-90ae-c4035c19df04";
    static bool ngxinited = false;
    static NVSDK_NGX_Parameter *ngxcaps = NULL;
    static wchar_t ngxdatapath[MAXSTRLEN], ngxlibpath[MAXSTRLEN];
    static const wchar_t *ngxlibpaths[1] = { ngxlibpath };
    static NVSDK_NGX_FeatureCommonInfo ngxinfo;
    static const NVSDK_NGX_Feature ngxfeatures[] = { NVSDK_NGX_Feature_SuperSampling, NVSDK_NGX_Feature_FrameGeneration, NVSDK_NGX_Feature_RayReconstruction };

    static void NVSDK_CONV ngxlog(const char *message, NVSDK_NGX_Logging_Level, NVSDK_NGX_Feature)
    {
        fprintf(stderr, "NGX: %s", message); // NGX threads call this too, so not the console
    }

    static void ngxsetup()
    {
        static bool done = false;
        if(done) return;
        done = true;
        // the feature libraries ship next to the binary, NGX keeps its logs and caches in the home dir
        char *base = SDL_GetBasePath();
        mbstowcs(ngxlibpath, base ? base : "./", MAXSTRLEN-1);
        if(base) SDL_free(base);
        string data;
        copystring(data, findfile("cache/ngx/", "w"));
        mbstowcs(ngxdatapath, data, MAXSTRLEN-1);
        memset(&ngxinfo, 0, sizeof(ngxinfo));
        ngxinfo.PathListInfo.Path = ngxlibpaths;
        ngxinfo.PathListInfo.Length = 1;
        if(dbgngx)
        {
            ngxinfo.LoggingInfo.LoggingCallback = ngxlog;
            ngxinfo.LoggingInfo.MinimumLoggingLevel = dbgngx > 1 ? NVSDK_NGX_LOGGING_LEVEL_VERBOSE : NVSDK_NGX_LOGGING_LEVEL_ON;
        }
    }

    static NVSDK_NGX_FeatureDiscoveryInfo ngxdiscovery(NVSDK_NGX_Feature feature)
    {
        NVSDK_NGX_FeatureDiscoveryInfo di;
        memset(&di, 0, sizeof(di));
        di.SDKVersion = NVSDK_NGX_Version_API;
        di.FeatureID = feature;
        di.Identifier.IdentifierType = NVSDK_NGX_Application_Identifier_Type_Project_Id;
        di.Identifier.v.ProjectDesc.ProjectId = ngxprojectid;
        di.Identifier.v.ProjectDesc.EngineType = NVSDK_NGX_ENGINE_TYPE_CUSTOM;
        di.Identifier.v.ProjectDesc.EngineVersion = versionstring;
        di.ApplicationDataPath = ngxdatapath;
        di.FeatureInfo = &ngxinfo;
        return di;
    }

    static void ngxaddext(vector<const char *> &exts, const char *name)
    {
        loopv(exts) if(!strcmp(exts[i], name)) return;
        exts.add(newstring(name)); // kept for as long as the instance or device lives
    }

    void ngx_instanceextensions(vector<const char *> &exts)
    {
        ngxsetup();
        loopi(int(sizeof(ngxfeatures)/sizeof(ngxfeatures[0])))
        {
            NVSDK_NGX_FeatureDiscoveryInfo di = ngxdiscovery(ngxfeatures[i]);
            uint32_t n = 0;
            VkExtensionProperties *props = NULL;
            if(NVSDK_NGX_SUCCEED(NVSDK_NGX_VULKAN_GetFeatureInstanceExtensionRequirements(&di, &n, &props)) && props)
                loopj(int(n)) ngxaddext(exts, props[j].extensionName);
        }
    }

    void ngx_deviceextensions(vector<const char *> &exts)
    {
        ngxsetup();
        loopi(int(sizeof(ngxfeatures)/sizeof(ngxfeatures[0])))
        {
            NVSDK_NGX_FeatureDiscoveryInfo di = ngxdiscovery(ngxfeatures[i]);
            uint32_t n = 0;
            VkExtensionProperties *props = NULL;
            if(NVSDK_NGX_SUCCEED(NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements(instance, physdev, &di, &n, &props)) && props)
                loopj(int(n)) ngxaddext(exts, props[j].extensionName);
        }
    }

    void ngx_init()
    {
        dlssavailable = dlssgavailable = dlssrravailable = 0;
        if(devprops.vendorID != 0x10DE) return; // NVIDIA only
        ngxsetup();
        NVSDK_NGX_Result r = NVSDK_NGX_VULKAN_Init_with_ProjectID(ngxprojectid, NVSDK_NGX_ENGINE_TYPE_CUSTOM, versionstring, ngxdatapath,
            instance, physdev, device, vkGetInstanceProcAddr, vkGetDeviceProcAddr, &ngxinfo);
        if(NVSDK_NGX_FAILED(r))
        {
            conoutf(colouryellow, "DLSS: NGX did not start (0x%X)", unsigned(r));
            return;
        }
        ngxinited = true;
        if(NVSDK_NGX_FAILED(NVSDK_NGX_VULKAN_GetCapabilityParameters(&ngxcaps)) || !ngxcaps)
        {
            conoutf(colouryellow, "DLSS: NGX has no capability parameters");
            return;
        }
        int ss = 0, fg = 0, rr = 0;
        unsigned int maxgen = 0;
        ngxcaps->Get(NVSDK_NGX_Parameter_SuperSampling_Available, &ss);
        ngxcaps->Get(NVSDK_NGX_Parameter_FrameGeneration_Available, &fg);
        ngxcaps->Get(NVSDK_NGX_Parameter_SuperSamplingDenoising_Available, &rr);
        if(fg && NVSDK_NGX_FAILED(ngxcaps->Get(NVSDK_NGX_DLSSG_Parameter_MultiFrameCountMax, &maxgen))) maxgen = 1;
        dlssavailable = ss ? 1 : 0;
        dlssgavailable = fg ? max(int(maxgen), 1) : 0;
        dlssrravailable = rr ? 1 : 0;
        conoutf(colourwhite, "DLSS: super resolution %s, frame generation %s, ray reconstruction %s",
            ss ? "yes" : "no", fg ? tempformatstring("up to %dx", dlssgavailable+1) : "no", rr ? "yes" : "no");
    }

    // DLSS frame generation
    static NVSDK_NGX_Handle *fghandle = NULL;
    static NVSDK_NGX_Parameter *fgparams = NULL;
    static int fgw = 0, fgh = 0, fgrw = 0, fgrh = 0;
    static VkFormat fgformat = VK_FORMAT_UNDEFINED;
    static bool fgfailed = false;

    int ngx_fgmaxgen() { return ngxinited ? dlssgavailable : 0; }

    void ngx_fgrelease()
    {
        if(fghandle) NVSDK_NGX_VULKAN_ReleaseFeature(fghandle);
        fghandle = NULL;
        if(fgparams) NVSDK_NGX_VULKAN_DestroyParameters(fgparams);
        fgparams = NULL;
        fgw = fgh = fgrw = fgrh = 0;
        fgformat = VK_FORMAT_UNDEFINED;
    }

    bool ngx_fgready(int w, int h, int rw, int rh, VkFormat format)
    {
        if(!ngxinited || !dlssgavailable) return false;
        if(fgw == w && fgh == h && fgrw == rw && fgrh == rh && fgformat == format) return fghandle != NULL && !fgfailed;
        // the caller made sure nothing in flight uses the old feature
        ngx_fgrelease();
        fgw = w;
        fgh = h;
        fgrw = rw;
        fgrh = rh;
        fgformat = format;
        fgfailed = true;
        if(NVSDK_NGX_FAILED(NVSDK_NGX_VULKAN_AllocateParameters(&fgparams)) || !fgparams)
        {
            conoutf(colourred, "DLSS: no parameters for frame generation");
            return false;
        }
        VkCommandBuffer cmd = begincommands();
        if(!cmd) return false;
        NVSDK_NGX_DLSSG_Create_Params cp;
        memset(&cp, 0, sizeof(cp));
        cp.Width = w;
        cp.Height = h;
        cp.NativeBackbufferFormat = format;
        cp.RenderWidth = rw;
        cp.RenderHeight = rh;
        cp.DynamicResolutionScaling = false;
        NVSDK_NGX_Result r = NGX_VK_CREATE_DLSSG(cmd, 1, 1, &fghandle, fgparams, &cp);
        bool ok = endcommands(cmd);
        if(NVSDK_NGX_FAILED(r) || !ok || !fghandle)
        {
            conoutf(colourred, "DLSS: frame generation could not start for %dx%d (0x%X)", w, h, unsigned(r));
            fghandle = NULL;
            return false;
        }
        fgfailed = false;
        conoutf(colourwhite, "DLSS: frame generation ready for %dx%d%s, motion and depth %dx%d", w, h, format == VK_FORMAT_R8G8B8A8_UNORM ? "" : " HDR", rw, rh);
        return true;
    }

    static NVSDK_NGX_Resource_VK ngxresource(VkImage image, VkImageView view, VkFormat format, int w, int h, bool readwrite)
    {
        VkImageSubresourceRange range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        return NVSDK_NGX_Create_ImageView_Resource_VK(view, image, range, format, w, h, readwrite);
    }

    bool ngx_fgevaluate(VkCommandBuffer cmd, const fginputs &in, int numgen)
    {
        if(!fghandle || fgfailed) return false;
        NVSDK_NGX_Resource_VK backbuffer = ngxresource(in.backbuffer.image, in.backbuffer.view, in.backbuffer.format, in.backbuffer.w, in.backbuffer.h, false),
                              depth = ngxresource(in.depth->image, in.depth->view, in.depth->format, in.depth->w, in.depth->h, false),
                              mvec = ngxresource(in.mvec->image, in.mvec->view, in.mvec->format, in.mvec->w, in.mvec->h, false),
                              hudless;
        if(in.hashudless) hudless = ngxresource(in.hudless.image, in.hudless.view, in.hudless.format, in.hudless.w, in.hudless.h, false);
        static const float identity[16] = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 };
        for(int g = 0; g < numgen; g++)
        {
            deviceimage &out = in.interp[g];
            NVSDK_NGX_Resource_VK interp = ngxresource(out.image, out.view, out.format, out.w, out.h, true);
            NVSDK_NGX_VK_DLSSG_Eval_Params ep;
            memset(&ep, 0, sizeof(ep));
            ep.pBackbuffer = &backbuffer;
            ep.pDepth = &depth;
            ep.pMVecs = &mvec;
            ep.pHudless = in.hashudless ? &hudless : NULL;
            ep.pOutputInterpFrame = &interp;

            NVSDK_NGX_DLSSG_Opt_Eval_Params op;
            op.multiFrameCount = numgen;
            op.multiFrameIndex = g+1;
            memcpy(op.cameraViewToClip, in.camera.viewtoclip, sizeof(op.cameraViewToClip));
            memcpy(op.clipToCameraView, in.camera.cliptoview, sizeof(op.clipToCameraView));
            memcpy(op.clipToLensClip, identity, sizeof(op.clipToLensClip));
            memcpy(op.clipToPrevClip, in.camera.cliptoprevclip, sizeof(op.clipToPrevClip));
            memcpy(op.prevClipToClip, in.camera.prevcliptoclip, sizeof(op.prevClipToClip));
            op.jitterOffset[0] = op.jitterOffset[1] = 0;
            op.mvecScale[0] = op.mvecScale[1] = dlssgmvecscale;
            op.cameraPinholeOffset[0] = op.cameraPinholeOffset[1] = 0;
            memcpy(op.cameraPos, in.camera.pos, sizeof(op.cameraPos));
            memcpy(op.cameraUp, in.camera.up, sizeof(op.cameraUp));
            memcpy(op.cameraRight, in.camera.right, sizeof(op.cameraRight));
            memcpy(op.cameraFwd, in.camera.fwd, sizeof(op.cameraFwd));
            op.cameraNear = in.camera.nearplane;
            op.cameraFar = in.camera.farplane;
            op.cameraFOV = in.camera.fov;
            op.cameraAspectRatio = in.camera.aspect;
            op.colorBuffersHDR = in.hdr;
            op.depthInverted = false;
            op.cameraMotionIncluded = true;
            op.reset = in.camera.reset;
            op.automodeOverrideReset = false;
            op.notRenderingGameFrames = false;
            op.orthoProjection = false;
            op.motionVectorsInvalidValue = 0;
            op.motionVectorsDilated = false;
            op.menuDetectionEnabled = false;
            op.mvecsSubrectSize.Width = in.mvec->w;
            op.mvecsSubrectSize.Height = in.mvec->h;
            op.depthSubrectSize.Width = in.depth->w;
            op.depthSubrectSize.Height = in.depth->h;
            if(in.hashudless)
            {
                op.hudLessSubrectSize.Width = in.hudless.w;
                op.hudLessSubrectSize.Height = in.hudless.h;
            }
            op.backbufferSubrectSize.Width = in.backbuffer.w;
            op.backbufferSubrectSize.Height = in.backbuffer.h;
            op.outputInterpSubrectSize.Width = out.w;
            op.outputInterpSubrectSize.Height = out.h;
            NVSDK_NGX_Result r = NGX_VK_EVALUATE_DLSSG(cmd, fghandle, fgparams, &ep, &op);
            if(NVSDK_NGX_FAILED(r))
            {
                conoutf(colourred, "DLSS: frame generation failed (0x%X), turning it off", unsigned(r));
                fgfailed = true;
                return false;
            }
        }
        return true;
    }

    // DLSS super resolution
    static NVSDK_NGX_Handle *srhandle = NULL;
    static NVSDK_NGX_Parameter *srparams = NULL;
    static int srw = 0, srh = 0, sroutw = 0, srouth = 0, srquality = -1;
    static bool srfailed = false;

    void ngx_srrelease()
    {
        if(srhandle) NVSDK_NGX_VULKAN_ReleaseFeature(srhandle);
        srhandle = NULL;
        if(srparams) NVSDK_NGX_VULKAN_DestroyParameters(srparams);
        srparams = NULL;
        srw = srh = sroutw = srouth = 0;
        srquality = -1;
    }

    bool ngx_srready(int w, int h, int outw, int outh, int quality)
    {
        if(!ngxinited || !dlssavailable) return false;
        if(srw == w && srh == h && sroutw == outw && srouth == outh && srquality == quality) return srhandle != NULL && !srfailed;
        ngx_srrelease();
        srw = w;
        srh = h;
        sroutw = outw;
        srouth = outh;
        srquality = quality;
        srfailed = true;
        if(NVSDK_NGX_FAILED(NVSDK_NGX_VULKAN_AllocateParameters(&srparams)) || !srparams) return false;
        VkCommandBuffer cmd = begincommands();
        if(!cmd) return false;
        NVSDK_NGX_DLSS_Create_Params cp;
        memset(&cp, 0, sizeof(cp));
        cp.Feature.InWidth = w;
        cp.Feature.InHeight = h;
        cp.Feature.InTargetWidth = outw;
        cp.Feature.InTargetHeight = outh;
        cp.Feature.InPerfQualityValue = NVSDK_NGX_PerfQuality_Value(quality);
        cp.InFeatureCreateFlags = NVSDK_NGX_DLSS_Feature_Flags_MVLowRes; // motion vectors at the render size, without jitter, colour already tonemapped
        NVSDK_NGX_Result r = NGX_VULKAN_CREATE_DLSS_EXT(cmd, 1, 1, &srhandle, srparams, &cp);
        bool ok = endcommands(cmd);
        if(NVSDK_NGX_FAILED(r) || !ok || !srhandle)
        {
            conoutf(colourred, "DLSS: super resolution could not start for %dx%d to %dx%d (0x%X)", w, h, outw, outh, unsigned(r));
            srhandle = NULL;
            return false;
        }
        srfailed = false;
        conoutf(colourwhite, "DLSS: super resolution ready, %dx%d to %dx%d", w, h, outw, outh);
        return true;
    }

    bool ngx_srevaluate(VkCommandBuffer cmd, const srinputs &in)
    {
        if(!srhandle || srfailed) return false;
        NVSDK_NGX_Resource_VK color = ngxresource(in.color->image, in.color->view, in.color->format, in.color->w, in.color->h, false),
                              output = ngxresource(in.output->image, in.output->view, in.output->format, in.output->w, in.output->h, true),
                              depth = ngxresource(in.depth->image, in.depth->view, in.depth->format, in.depth->w, in.depth->h, false),
                              mvec = ngxresource(in.mvec->image, in.mvec->view, in.mvec->format, in.mvec->w, in.mvec->h, false);
        NVSDK_NGX_VK_DLSS_Eval_Params ep;
        memset(&ep, 0, sizeof(ep));
        ep.Feature.pInColor = &color;
        ep.Feature.pInOutput = &output;
        ep.pInDepth = &depth;
        ep.pInMotionVectors = &mvec;
        ep.InJitterOffsetX = in.jitterx;
        ep.InJitterOffsetY = in.jittery;
        ep.InRenderSubrectDimensions.Width = in.color->w;
        ep.InRenderSubrectDimensions.Height = in.color->h;
        ep.InReset = in.reset ? 1 : 0;
        ep.InMVScaleX = ep.InMVScaleY = 1;
        ep.InPreExposure = ep.InExposureScale = 1;
        NVSDK_NGX_Result r = NGX_VULKAN_EVALUATE_DLSS_EXT(cmd, srhandle, srparams, &ep);
        if(NVSDK_NGX_FAILED(r))
        {
            conoutf(colourred, "DLSS: super resolution failed (0x%X), turning it off", unsigned(r));
            srfailed = true;
            return false;
        }
        return true;
    }

    // DLSS ray reconstruction
    static NVSDK_NGX_Handle *rrhandle = NULL;
    static NVSDK_NGX_Parameter *rrparams = NULL;
    static int rrw = 0, rrh = 0;
    static bool rrfailed = false;

    void ngx_rrrelease()
    {
        if(rrhandle) NVSDK_NGX_VULKAN_ReleaseFeature(rrhandle);
        rrhandle = NULL;
        if(rrparams) NVSDK_NGX_VULKAN_DestroyParameters(rrparams);
        rrparams = NULL;
        rrw = rrh = 0;
    }

    bool ngx_rrready(int w, int h)
    {
        if(!ngxinited || !dlssrravailable) return false;
        if(rrw == w && rrh == h) return rrhandle != NULL && !rrfailed;
        ngx_rrrelease();
        rrw = w;
        rrh = h;
        rrfailed = true;
        if(NVSDK_NGX_FAILED(NVSDK_NGX_VULKAN_AllocateParameters(&rrparams)) || !rrparams) return false;
        VkCommandBuffer cmd = begincommands();
        if(!cmd) return false;
        NVSDK_NGX_DLSSD_Create_Params cp;
        memset(&cp, 0, sizeof(cp));
        cp.InDenoiseMode = NVSDK_NGX_DLSS_Denoise_Mode_DLUnified;
        cp.InRoughnessMode = NVSDK_NGX_DLSS_Roughness_Mode_Packed; // in the normals' w
        cp.InUseHWDepth = NVSDK_NGX_DLSS_Depth_Type_HW;
        cp.InWidth = cp.InTargetWidth = w;
        cp.InHeight = cp.InTargetHeight = h;
        cp.InPerfQualityValue = NVSDK_NGX_PerfQuality_Value_DLAA;
        // motion vectors at the render size without jitter, the lit frame before tonemapping, exposed by DLSS itself
        cp.InFeatureCreateFlags = NVSDK_NGX_DLSS_Feature_Flags_MVLowRes|NVSDK_NGX_DLSS_Feature_Flags_IsHDR|NVSDK_NGX_DLSS_Feature_Flags_AutoExposure;
        NVSDK_NGX_Result r = NGX_VULKAN_CREATE_DLSSD_EXT1(device, cmd, 1, 1, &rrhandle, rrparams, &cp);
        bool ok = endcommands(cmd);
        if(NVSDK_NGX_FAILED(r) || !ok || !rrhandle)
        {
            conoutf(colourred, "DLSS: ray reconstruction could not start for %dx%d (0x%X)", w, h, unsigned(r));
            rrhandle = NULL;
            return false;
        }
        rrfailed = false;
        conoutf(colourwhite, "DLSS: ray reconstruction ready, %dx%d", w, h);
        return true;
    }

    bool ngx_rrevaluate(VkCommandBuffer cmd, const rrinputs &in)
    {
        if(!rrhandle || rrfailed) return false;
        #define RRRES(name, img, rw) NVSDK_NGX_Resource_VK name = ngxresource(img->image, img->view, img->format, img->w, img->h, rw)
        RRRES(color, in.color, false);
        RRRES(diffuse, in.diffuse, false);
        RRRES(specular, in.specular, false);
        RRRES(normals, in.normals, false);
        RRRES(mvec, in.mvec, false);
        RRRES(depth, in.depth, false);
        RRRES(output, in.output, true);
        #undef RRRES
        NVSDK_NGX_VK_DLSSD_Eval_Params ep;
        memset(&ep, 0, sizeof(ep));
        ep.pInColor = &color;
        ep.pInOutput = &output;
        ep.pInDiffuseAlbedo = &diffuse;
        ep.pInSpecularAlbedo = &specular;
        ep.pInNormals = &normals;
        ep.pInDepth = &depth;
        ep.pInMotionVectors = &mvec;
        ep.InJitterOffsetX = in.jitterx;
        ep.InJitterOffsetY = in.jittery;
        ep.InRenderSubrectDimensions.Width = in.color->w;
        ep.InRenderSubrectDimensions.Height = in.color->h;
        ep.InReset = in.reset ? 1 : 0;
        ep.InMVScaleX = ep.InMVScaleY = 1;
        ep.InPreExposure = ep.InExposureScale = 1;
        ep.pInWorldToViewMatrix = (float *)in.worldtoview;
        ep.pInViewToClipMatrix = (float *)in.viewtoclip;
        NVSDK_NGX_Result r = NGX_VULKAN_EVALUATE_DLSSD_EXT(cmd, rrhandle, rrparams, &ep);
        if(NVSDK_NGX_FAILED(r))
        {
            conoutf(colourred, "DLSS: ray reconstruction failed (0x%X), turning it off", unsigned(r));
            rrfailed = true;
            return false;
        }
        return true;
    }

    void ngx_shutdown()
    {
        ngx_rrrelease();
        ngx_srrelease();
        ngx_fgrelease();
        if(ngxcaps) NVSDK_NGX_VULKAN_DestroyParameters(ngxcaps);
        ngxcaps = NULL;
        if(ngxinited) NVSDK_NGX_VULKAN_Shutdown1(device);
        ngxinited = false;
        dlssavailable = dlssgavailable = dlssrravailable = 0;
    }
}
#else
namespace vk
{
    void ngx_instanceextensions(vector<const char *> &exts) {}
    void ngx_deviceextensions(vector<const char *> &exts) {}
    void ngx_init() {}
    void ngx_shutdown() {}
    int ngx_fgmaxgen() { return 0; }
    bool ngx_srready(int w, int h, int outw, int outh, int quality) { return false; }
    bool ngx_srevaluate(VkCommandBuffer cmd, const srinputs &in) { return false; }
    void ngx_srrelease() {}
    bool ngx_rrready(int w, int h) { return false; }
    bool ngx_rrevaluate(VkCommandBuffer cmd, const rrinputs &in) { return false; }
    void ngx_rrrelease() {}
    bool ngx_fgready(int w, int h, int renderw, int renderh, VkFormat format) { return false; }
    bool ngx_fgevaluate(VkCommandBuffer cmd, const fginputs &in, int numgen) { return false; }
    void ngx_fgrelease() {}
}
#endif
