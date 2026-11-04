// Vulkan presenting: Vulkan owns the window, the device and the swapchain, and the GL renderer draws each frame
// into an image Vulkan allocated and shares with GL. DLSS and ray tracing run on the same Vulkan device.
#ifndef ENGINE_VK_H
#define ENGINE_VK_H

extern int vulkan;
extern bool vkactive;

// window setup, in this order: vk_createwindow (the visible Vulkan window), vk_createglwindow (the hidden window
// the GL context lives on), then vk_init once that context is current; any false return means GL presents instead
extern SDL_Window *vk_createwindow(const char *caption, int x, int y, int w, int h, int flags);
extern SDL_Window *vk_createglwindow();
extern bool vk_init();
extern void vk_cleanupgl(); // while the GL context is still current
extern void vk_shutdown();  // before the window goes
extern void vk_destroyglwindow();

extern void vk_getdrawablesize(int &w, int &h);
extern void vk_resize(int w, int h); // screen images follow the drawable size
extern void vk_beginframe();         // GL draws into a screen image Vulkan is done with
extern void vk_present();            // hands the screen image to Vulkan, which presents it
extern void vk_snapshotview();      // when the G-buffer of the main view is set up
extern int gvelocity;
extern void vk_partmotion(const void *owner, const void *part, const matrix4 &world, matrix4 &cur, matrix4 &prev);
extern void vk_capturefg();         // after the scene: motion vectors and depth for frame generation
extern void vk_capturehudless();    // once the scene is on screen and before the HUD goes over it
extern void vk_setvsync(int vsync, int tear);
extern void vk_updatehdr();         // the start of a frame: HDR output turns on or off
extern bool hdrscreen;              // HDR output is on, so the tonemapped frame travels to the screen in 16 bits
static inline GLenum screenformat(GLenum sdr) { return !hdrscreen ? sdr : (sdr == GL_RGB || sdr == GL_RGB8 ? GL_RGB16 : GL_RGBA16); }
extern void vk_latencysleep();      // the start of a frame, before input
extern void vk_latencymark(int marker); // VkLatencyMarkerNV: 1 simulation end, 2 render submission start, 6 input sample

extern bool vk_drawmotion(GLuint fbo, int w, int h, const matrix4 &screenm, const matrix4 &prevscreen);

// DLSS anti-aliasing, see vkdlss.cpp
extern bool vk_dlssactive();
extern void vk_dlssnextjitter();
extern vec2 vk_dlssjitter();       // render pixels
extern GLuint vk_dlssinput();      // where the tonemapped frame goes
extern void vk_dlssapply(GLuint outfbo);
extern void vk_dlsscleanupgl();
extern void vk_dlssshutdown();

// ray tracing, see vkrt.cpp
// an animated model's vertex for the ray traced scene, skinned there the way the GL shaders skin it
struct rtskinvert
{
    float pos[4], weights[4];
    uint bones; // four bone indices into the dual quaternions, a byte each
    uint pad[3];
};
extern matrix4 vkrt_partworld;   // the model part being drawn, placed in the world, set before its meshes draw
extern bool vkrt_capturing();    // animated models drawn now go into the ray traced scene
extern bool vkrt_hasskel(const void *group);
extern void vkrt_registerskel(const void *group, const rtskinvert *verts, int numverts, const uint *indices, int numindices);
extern void vkrt_captureskel(const void *group, const void *owner, const void *part, const dualquat *bones, int numbones);
extern void vkrt_beginframe();
extern void vkrt_pausecapture(bool pause);
extern bool vkrt_playershadows(); // players' sun shadows are traced, so the shadow map leaves them out
extern bool vkrt_aoavailable();  // ray traced ambient occlusion is wanted and possible: picks the lighting shaders
extern bool vkrt_sunavailable(); // the same for the sun's shadow
extern bool vkrt_active();       // traced this frame, so the result can be read
extern bool vkrt_sunready();     // the world's sun shadow comes from ray tracing, the shadow map keeps the rest
extern GLuint vkrt_texture();    // the result for GL's lighting: r ambient occlusion, g sun shadow
extern void vkrt_trace();        // between the G-buffer and lighting
extern void vkrt_worldchanged();
extern void vkrt_newmap();       // a map was loaded: its textures are read again for reflections
// ray traced reflections, after lighting: a light that can reach what is reflected, laid out as the shader reads it
struct rtrefllight
{
    vec o;
    float radius;
    vec color;     // as the deferred shading scales it
    float flags;   // 1 when it casts no shadow
    vec spotdir;   // from the spot's target towards the light
    float spotscale; // 1/(1 - cos of the spot's angle), 0 for a point light
};
// how the deferred shading lights the scene, for what is reflected off screen
struct rtreflshading
{
    vec ambient, suncolor, sundir, skycolor, fogcolor;
    float glowscale, fogdensity, fogscale;
    bool sun;
};
extern float rtreflrough;          // roughness at which traced reflections give way to the environment maps
extern bool reflectionsenabled();  // renderlights.cpp: the reflection pass runs this frame
extern bool vkrt_reflavailable();
// the reflected light, premultiplied by how much of it the rays found (alpha), or 0 when there is none this frame
extern GLuint vkrt_reflect(GLuint litfbo, int w, int h, const rtreflshading &s, const rtrefllight *lights, int numlights);
extern void vkrt_cleanupgl();
extern void vkrt_shutdown();
extern bool vk_hastear();

#endif
