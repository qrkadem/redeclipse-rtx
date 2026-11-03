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
extern void vk_capturefg();         // after the scene: motion vectors and depth for frame generation
extern void vk_capturehudless();    // once the scene is on screen and before the HUD goes over it
extern void vk_setvsync(int vsync, int tear);

// ray tracing, see vkrt.cpp
extern bool vkrt_aoavailable();  // ray traced ambient occlusion is wanted and possible: picks the lighting shaders
extern bool vkrt_sunavailable(); // the same for the sun's shadow
extern bool vkrt_active();       // traced this frame, so the result can be read
extern GLuint vkrt_texture();    // the result for GL's lighting: r ambient occlusion, g sun shadow
extern void vkrt_trace();        // between the G-buffer and lighting
extern void vkrt_worldchanged();
extern void vkrt_cleanupgl();
extern void vkrt_shutdown();
extern bool vk_hastear();

#endif
