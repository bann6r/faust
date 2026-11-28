"""Interactive Faust PBR material laboratory.

Run: python demo.py
Drag with left mouse to orbit; scroll to zoom; keys 1-4 change lighting presets.
"""
import math
import glfw
import moderngl
import numpy as np

from faust import Camera, DirectionalLight, Environment, Material, Mesh, Renderer, Scene
from faust import cube_data, sphere_data


def transform(position=(0,0,0), scale=(1,1,1)):
    m=np.eye(4,dtype="f4"); m[:3,:3]=np.diag(scale); m[:3,3]=position
    return m


def make_environment(ctx):
    """Small analytic studio cubemaps; replace with prefiltered HDR assets in production."""
    colors=[(0.34,0.39,0.48),(0.12,0.15,0.21),(0.62,0.68,0.78),
            (0.055,0.065,0.09),(0.28,0.22,0.16),(0.16,0.20,0.27)]
    irradiance=ctx.texture_cube((16,16),3,dtype="f2")
    prefiltered=ctx.texture_cube((64,64),3,dtype="f2")
    for face,color in enumerate(colors):
        a=np.empty((16,16,3),dtype="f2"); a[:]=np.asarray(color,dtype="f2")*0.75
        irradiance.write(face,a.tobytes())
        b=np.empty((64,64,3),dtype="f2")
        # A broad vertical studio gradient gives metals a readable environment.
        for y in range(64):
            t=y/63
            b[y,:,:]=np.asarray(color,dtype="f4")*(0.45+0.9*(1-t))
        prefiltered.write(face,b.tobytes())
    prefiltered.build_mipmaps()
    irradiance.filter=(moderngl.LINEAR,moderngl.LINEAR)
    prefiltered.filter=(moderngl.LINEAR_MIPMAP_LINEAR,moderngl.LINEAR)
    lut_data=np.zeros((64,64,2),dtype="f2")
    for y in range(64):
        rough=y/63
        for x in range(64):
            ndv=x/63
            lut_data[y,x]=(0.5+0.45*ndv*(1-rough),0.04*(1-ndv)*rough)
    lut=ctx.texture((64,64),2,lut_data.tobytes(),dtype="f2")
    lut.filter=(moderngl.LINEAR,moderngl.LINEAR)
    return Environment(irradiance,prefiltered,lut,6.0),(irradiance,prefiltered,lut)


def build_scene(ctx):
    scene=Scene(DirectionalLight(direction=(-0.45,-1.0,-0.35),radiance=(5.0,4.7,4.3)))
    sphere=Mesh(ctx,sphere_data(48,32)); cube=Mesh(ctx,cube_data())
    roughnesses=(0.08,0.22,0.38,0.56,0.76,0.94)
    # Dielectrics: identical base color, controlled roughness sweep.
    for i,r in enumerate(roughnesses):
        scene.add(sphere,Material(f"Dielectric {r:.2f}",(0.72,0.16,0.075),0.0,r),
                  transform((-4.2+i*1.68,1.0,1.3),(0.72,0.72,0.72)))
    metals=[
        ("Gold",(1.0,0.62,0.18),1.0,0.18),
        ("Copper",(0.95,0.38,0.20),1.0,0.24),
        ("Chrome",(0.78,0.82,0.88),1.0,0.075),
        ("Brushed",(0.72,0.76,0.82),1.0,0.48),
        ("Matte alloy",(0.58,0.62,0.68),1.0,0.82),
        ("Black metal",(0.12,0.14,0.17),1.0,0.32),
    ]
    for i,(name,color,metal,rough) in enumerate(metals):
        scene.add(sphere,Material(name,color,metal,rough),
                  transform((-4.2+i*1.68,1.0,-1.25),(0.72,0.72,0.72)))
    floor=Material("Dark ceramic",(0.19,0.21,0.24),0.0,0.78)
    scene.add(cube,floor,transform((0,-0.22,0),(6.2,0.22,3.7)))
    # A small emissive practical adds a controlled highlight and bloom source.
    scene.add(sphere,Material("Warm emitter",(1.0,0.55,0.22),0.0,0.25,
                              emissive=(7.0,2.2,0.45)),
              transform((4.5,3.2,-3.0),(0.22,0.22,0.22)),cast_shadows=False)
    scene.add_point_light(__import__("faust").PointLight((0,4,3),(32,25,18),12))
    return scene,(sphere,cube)


def main():
    if not glfw.init(): raise RuntimeError("GLFW initialization failed")
    window=None; ctx=None; renderer=None; meshes=[]; env_resources=[]
    try:
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR,3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR,3)
        glfw.window_hint(glfw.OPENGL_PROFILE,glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT,glfw.TRUE)
        glfw.window_hint(glfw.SRGB_CAPABLE,glfw.TRUE)
        window=glfw.create_window(1440,900,"Faust | PBR material laboratory",None,None)
        if not window: raise RuntimeError("Could not create an OpenGL 3.3 window")
        glfw.make_context_current(window); glfw.swap_interval(1)
        ctx=moderngl.create_context(require=330)
        width,height=glfw.get_framebuffer_size(window)
        camera=Camera(position=(0,4.2,12.5),target=(0,0.8,0),aspect=width/max(height,1))
        scene,meshes=build_scene(ctx)
        environment,env_resources=make_environment(ctx)
        renderer=Renderer(ctx,(max(1,width),max(1,height)),shadow_size=2048,
                          bloom=True,exposure=1.0,bloom_strength=0.14)
        renderer.set_environment(environment)
        presets={
          glfw.KEY_1:("Studio", (5.0,4.7,4.3),1.0,0.14),
          glfw.KEY_2:("Overcast", (1.8,2.0,2.5),0.85,0.04),
          glfw.KEY_3:("High contrast", (9.0,6.2,4.0),0.78,0.22),
          glfw.KEY_4:("Neon", (2.0,2.5,8.0),1.2,0.32),
        }
        title="Studio"; dragging=False; last=(0.0,0.0)
        yaw=0.0; pitch=0.22; distance=13.3
        def mouse_button(win,button,action,mods):
            nonlocal dragging,last
            if button==glfw.MOUSE_BUTTON_LEFT:
                dragging=action==glfw.PRESS
                last=glfw.get_cursor_pos(win)
        def cursor(win,x,y):
            nonlocal last,yaw,pitch
            if dragging:
                dx=x-last[0]; dy=y-last[1]
                yaw-=dx*0.006; pitch=np.clip(pitch+dy*0.006,-0.95,0.95)
                camera.orbit(yaw,pitch,distance)
            last=(x,y)
        def scroll(win,xoff,yoff):
            nonlocal distance
            distance=float(np.clip(distance*math.exp(-yoff*0.08),7.0,25.0))
            camera.orbit(yaw,pitch,distance)
        glfw.set_mouse_button_callback(window,mouse_button)
        glfw.set_cursor_pos_callback(window,cursor)
        glfw.set_scroll_callback(window,scroll)
        glfw.set_input_mode(window,glfw.STICKY_KEYS,glfw.TRUE)
        print("Faust PBR lab | drag: orbit | scroll: zoom | 1-4: lighting presets | Esc: quit")
        while not glfw.window_should_close(window):
            glfw.poll_events()
            if glfw.get_key(window,glfw.KEY_ESCAPE)==glfw.PRESS: break
            for key,(name,radiance,exposure,bloom) in presets.items():
                if glfw.get_key(window,key)==glfw.PRESS:
                    scene.directional_light=DirectionalLight(
                        direction=(-0.45,-1.0,-0.35),radiance=radiance)
                    renderer.exposure=exposure; renderer.bloom_strength=bloom; title=name
            w,h=glfw.get_framebuffer_size(window)
            if w<=0 or h<=0: continue
            camera.resize(w,h)
            renderer.render(scene,camera,target_size=(w,h))
            glfw.set_window_title(window,f"Faust | {title} | PBR material laboratory")
            glfw.swap_buffers(window)
    finally:
        if renderer: renderer.release()
        for resource in env_resources:
            resource.release()
        for mesh in meshes:
            mesh.release()
        if ctx: ctx.release()
        if window: glfw.destroy_window(window)
        glfw.terminate()


if __name__=="__main__": main()
