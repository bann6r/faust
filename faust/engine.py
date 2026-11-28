"""A compact OpenGL 3.3 core renderer for Faust.

The application owns the GLFW window/context. Engine owns renderer-created GPU
objects; CPU scene descriptions can outlive a context and be uploaded later.
Matrices are NumPy row-major and are transposed when uploaded to GLSL.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Optional, Sequence

import moderngl
import numpy as np

from . import shaders


def _v3(value, name="vector"):
    a=np.asarray(value,dtype=np.float32)
    if a.shape!=(3,) or not np.isfinite(a).all():
        raise ValueError(f"{name} must be a finite vec3")
    return a

def _mat4(value,name="matrix"):
    a=np.asarray(value,dtype=np.float32)
    if a.shape!=(4,4) or not np.isfinite(a).all():
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    return a

def _write_matrix(program,name,value):
    program[name].write(np.asarray(value,dtype="f4").T.tobytes())

def normalize(v):
    v=np.asarray(v,dtype=np.float32)
    n=float(np.linalg.norm(v))
    if n<1e-8: raise ValueError("cannot normalize a zero vector")
    return v/n

def look_at(eye,target,up=(0,1,0)):
    eye,target,up=_v3(eye),_v3(target),normalize(up)
    f=normalize(target-eye); s=normalize(np.cross(f,up)); u=np.cross(s,f)
    m=np.eye(4,dtype="f4"); m[0,:3]=s; m[1,:3]=u; m[2,:3]=-f
    m[:3,3]=(-np.dot(s,eye),-np.dot(u,eye),np.dot(f,eye))
    return m

def perspective(fov_degrees,aspect,near=0.1,far=250.0):
    if aspect<=0 or near<=0 or far<=near or not 1<fov_degrees<179:
        raise ValueError("invalid perspective parameters")
    f=1.0/math.tan(math.radians(fov_degrees)*0.5)
    return np.array([[f/aspect,0,0,0],[0,f,0,0],
      [0,0,(far+near)/(near-far),2*far*near/(near-far)],
      [0,0,-1,0]],dtype="f4")

def orthographic(left,right,bottom,top,near,far):
    if right==left or top==bottom or far==near: raise ValueError("invalid orthographic bounds")
    return np.array([[2/(right-left),0,0,-(right+left)/(right-left)],
      [0,2/(top-bottom),0,-(top+bottom)/(top-bottom)],
      [0,0,-2/(far-near),-(far+near)/(far-near)],[0,0,0,1]],dtype="f4")


@dataclass
class Camera:
    position: Sequence[float]=(0.0,2.0,7.0)
    target: Sequence[float]=(0.0,0.8,0.0)
    up: Sequence[float]=(0.0,1.0,0.0)
    fov: float=45.0
    near: float=0.1
    far: float=150.0
    aspect: float=16/9
    def __post_init__(self):
        self.position=_v3(self.position,"position"); self.target=_v3(self.target,"target")
        self.up=normalize(self.up)
    @property
    def view(self): return look_at(self.position,self.target,self.up)
    @property
    def projection(self): return perspective(self.fov,self.aspect,self.near,self.far)
    def resize(self,width,height):
        if width>0 and height>0: self.aspect=width/height; return True
        return False
    def orbit(self,yaw,pitch,distance=None):
        offset=self.position-self.target
        radius=float(distance if distance is not None else np.linalg.norm(offset))
        yaw=float(yaw); pitch=float(np.clip(pitch,-1.48,1.48))
        self.position=self.target+np.array([radius*math.cos(pitch)*math.sin(yaw),
          radius*math.sin(pitch),radius*math.cos(pitch)*math.cos(yaw)],dtype="f4")


@dataclass(frozen=True)
class Material:
    name: str="Default"
    albedo: tuple=(0.8,0.8,0.8)
    metallic: float=0.0
    roughness: float=0.5
    ao: float=1.0
    emissive: tuple=(0.0,0.0,0.0)
    albedo_texture: object=None
    def __post_init__(self):
        for name in ("albedo","emissive"):
            c=_v3(getattr(self,name),name)
            if np.any(c<0): raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self,name,tuple(float(x) for x in c))
        for name in ("metallic","roughness","ao"):
            x=float(getattr(self,name))
            if not math.isfinite(x) or not 0<=x<=1: raise ValueError(f"{name} must be in [0,1]")
            object.__setattr__(self,name,x)


@dataclass(frozen=True)
class DirectionalLight:
    direction: tuple=(-0.5,-1.0,-0.35)
    radiance: tuple=(4.5,4.2,3.8)
    cast_shadows: bool=True
    shadow_extent: float=12.0
    shadow_near: float=0.1
    shadow_far: float=40.0
    shadow_bias: float=0.0015
    def __post_init__(self):
        object.__setattr__(self,"direction",tuple(normalize(_v3(self.direction,"direction"))))
        object.__setattr__(self,"radiance",tuple(_v3(self.radiance,"radiance")))
        if self.shadow_extent<=0 or self.shadow_far<=self.shadow_near: raise ValueError("invalid shadow camera")
    def matrix(self,target=(0,0,0)):
        target=_v3(target)
        eye=target-normalize(self.direction)*self.shadow_extent
        up=(0,0,1) if abs(float(np.dot(normalize(self.direction),(0,1,0))))>0.96 else (0,1,0)
        return orthographic(-self.shadow_extent,self.shadow_extent,-self.shadow_extent,
            self.shadow_extent,self.shadow_near,self.shadow_far) @ look_at(eye,target,up)


@dataclass(frozen=True)
class PointLight:
    position: tuple=(0,2,0)
    radiance: tuple=(10,8,6)
    radius: float=8.0
    def __post_init__(self):
        object.__setattr__(self,"position",tuple(_v3(self.position,"position")))
        object.__setattr__(self,"radiance",tuple(_v3(self.radiance,"radiance")))
        if self.radius<=0: raise ValueError("point-light radius must be positive")


@dataclass
class MeshData:
    vertices: np.ndarray
    indices: Optional[np.ndarray]=None
    name: str="mesh"
    def __post_init__(self):
        v=np.asarray(self.vertices,dtype="f4")
        if v.ndim!=2 or v.shape[1]!=8 or len(v)<3 or not np.isfinite(v).all():
            raise ValueError("vertices must be finite (N,8): position, normal, uv")
        self.vertices=np.ascontiguousarray(v)
        if self.indices is not None:
            i=np.asarray(self.indices,dtype="u4").reshape(-1)
            if len(i)<3 or int(i.max())>=len(v): raise ValueError("invalid mesh indices")
            self.indices=np.ascontiguousarray(i)


class Mesh:
    def __init__(self,ctx,data:MeshData):
        self.ctx=ctx; self.name=data.name; self.count=len(data.indices) if data.indices is not None else len(data.vertices)
        self.vbo=ctx.buffer(data.vertices.tobytes())
        self.ibo=ctx.buffer(data.indices.tobytes()) if data.indices is not None else None
        self._vaos={}; self._released=False
    def vao(self,program):
        key=id(program)
        if key not in self._vaos:
            if "in_normal" in program and "in_uv" in program:
                content=[(self.vbo,"3f 3f 2f","in_position","in_normal","in_uv")]
            else:
                # Depth-only shaders skip the packed normal and UV attributes.
                content=[(self.vbo,"3f 20x","in_position")]
            self._vaos[key]=self.ctx.vertex_array(program,content,self.ibo)
        return self._vaos[key]
    def render(self,program):
        if self._released: raise RuntimeError("mesh has been released")
        self.vao(program).render(mode=moderngl.TRIANGLES)
    def release(self):
        if self._released:return
        for vao in self._vaos.values(): vao.release()
        self._vaos.clear()
        if self.ibo:self.ibo.release()
        self.vbo.release(); self._released=True


@dataclass
class RenderItem:
    mesh: object
    material: Material
    model: np.ndarray=field(default_factory=lambda:np.eye(4,dtype="f4"))
    cast_shadows: bool=True
    receive_shadows: bool=True


class Scene:
    def __init__(self,light=None):
        self.directional_light=light or DirectionalLight()
        self.point_lights=[]; self.objects=[]
    def add(self,mesh,material,model=None,*,cast_shadows=True,receive_shadows=True):
        item=RenderItem(mesh,material,_mat4(np.eye(4) if model is None else model),
                        cast_shadows,receive_shadows)
        self.objects.append(item); return item
    def add_point_light(self,light):
        if len(self.point_lights)>=8: raise ValueError("OpenGL shader supports at most 8 point lights")
        self.point_lights.append(light); return light
    def remove(self,item): self.objects.remove(item)
    def clear(self): self.objects.clear(); self.point_lights.clear()


class Environment:
    """Non-owning IBL inputs: irradiance cubemap, prefiltered cubemap and BRDF LUT.

    Supply linearly encoded floating-point textures generated by an offline
    tool or an asset pipeline. Faust binds them but does not silently own them.
    """
    def __init__(self,irradiance,prefiltered,brdf_lut,max_lod=5.0):
        self.irradiance=irradiance; self.prefiltered=prefiltered
        self.brdf_lut=brdf_lut; self.max_lod=float(max_lod)


class _Target:
    def __init__(self,ctx,size,depth=False):
        self.color=ctx.texture(size,4,dtype="f2"); self.color.filter=(moderngl.LINEAR,moderngl.LINEAR)
        self.depth=ctx.depth_renderbuffer(size) if depth else None
        self.fbo=ctx.framebuffer([self.color],self.depth)
    def release(self):
        self.fbo.release()
        if self.depth:self.depth.release()
        self.color.release()


class Renderer:
    def __init__(self,ctx,size=(1280,720),*,shadow_size=2048,bloom=True,exposure=1.0,bloom_strength=0.08):
        self.ctx=ctx; self.width,self.height=map(int,size)
        if min(self.width,self.height)<=0: raise ValueError("render size must be positive")
        self.exposure=float(exposure); self.bloom_strength=float(bloom_strength); self.bloom_enabled=bloom
        self.pbr=ctx.program(vertex_shader=shaders.PBR_VERTEX,fragment_shader=shaders.PBR_FRAGMENT)
        self.shadow=ctx.program(vertex_shader=shaders.SHADOW_VERTEX,fragment_shader=shaders.SHADOW_FRAGMENT)
        self.fullscreen=ctx.program(vertex_shader=shaders.FULLSCREEN_VERTEX,fragment_shader=shaders.COMPOSITE_FRAGMENT)
        self.bright=ctx.program(vertex_shader=shaders.FULLSCREEN_VERTEX,fragment_shader=shaders.BRIGHT_FRAGMENT)
        self.blur=ctx.program(vertex_shader=shaders.FULLSCREEN_VERTEX,fragment_shader=shaders.BLUR_FRAGMENT)
        quad=np.array([-1,-1,0,0, 1,-1,1,0, 1,1,1,1, -1,-1,0,0, 1,1,1,1, -1,1,0,1],dtype="f4")
        self.quad_buffer=ctx.buffer(quad.tobytes())
        self.quad_vaos=[ctx.vertex_array(p,[(self.quad_buffer,"2f 2f","in_position","in_uv")]) for p in (self.fullscreen,self.bright,self.blur)]
        self.hdr=_Target(ctx,(self.width,self.height),True)
        half=(max(1,self.width//2),max(1,self.height//2))
        self.bloom_a=_Target(ctx,half); self.bloom_b=_Target(ctx,half)
        self.shadow_depth=ctx.depth_texture((shadow_size,shadow_size))
        self.shadow_depth.filter=(moderngl.NEAREST,moderngl.NEAREST)
        self.shadow_depth.compare_func=""
        self.shadow_depth.repeat_x=False; self.shadow_depth.repeat_y=False
        self.shadow_fbo=ctx.framebuffer(depth_attachment=self.shadow_depth)
        self.shadow_size=shadow_size; self.environment=None; self._released=False
        self._set_samplers()
    def _set_samplers(self):
        for n,u in (("u_albedo_map",0),("u_irradiance_map",1),("u_prefilter_map",2),
                    ("u_brdf_lut",3),("u_shadow_map",4)):
            self.pbr[n].value=u
        self.bright["u_source"].value=0; self.blur["u_source"].value=0
        self.fullscreen["u_hdr"].value=0; self.fullscreen["u_bloom"].value=1
    def set_environment(self,environment:Optional[Environment]): self.environment=environment
    def resize(self,width,height):
        width,height=int(width),int(height)
        if width<=0 or height<=0:return False
        if (width,height)==(self.width,self.height):return True
        self.hdr.release(); self.bloom_a.release(); self.bloom_b.release()
        self.width,self.height=width,height
        self.hdr=_Target(self.ctx,(width,height),True)
        half=(max(1,width//2),max(1,height//2))
        self.bloom_a=_Target(self.ctx,half); self.bloom_b=_Target(self.ctx,half)
        return True
    def _draw_shadows(self,scene,light_matrix):
        if not scene.directional_light.cast_shadows:return
        self.shadow_fbo.use(); self.ctx.viewport=(0,0,self.shadow_size,self.shadow_size)
        self.ctx.enable(moderngl.DEPTH_TEST); self.ctx.depth_func="<"
        self.shadow_fbo.clear(depth=1.0)
        for item in scene.objects:
            if not item.cast_shadows:continue
            _write_matrix(self.shadow,"u_model",item.model)
            _write_matrix(self.shadow,"u_light_space",light_matrix)
            item.mesh.render(self.shadow)
    def _draw_scene(self,scene,camera,light_matrix):
        self.hdr.fbo.use(); self.ctx.viewport=(0,0,self.width,self.height)
        self.hdr.fbo.clear(0.012,0.016,0.024,1.0,depth=1.0)
        self.ctx.enable(moderngl.DEPTH_TEST); self.ctx.depth_func="<"
        sun=scene.directional_light
        for item in scene.objects:
            p=self.pbr
            _write_matrix(p,"u_model",item.model); _write_matrix(p,"u_view",camera.view)
            _write_matrix(p,"u_projection",camera.projection); _write_matrix(p,"u_light_space",light_matrix)
            linear=np.asarray(item.model,dtype="f4")[:3,:3]
            if abs(float(np.linalg.det(linear)))<1e-9: raise ValueError("singular model transform")
            p["u_normal_matrix"].write(np.linalg.inv(linear).T.astype("f4").T.tobytes())
            p["u_camera_pos"].value=tuple(float(x) for x in camera.position)
            m=item.material
            p["u_albedo"].value=m.albedo; p["u_metallic"].value=m.metallic
            p["u_roughness"].value=max(0.045,m.roughness); p["u_ao"].value=m.ao
            p["u_emissive"].value=m.emissive
            tex=m.albedo_texture
            p["u_has_albedo_map"].value=tex is not None
            if tex: tex.use(0)
            p["u_sun_direction"].value=sun.direction; p["u_sun_radiance"].value=sun.radiance
            p["u_shadows_enabled"].value=bool(sun.cast_shadows and item.receive_shadows)
            p["u_shadow_texel"].value=(1/self.shadow_size,1/self.shadow_size)
            p["u_shadow_bias"].value=sun.shadow_bias
            self.shadow_depth.use(4)
            env=self.environment
            p["u_has_ibl"].value=env is not None
            if env:
                env.irradiance.use(1); env.prefiltered.use(2); env.brdf_lut.use(3)
                p["u_prefilter_max_lod"].value=env.max_lod
            else:
                p["u_prefilter_max_lod"].value=0.0
            lights=scene.point_lights
            p["u_point_count"].value=len(lights)
            positions=np.zeros((8,4),dtype="f4")
            radiances=np.zeros((8,3),dtype="f4")
            for i,l in enumerate(lights):
                positions[i,:3]=l.position; positions[i,3]=l.radius
                radiances[i]=l.radiance
            p["u_point_position_radius"].write(positions.tobytes())
            p["u_point_radiance"].write(radiances.tobytes())
            item.mesh.render(p)
    def _postprocess(self,output=None):
        self.ctx.disable(moderngl.DEPTH_TEST)
        bloom_tex=self.bloom_b.color
        if self.bloom_enabled:
            self.bright["u_threshold"].value=1.0
            self.hdr.color.use(0); self.bloom_a.fbo.use()
            self.ctx.viewport=(0,0,*self.bloom_a.color.size); self.quad_vaos[1].render()
            source=self.bloom_a
            for i in range(6):
                target=self.bloom_b if i%2==0 else self.bloom_a
                self.blur["u_direction"].value=(1/source.color.size[0],0) if i%2==0 else (0,1/source.color.size[1])
                source.color.use(0); target.fbo.use()
                self.ctx.viewport=(0,0,*target.color.size); self.quad_vaos[2].render()
                source=target
            bloom_tex=source.color
        if output is None: self.ctx.screen.use()
        else: output.use()
        self.ctx.viewport=(0,0,self.width,self.height)
        self.hdr.color.use(0); bloom_tex.use(1)
        self.fullscreen["u_exposure"].value=self.exposure
        self.fullscreen["u_bloom_strength"].value=self.bloom_strength if self.bloom_enabled else 0.0
        self.quad_vaos[0].render()
    def render(self,scene,camera,*,target_size=None,output=None):
        if self._released:raise RuntimeError("renderer has been released")
        if target_size:self.resize(*target_size); camera.resize(*target_size)
        light_matrix=scene.directional_light.matrix()
        self._draw_shadows(scene,light_matrix)
        self._draw_scene(scene,camera,light_matrix)
        self._postprocess(output)
    def release(self):
        if self._released:return
        for r in (self.shadow_fbo,self.shadow_depth,self.hdr,self.bloom_a,self.bloom_b,
                  *self.quad_vaos,self.quad_buffer,self.fullscreen,self.bright,self.blur,
                  self.shadow,self.pbr):
            if r is not None:r.release()
        self._released=True


def cube_data():
    """36 vertices, outward-facing normals, UVs."""
    faces=[((-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1),(0,0,1)),
      ((1,-1,-1),(-1,-1,-1),(-1,1,-1),(1,1,-1),(0,0,-1)),
      ((1,-1,1),(1,-1,-1),(1,1,-1),(1,1,1),(1,0,0)),
      ((-1,-1,-1),(-1,-1,1),(-1,1,1),(-1,1,-1),(-1,0,0)),
      ((-1,1,1),(1,1,1),(1,1,-1),(-1,1,-1),(0,1,0)),
      ((-1,-1,-1),(1,-1,-1),(1,-1,1),(-1,-1,1),(0,-1,0))]
    out=[]
    for a,b,c,d,n in faces:
        for p in (a,b,c,a,c,d):out.extend((*p,*n,(p[0]+1)*.5,(p[1]+1)*.5))
    return MeshData(np.asarray(out,dtype="f4").reshape(-1,8),name="cube")

def sphere_data(segments=48,rings=24):
    verts=[]; indices=[]
    for y in range(rings+1):
        v=y/rings; phi=v*math.pi
        for x in range(segments+1):
            u=x/segments; theta=u*2*math.pi
            n=(math.sin(phi)*math.cos(theta),math.cos(phi),math.sin(phi)*math.sin(theta))
            verts.append((*n,*n,u,v))
    for y in range(rings):
        for x in range(segments):
            a=y*(segments+1)+x; b=a+segments+1
            indices.extend((a,b,a+1,a+1,b,b+1))
    return MeshData(np.asarray(verts,dtype="f4"),np.asarray(indices,dtype="u4"),"sphere")
