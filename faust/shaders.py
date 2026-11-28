"""GLSL 330 core shaders used by Faust's forward PBR pipeline."""

PBR_VERTEX = """#version 330 core
in vec3 in_position;
in vec3 in_normal;
in vec2 in_uv;
uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;
uniform mat4 u_light_space;
uniform mat3 u_normal_matrix;
out vec3 v_world_pos;
out vec3 v_normal;
out vec2 v_uv;
out vec4 v_shadow_pos;
void main() {
    vec4 world = u_model * vec4(in_position, 1.0);
    v_world_pos = world.xyz;
    v_normal = normalize(u_normal_matrix * in_normal);
    v_uv = in_uv;
    v_shadow_pos = u_light_space * world;
    gl_Position = u_projection * u_view * world;
}
"""

PBR_FRAGMENT = """#version 330 core
#define MAX_POINT_LIGHTS 8
const float PI = 3.14159265359;
in vec3 v_world_pos;
in vec3 v_normal;
in vec2 v_uv;
in vec4 v_shadow_pos;
out vec4 frag_color;
uniform vec3 u_camera_pos;
uniform vec3 u_albedo;
uniform float u_metallic;
uniform float u_roughness;
uniform float u_ao;
uniform vec3 u_emissive;
uniform sampler2D u_albedo_map;
uniform bool u_has_albedo_map;
uniform samplerCube u_irradiance_map;
uniform samplerCube u_prefilter_map;
uniform sampler2D u_brdf_lut;
uniform bool u_has_ibl;
uniform float u_prefilter_max_lod;
uniform vec3 u_sun_direction;
uniform vec3 u_sun_radiance;
uniform bool u_shadows_enabled;
uniform sampler2D u_shadow_map;
uniform vec2 u_shadow_texel;
uniform float u_shadow_bias;
uniform int u_point_count;
uniform vec4 u_point_position_radius[MAX_POINT_LIGHTS];
uniform vec3 u_point_radiance[MAX_POINT_LIGHTS];

float distribution_ggx(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float d = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / max(PI * d * d, 1e-6);
}
float geometry_schlick_ggx(float NdotX, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return NdotX / max(NdotX * (1.0 - k) + k, 1e-6);
}
float geometry_smith(vec3 N, vec3 V, vec3 L, float roughness) {
    return geometry_schlick_ggx(max(dot(N,V),0.0),roughness) *
           geometry_schlick_ggx(max(dot(N,L),0.0),roughness);
}
vec3 fresnel_schlick(float cosTheta, vec3 F0) {
    return F0 + (1.0-F0)*pow(clamp(1.0-cosTheta,0.0,1.0),5.0);
}
vec3 fresnel_roughness(float cosTheta, vec3 F0, float roughness) {
    return F0 + (max(vec3(1.0-roughness),F0)-F0) *
           pow(clamp(1.0-cosTheta,0.0,1.0),5.0);
}
vec3 evaluate_brdf(vec3 N, vec3 V, vec3 L, vec3 albedo, float metallic, float roughness, vec3 radiance) {
    vec3 H = normalize(V+L);
    float NdotL=max(dot(N,L),0.0), NdotV=max(dot(N,V),0.0);
    vec3 F0=mix(vec3(0.04),albedo,metallic);
    vec3 F=fresnel_schlick(max(dot(H,V),0.0),F0);
    float D=distribution_ggx(N,H,roughness);
    float G=geometry_smith(N,V,L,roughness);
    vec3 spec=(D*G*F)/max(4.0*NdotV*NdotL,1e-5);
    vec3 kd=(1.0-F)*(1.0-metallic);
    return (kd*albedo/PI+spec)*radiance*NdotL;
}
float shadow_visibility(vec3 N, vec3 L) {
    if (!u_shadows_enabled) return 1.0;
    vec3 proj=v_shadow_pos.xyz/v_shadow_pos.w;
    proj=proj*0.5+0.5;
    if (proj.z>1.0 || proj.x<0.0 || proj.x>1.0 || proj.y<0.0 || proj.y>1.0) return 1.0;
    float bias=max(u_shadow_bias*(1.0-dot(N,L)),u_shadow_bias*0.25);
    float visibility=0.0;
    for(int x=-1;x<=1;x++) for(int y=-1;y<=1;y++) {
        float closest=texture(u_shadow_map,proj.xy+vec2(x,y)*u_shadow_texel).r;
        visibility += proj.z-bias <= closest ? 1.0 : 0.0;
    }
    return visibility/9.0;
}
void main() {
    vec3 albedo=u_albedo;
    if(u_has_albedo_map) albedo*=texture(u_albedo_map,v_uv).rgb;
    vec3 N=normalize(v_normal), V=normalize(u_camera_pos-v_world_pos);
    float metallic=clamp(u_metallic,0.0,1.0);
    float roughness=clamp(u_roughness,0.045,1.0);
    vec3 Lo=vec3(0.0);
    vec3 sunL=normalize(-u_sun_direction);
    Lo += evaluate_brdf(N,V,sunL,albedo,metallic,roughness,u_sun_radiance) *
          shadow_visibility(N,sunL);
    for(int i=0;i<u_point_count;i++) {
        vec3 delta=u_point_position_radius[i].xyz-v_world_pos;
        float distance=length(delta);
        float radius=u_point_position_radius[i].w;
        float attenuation=pow(clamp(1.0-distance/radius,0.0,1.0),2.0) /
                         max(distance*distance,0.01);
        Lo += evaluate_brdf(N,V,normalize(delta),albedo,metallic,roughness,
                            u_point_radiance[i]*attenuation);
    }
    vec3 F0=mix(vec3(0.04),albedo,metallic);
    vec3 ambient=vec3(0.025)*albedo*(1.0-metallic)*u_ao;
    if(u_has_ibl) {
        vec3 F=fresnel_roughness(max(dot(N,V),0.0),F0,roughness);
        vec3 kd=(1.0-F)*(1.0-metallic);
        vec3 irradiance=texture(u_irradiance_map,N).rgb;
        vec3 diffuse=irradiance*albedo;
        vec3 R=reflect(-V,N);
        vec3 prefiltered=textureLod(u_prefilter_map,R,roughness*u_prefilter_max_lod).rgb;
        vec2 brdf=texture(u_brdf_lut,vec2(max(dot(N,V),0.0),roughness)).rg;
        vec3 specular=prefiltered*(F*brdf.x+brdf.y);
        ambient=(kd*diffuse+specular)*u_ao;
    }
    vec3 color=Lo+ambient+u_emissive;
    frag_color=vec4(color,1.0);
}
"""

FULLSCREEN_VERTEX = """#version 330 core
in vec2 in_position;
in vec2 in_uv;
out vec2 v_uv;
void main(){ v_uv=in_uv; gl_Position=vec4(in_position,0.0,1.0); }
"""

BRIGHT_FRAGMENT = """#version 330 core
in vec2 v_uv; out vec4 frag_color;
uniform sampler2D u_source;
uniform float u_threshold;
void main(){
 vec3 c=texture(u_source,v_uv).rgb;
 float l=max(max(c.r,c.g),c.b);
 frag_color=vec4(c*max(l-u_threshold,0.0)/max(l,1e-5),1.0);
}
"""

BLUR_FRAGMENT = """#version 330 core
in vec2 v_uv; out vec4 frag_color;
uniform sampler2D u_source;
uniform vec2 u_direction;
void main(){
 const float w[5]=float[5](0.227027,0.1945946,0.1216216,0.054054,0.016216);
 vec3 c=texture(u_source,v_uv).rgb*w[0];
 for(int i=1;i<5;i++){
  vec2 o=u_direction*float(i);
  c+=(texture(u_source,v_uv+o).rgb+texture(u_source,v_uv-o).rgb)*w[i];
 }
 frag_color=vec4(c,1.0);
}
"""

COMPOSITE_FRAGMENT = """#version 330 core
in vec2 v_uv; out vec4 frag_color;
uniform sampler2D u_hdr;
uniform sampler2D u_bloom;
uniform float u_exposure;
uniform float u_bloom_strength;
vec3 aces(vec3 x){
 const float a=2.51,b=0.03,c=2.43,d=0.59,e=0.14;
 return clamp((x*(a*x+b))/(x*(c*x+d)+e),0.0,1.0);
}
void main(){
 vec3 hdr=texture(u_hdr,v_uv).rgb+
          texture(u_bloom,v_uv).rgb*u_bloom_strength;
 vec3 mapped=aces(hdr*u_exposure);
 // The default GLFW framebuffer is sRGB when the window requests it.
 // Faust writes linear output and enables GL_FRAMEBUFFER_SRGB when available.
 frag_color=vec4(mapped,1.0);
}
"""

SHADOW_VERTEX = """#version 330 core
in vec3 in_position;
uniform mat4 u_model;
uniform mat4 u_light_space;
void main(){ gl_Position=u_light_space*u_model*vec4(in_position,1.0); }
"""
SHADOW_FRAGMENT = """#version 330 core
void main(){}
"""
