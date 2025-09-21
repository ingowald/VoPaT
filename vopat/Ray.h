// ======================================================================== //
// Copyright 2022++ Ingo Wald                                               //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#pragma once

#include "common/vopat.h"

namespace vopat {

  inline __device__
  float fixDir(float f) { return (f==0.f)?1e-6f:f; }
  
  inline __device__
  vec3f fixDir(vec3f v)
  { return {fixDir(v.x),fixDir(v.y),fixDir(v.z)}; }

  // ==================================================================

  struct Ray {
    enum { HitType_None = 0,
           HitType_Volume,
           HitType_Surf_Glass,
           HitType_Surf_Diffuse
    };

    struct {
      /*! note this always refers to a GLOBAL pixel ID even if we
        use islands; ie, this number may be LARGER than the number
        of pixels in the local frame buffer */
      uint32_t    pixelID    : 22;
      uint32_t    numBounces :  4;
      uint32_t    dbg        :  1;
      uint32_t    crosshair  :  1;
      uint32_t    isShadow   :  1;
      uint32_t    hitType    :  2;
    };
    vec3f       origin;
    float tMax = 1e20f;
    
    inline __device__ vec3f getOrigin() const { return origin; }
    inline __device__ void setOrigin(vec3f org) { origin = org; }
    
#if 1
    inline __device__ void setDirection(vec3f v) { direction = to_half(fixDir(normalize(v))); }
    inline __device__ vec3f getDirection() const { return from_half(direction); }
    small_vec3f direction;
#else
    inline __device__ void setDirection(vec3f v) { direction = fixDir(normalize(v)); }
    inline __device__ vec3f getDirection() const { return direction; }
    vec3f direction;
#endif
    small_vec3f throughput;
    
    /* node that spawned this ray, so later stages can reconstruct
       where this ray has already been */
    int16_t spawningRank;
    // int16_t dbg_destRank;
    union {
      struct { small_vec3f color; } volume;
      struct { small_vec3f N; half ior; } surf_glass;
      struct { small_vec3f N; small_vec3f color; } surf_diffuse;
    } hit;

    int dbg_srcRank;
    int dbg_dstRank;
    int dbg_srcIndex;
  };

  // ==================================================================

  struct Intersection {
    enum Type { NONE=0, VOLUME, ISO, /*TODO:*/Mesh };

    /*! surface type that we intersected with */
    Type type = NONE;

    /*! t of ray/surface intersection; FLT_MAX: inval */
    float t;

    union {
      struct { int primID; } surf;
    };
    
  };


  // ==================================================================

  inline __device__ bool checkOrigin(float x)
  {
    if (isnan(x) || fabsf(x) > 1e4f)
      return false;
    return true;
  }
  
  inline __device__ bool checkOrigin(vec3f org)
  { return checkOrigin(org.x) && checkOrigin(org.y) && checkOrigin(org.z); }

  inline __device__ bool checkOrigin(const Ray &ray)
  { return checkOrigin(ray.origin); }

  
  


  inline __device__
  bool boxTest(box3f box,
               vec3f org,
               vec3f dir,
               float &t0,
               float &t1,
               bool dbg=false)
  {
    vec3f t_lo = (box.lower - org) * rcp(dir);
    vec3f t_hi = (box.upper - org) * rcp(dir);

    vec3f t_nr = min(t_lo,t_hi);
    vec3f t_fr = max(t_lo,t_hi);

    t0 = max(t0,reduce_max(t_nr));
    t1 = min(t1,reduce_min(t_fr));
    if (dbg)
      printf("testing ray (%f %f %f)(%f %f %f)\n box (%f %f %f)(%f %f %f)\n -> t_lo %f %f %f t_hi %f %f %f \n -> t_nr %f %f %f t_fr %f %f %f -> t0 %f t1 %f\n",
             org.x,
             org.y,
             org.z,
             dir.x,
             dir.y,
             dir.z,
             box.lower.x,
             box.lower.y,
             box.lower.z,
             box.upper.x,
             box.upper.y,
             box.upper.z,
             t_lo.x,
             t_lo.y,
             t_lo.z,
             t_hi.x,
             t_hi.y,
             t_hi.z,
             t_nr.x,
             t_nr.y,
             t_nr.z,
             t_fr.x,
             t_fr.y,
             t_fr.z,
             t0,t1
             );

    return (t0 <= t1);
  }


  inline __device__
  bool boxTest(box3f box,
               Ray ray,
               float &t0,
               float &t1,
               bool dbg=false)
  {
    vec3f dir = ray.getDirection();

    if (dbg)
      printf(" ray (%f %f %f)(%f %f %f) box (%f %f %f)(%f %f %f)\n",
             ray.origin.x,
             ray.origin.y,
             ray.origin.z,
             dir.x,
             dir.y,
             dir.z,
             box.lower.x,
             box.lower.y,
             box.lower.z,
             box.upper.x,
             box.upper.y,
             box.upper.z);

    return boxTest(box,ray.origin,dir,t0,t1,dbg);
  }

  
}

