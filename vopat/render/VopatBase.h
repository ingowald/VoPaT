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

#include "VolumeRendererBase.h"
#include "SurfaceIntersector.h"
#include "Ray.h"
#include "vopat/LaunchParams.h"

namespace vopat {

  struct Vopat
  {
    static inline __device__
    Ray generateRay(const ForwardGlobals &globals,
                    vec2i pixelID,
                    vec2f pixelPos);

    static  inline __device__
    int computeNextNode(const VolumeGlobals &vopat,
                        const Ray &ray,
                        const float t_already_travelled,
                        bool dbg);
    static inline __device__
    int computeInitialRank(const VolumeGlobals &vopat,
                           Ray ray,
                           bool dbg=false);

    static inline __device__
    vec3f backgroundColor(const Ray &ray,
                          const ForwardGlobals &globals)
    {
      int iy = ray.pixelID / globals.worldFbSize.x;
      float t = iy / float(globals.worldFbSize.y);
      const vec3f c = (1.0f - t)*vec3f(1.0f, 1.0f, 1.0f) + t * vec3f(0.5f, 0.7f, 1.0f);
      return c;
    }
  };


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
    
    vec3f t_lo = (box.lower - ray.origin) * rcp(dir);
    vec3f t_hi = (box.upper - ray.origin) * rcp(dir);
    
    vec3f t_nr = min(t_lo,t_hi);
    vec3f t_fr = max(t_lo,t_hi);

    t0 = max(t0,reduce_max(t_nr));
    t1 = min(t1,reduce_min(t_fr));
    if (dbg) printf("  -> t0 %f t1 %f\n",t0,t1);
    return (t0 <= t1);
  }

  
  inline __device__
  Ray Vopat::generateRay(const ForwardGlobals &globals,
                         vec2i pixelID,
                         vec2f pixelPos)
  {
    Ray ray;
    ray.pixelID  = pixelID.x + globals.worldFbSize.x*pixelID.y;
    ray.isShadow = false;
    ray.origin = globals.camera.lens_00;
    vec3f dir
      = globals.camera.dir_00
      + globals.camera.dir_du * (pixelID.x+pixelPos.x)
      + globals.camera.dir_dv * (pixelID.y+pixelPos.y);
    ray.setDirection(dir);
    ray.throughput = to_half(vec3f(1.f));
    return ray;
  }

  inline __device__
  int Vopat::computeNextNode(const VolumeGlobals &vopat,
                             const Ray &ray,
                             const float t_already_travelled,
                             bool dbg)
  {
    if (dbg) printf("finding next that's t >= %f and rank != %i\n",
                    t_already_travelled,vopat.islandRank);
      
    int closest = -1;
    float t_closest = CUDART_INF;
    for (int i=0;i<vopat.islandSize;i++) {
      if (i == vopat.islandRank) continue;
        
      float t0 = t_already_travelled * (1.f+1e-5f);
      float t1 = t_closest; 
      if (!boxTest(vopat.rankBoxes[i],ray,t0,t1,dbg))
        continue;
      // if (t0 == t1)
      //   continue;
      
      if (dbg) printf("   accepted rank %i dist %f\n",i,t0);
      t_closest = t0;
      closest = i;
    }
    if (ray.dbg) printf("(%i) NEXT rank is %i\n",vopat.islandRank,closest);
    return closest;
  }

  inline __device__
  int Vopat::computeInitialRank(const VolumeGlobals &vopat,
                                Ray ray,
                                bool dbg)
  {
    int closest = -1;
    float t_closest = CUDART_INF;
    for (int i=0;i<vopat.islandSize;i++) {
      float t_min = 0.f;
      float t_max = t_closest;
      if (!boxTest(vopat.rankBoxes[i],ray,t_min,t_max))
        continue;
      closest = i;
      t_closest = t_min;
    }
    // if (ray.dbg) printf("(%i) INITIAL rank is %i\n",vopat.myRank,closest);
    return closest;
  }
  
} // ::vopat


  
