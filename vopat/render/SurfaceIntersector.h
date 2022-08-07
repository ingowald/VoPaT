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

#include <cfloat>

namespace vopat {

  struct Intersection {
    enum Type { NONE=0, VOLUME, ISO, /*TODO:*/Mesh };

    /*! surface type that we intersected with */
    Type type = NONE;

    /*! t of ray/surface intersection; FLT_MAX: inval */
    float t = FLT_MAX;
    
    /*! intersection position */
    // vec3f isectPos;

    // /* primitive ID, -1 if not applicable */
    // int primID;

    // /* mesh ID, -1 if not applicable */
    // int meshID;

    /*! geometric normal */
    vec3f Ng;

    /*! shading normal */
    // vec3f Ns;
    
    /*! diffuse RGB color */
    vec3f kd;

    /*! sample an outgoing directoin (based on what hit it was), and return PDF of this sampl e*/
    inline __device__
    float sample(Random &rnd,
                 vec3f &outDir,
                 bool dbg=false);
    
    /*! evaluate BRDF or phase function for given direction */
    inline __device__
    vec3f eval(const vec3f outDir, bool dbg=false);
  };

  
  inline __device__
  vec3f Intersection::eval(const vec3f outDir, bool dbg)
  {
    // if (dbg)
    //   printf("eval: dir = %f %f %f, Ng = %f %f %f, kd = %f %f %f\n",
    //          outDir.x,
    //          outDir.y,
    //          outDir.z,
    //          Ng.x,
    //          Ng.y,
    //          Ng.z,
    //          kd.x,
    //          kd.y,
    //          kd.z);
    if (type == VOLUME)
      return kd;
    else
      return kd * max(0.f,dot(Ng,outDir));
  }

  inline __device__
  float Intersection::sample(Random &rnd,
                             vec3f &outDir,
                             bool dbg)
  {
    if (type == NONE)
      return 0.f;

    // oh gawd this is a horrible way of doing this:
    do {
      outDir.x = 2.f*rnd() - 1.f;
      outDir.y = 2.f*rnd() - 1.f;
      outDir.z = 2.f*rnd() - 1.f;
      // if (dbg) printf(" sample try %f %f %f\n",
      //                 outDir.x,
      //                 outDir.y,
      //                 outDir.z);
    } while (dot(outDir,outDir) > 1.f);
    outDir = normalize(outDir);

    if (type == VOLUME) {
      // use this random direction ....
    } else {
      // make sure it faces to the right side:
      if (dot(outDir,Ng) < 0.f)
        outDir = - outDir;
    }

    // if (dbg) printf(" sample outDir %f %f %f\n",
    //                 outDir.x,
    //                 outDir.y,
    //                 outDir.z);
      
    return 1.f;
  }
  


  struct SurfaceIntersector {

    struct Globals {

      /*! my *lcoal* per-rank voxel data */
#if VOPAT_UMESH
      UMeshData umesh;
#else
      VoxelData volume;
#endif

      constexpr static unsigned MaxISOs = 4;

      struct {
        int numActive;
        int *active;
        float *values;
        vec3f *colors;
      } iso;

      vec3f gradientDelta;

      int    islandRank;
      box3f *rankBoxes;
      int    numRanks;
      box3f  myRegion;

      inline __device__ void intersect(Intersection &dg,
                                       const vec3f org,
                                       const vec3f dir,
                                       float t0,
                                       float t1,
                                       bool dbg=false) const
      {

#if VOPAT_UMESH
        auto &volume = umesh;
#endif
        
        t1 = min(t1,dg.t);
        if (t1 <= t0) return;

        const float dt = .5f;

        float t_next = t0;
        float v_next = FLT_MAX;
        volume.sample(v_next,org+t_next*dir-myRegion.lower,dbg);

        float t_next_step = int(t0 / dt) * dt + dt;
        if (t_next_step < t0) t_next_step = t0+dt;
        
        while (true) {
          const float t_last = t_next;
          const float v_last = v_next;

          t_next = min(t1,t_next_step);
          if (t_next <= t_last) break;
          t_next_step += dt;
          
          volume.sample(v_next,org+t_next*dir-myRegion.lower,dbg);
          
          // if (dbg) printf("  t (%f %f) val (%f %f) iso %f\n",
          //                 t_last,t_next,v_last,v_next,iso.values[0]);
          if (v_next == v_last)
            continue;
          
          for (int i=0; i<MaxISOs; ++i) {
            if (!iso.active[i]) continue;
            if (isnan(iso.values[i])) continue;
            if (min(v_next,v_last) >= iso.values[i]) continue;
            if (max(v_next,v_last) <= iso.values[i]) continue;
            
            float alpha = (iso.values[i]-v_last)/(v_next-v_last);
            if (isnan(alpha)) continue;
            // float tIso = min(max(t_last * (t_next-t_last),t_last),t_next);
            float tIso = (1.f-alpha)*t_last + alpha*t_next;
            if (tIso >= dg.t) continue;

            const vec3f isopt = org + tIso * dir - myRegion.lower;
            float v = 0.f;
            if (!volume.sample(v,isopt,dbg))
              continue;
            vec3f grad(0.f);
            if (!volume.gradient(grad,isopt,gradientDelta,dbg))
              continue;
            if (isnan(grad.x) ||
                isnan(grad.y) ||
                isnan(grad.z))
              return;
              
            if (dot(grad,grad) < 1e-10f)
              grad = -dir;
            dg.type   = Intersection::ISO;
            dg.t      = tIso;
            dg.Ng     = normalize(grad);
            dg.kd     = iso.colors[i];
            t1 = min(t1,dg.t);
          }
          
        }
      }
    };

    Globals globals;
    CUDAArray<int>   isoActive;
    CUDAArray<float> isoValues;
    CUDAArray<vec3f> isoColors;

    void setISO(int numActive,
                const std::vector<int> &active,
                const std::vector<float> &values,
                const std::vector<vec3f> &colors)
    {
      globals.iso.numActive = numActive;
      isoActive.upload(active);
      isoValues.upload(values);
      isoColors.upload(colors);
    }

  };

} // ::vopat

