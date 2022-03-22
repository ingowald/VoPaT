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

  struct Surflet {
    enum Type { /*TODO:*/Mesh, ISO, None, };

    /*! surface type that we intersected with */
    Type type;

    /*! t of ray/surface intersection; FLT_MAX: inval */
    float t;

    /*! intersection position */
    vec3f isectPos;

    /* primitive ID, -1 if not applicable */
    int primID;

    /* mesh ID, -1 if not applicable */
    int meshID;

    /*! geometric normal */
    vec3f gn;

    /*! shading normal */
    vec3f sn;

    /*! diffuse RGB color */
    vec3f kd;

    inline __device__ bool wasHit() const
    {
      return t < FLT_MAX;
    }
  };


  struct SurfaceIntersector {

    struct Globals {

      /*! my *lcoal* per-rank voxel data */
      VoxelData volume;

      constexpr static unsigned MaxISOs = 4;

      struct {
        float *values;
        int *active;
      } iso;

      vec3f gradientDelta;

      int    myRank;
      box3f *rankBoxes;
      int    numRanks;
      box3f  myRegion;

      template <typename RayType>
      inline __device__ Surflet intersect(const RayType &ray,
                                          const float tmin = 0.f,
                                          const float tmax = FLT_MAX) const
      {
        Surflet res{Surflet::None,FLT_MAX,vec3f(0.f),-1,-1,vec3f(0.f),vec3f(0.f)};

        // ISOs
        {
          const float dt = .5f;
          Surflet resISO{Surflet::None,FLT_MAX,vec3f(0.f),-1,-1,vec3f(0.f),vec3f(0.f)};
          float t0 = 0.f;
          while (t0 < tmin) t0 += dt;
          float t1 = 0.f;
          while (t1 < tmax) t1 += dt;
          if (ray.dbg) {
            printf("Integrating ISOs, t0=%f, t1=%f\n",t0,t1);
          }
          float t_last = t0;
          const vec3f p_last = ray.origin + t_last * ray.getDirection() - myRegion.lower;
          float v_last = 0.f;
          volume.sample(v_last,p_last,ray.dbg);
          if (ray.dbg) {
            printf("ISO sample pos: (%f,%f,%f) (t=%f), value: %f\n",p_last.x,p_last.y,p_last.z,
                   t_last,v_last);
          }
          float t_i = t0+dt;
          for (;true;t_i += dt) {
            const float t_next = min(t_i,t1);
            const vec3f pos = ray.origin + t_next * ray.getDirection() - myRegion.lower;
            float v_next = 0.f;
            if (!volume.sample(v_next,pos,ray.dbg)) {
              break;
            }
            if (ray.dbg) {
              printf("ISO sample pos: (%f,%f,%f) (t=%f), value: %f\n",pos.x,pos.y,pos.z,
                     t_next,v_next);
            }
            if (isnan(v_next) || isnan(v_last))
              break;

            bool wasHit = false;
            for (int i=0; i<MaxISOs; ++i) {
              if (iso.active[i] && min(v_next,v_last) <= iso.values[i]
                                && max(v_next,v_last) >= iso.values[i])
              {
                float alpha = (iso.values[i]-v_last)/(v_next-v_last);
                float thit = min(max(t_last * (t_next-t_last),t_last),t_next);

                const vec3f isopt = ray.origin + thit * ray.getDirection() - myRegion.lower;
                float v = 0.f;
                if (!volume.sample(v,isopt,ray.dbg))
                  continue;
                vec3f grad(0.f);
                if (!volume.gradient(grad,isopt,gradientDelta,ray.dbg))
                  continue;
                if (dot(grad,grad) < 1e-10f)
                  grad = -ray.getDirection();
                vec3f N = normalize(grad);
                // face-forward
                if (dot(N,ray.getDirection()) > 0.f)
                  N = -N;

                wasHit = true;
                resISO.type = Surflet::ISO;
                resISO.t        = thit;
                resISO.isectPos = isopt;
                resISO.gn       = N;
                resISO.sn       = N;
                resISO.kd       = vec3f(.8f);
                break;
              }
            }

            if (wasHit)
              break;

            t_last = t_next;
            v_last = v_next;
            if (t_next >= t1) break;
          }

          if (resISO.t < res.t) res = resISO;
        }

        return res;
      }
    };

    Globals globals;
    CUDAArray<float> isoValues;
    CUDAArray<int>   isoActive;
  };

} // ::vopat

