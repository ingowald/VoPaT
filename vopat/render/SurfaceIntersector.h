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

      template <typename RayType>
      inline __device__ Surflet intersect(const RayType &ray) const
      {
        Surflet res{Surflet::None,FLT_MAX,vec3f(0.f),-1,-1,vec3f(0.f),vec3f(0.f)};

        return res;
      }
    };

    Globals globals;
  };

} // ::vopat

