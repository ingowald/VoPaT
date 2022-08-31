// ======================================================================== //
// Copyright 2022-2022 Ingo Wald                                            //
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

#include "vopat/RayForwardingRenderer.h"
#include "vopat/VolumeRendererBase.h"
#include "vopat/SurfaceIntersector.h"
#include "vopat/NextDomainKernel.h"

namespace vopat {

  using namespace owl;
  using namespace owl::common;
  using Random = owl::common::LCG<8>;

  using ForwardGlobals = typename RayForwardingRenderer::Globals;
  using VolumeGlobals  = typename VolumeRenderer::Globals;
  using SurfaceGlobals = typename SurfaceIntersector::Globals;
    
  /*! "triangle mesh" geometry type for shared-faces method */
  struct UMeshGeom {
    vec4i *tets;
    vec3f *vertices;
    float *scalars;
    vec2i *tetsOnFace;
  };
  
  struct LaunchParams {
    static inline __device__ const LaunchParams &get();
    
    ForwardGlobals           forwardGlobals;
    VolumeGlobals            volumeGlobals;
    SurfaceGlobals           surfaceGlobals;
    NextDomainKernel::LPData nextDomainKernel;
  };
  
}
