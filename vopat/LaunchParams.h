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

#include "common.h"
#include "owl/common/math/random.h"
#include "owl/common/math/box.h"
#include "render/RayForwardingRenderer.h"
#include "render/VolumeRendererBase.h"
#include "render/SurfaceIntersector.h"
#include "render/NextDomainKernel.h"

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
    ForwardGlobals         forwardGlobals;
    VolumeGlobals          volumeGlobals;
    SurfaceGlobals         surfaceGlobals;
    NextDomainKernel::DD   nextDomainKernel;
    // OptixTraversableHandle umeshSampleBVH;
    // struct {
    //   uint32_t *pointer;
    //   float4   *accum;
    //   vec2i     size;
    // } fb;
    // struct {
    //   vec3f lens_00;
    //   vec3f dir_00;
    //   vec3f dir_du;
    //   vec3f dir_dv;
    // } camera;
    // int       accumID;
    // OptixTraversableHandle world;
  };
  
}
