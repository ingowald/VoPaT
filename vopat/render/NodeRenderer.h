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

#include "owl/owl_device.h"
#include "vopat/render/VopatRenderer.h"
#include "owl/common/math/random.h"

namespace vopat {

  using Random = owl::common::LCG<8>;

  inline __device__ vec3f lightColor() { return vec3f(1.f,1.f,1.f); }
  inline __device__ vec3f lightDirection()
  {
    return vec3f(1.f,.1f,.5f);
    // return (0.f,0.f,1.f);
  }

  inline __device__
  float fixDir(float f) { return (f==0.f)?1e-6f:f; }
  
  inline __device__
  vec3f fixDir(vec3f v)
  { return {fixDir(v.x),fixDir(v.y),fixDir(v.z)}; }
  
}
