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

#include "vopat/common.h"

namespace vopat {

  /*! captures all the rendering-specific config such as transfer
      fucntion, iso-values, etc; to allow user to store/freeze a
      config and load that upon next run. note this also includes some
      gui-speicific values that the renderer may not even know
      about */
  struct ModelConfig {
    struct {
      interval<float>    absDomain;
      interval<float>    relDomain;
      std::vector<vec4f> colorMap;
      float              opacityScale;
    } xf;
    struct {
      vec3f from, at, up;
      float fovy;
    } camera;

    void save(const std::string &fileName);
    static ModelConfig load(const std::string &fileName);
  };
  
}
