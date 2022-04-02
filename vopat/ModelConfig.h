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
#include "vopat/mpi/MPIRenderer.h"

namespace vopat {

  /*! captures all the rendering-specific config such as transfer
      fucntion, iso-values, etc; to allow user to store/freeze a
      config and load that upon next run. note this also includes some
      gui-speicific values that the renderer may not even know
      about */
  struct ModelConfig {
    typedef std::shared_ptr<ModelConfig> SP;
    
    enum { maxDirectionalLights = 2 };

    ModelConfig()
    {
      lights.directional.push_back({vec3f(.1,1.f,.1f),vec3f(1.f)});
    }

    ModelConfig(const ModelConfig &) = default;
    ModelConfig &operator=(const ModelConfig &) = default;

    // ==================================================================
    struct {
      inline interval<float> getRange() const {
        return {
                absDomain.lower + (relDomain.lower/100.f) * (absDomain.upper-absDomain.lower),
                absDomain.lower + (relDomain.upper/100.f) * (absDomain.upper-absDomain.lower)
        };
      }
      inline float getDensity() const
      { return powf(1.1f,opacityScale-100); }
        
      interval<float>    absDomain;
      interval<float>    relDomain { 0.f, 100.f };
      std::vector<vec4f> colorMap;
      float              opacityScale { 100.f };
    } xf;

    // ==================================================================
    struct {
      vec3f from, at, up { 0, 0, 0 };
      float fovy = 70.f;
    } camera;

    // ==================================================================
    struct {
      float ambient = .03f;
      std::vector<MPIRenderer::DirectionalLight> directional;
    } lights;

    void save(const std::string &fileName);
    static ModelConfig load(const std::string &fileName);
  };
  
}
