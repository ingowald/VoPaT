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

#include "vopat/render/Renderer.h"
#include "vopat/model/Model.h"

namespace vopat {

  struct Globals {
    int          sampleID;
    Camera       camera;
    vec2i        fbSize;
    small_vec3f *fbPointer;
    int         *tallies;
  };

  struct OptixRenderer : public AddWorkersRenderer {
    OptixRenderer(CommBackend *comm,
                  Model::SP model,
                  int numSPP);

    // ==================================================================
    // main things we NEED to implement
    // ==================================================================
    
    /*! asks the _workers_ to render locally; will not get called on
        master, but assumes that all workers render their local frame
        buffer(s) that we'll then merge */
    void renderLocal() override;
    void screenShot() override;

    // ==================================================================
    // things we intercept to know what to do
    // ==================================================================
    
    void resizeFrameBuffer(const vec2i &newSize)  override;
    // void resetAccumulation()  override;
    // void setCamera(const Camera &camera)  override;

    // void composeRegion(const vec2i &ourRegionSize,
    //                    const small_vec3f *compInputs,
    //                    uint32_t *compOutputs);
    
    CUDAArray<int>         tallies;
    Globals globals;
  };
  
}
