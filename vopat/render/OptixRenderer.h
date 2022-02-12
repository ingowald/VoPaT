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

  template<typename T>
  struct CUDAArray {

    void resize(size_t N)
    {
      if (this->N == N) return;
      this->N = N;
      if (devMem) CUDA_CALL(Free(devMem));
      devMem = 0;
      CUDA_CALL(MallocManaged(&devMem,N*sizeof(T)));
    }
    
    T *get() const { return devMem; }
    
    T     *devMem = 0;
    size_t N      = 0;
  };

  struct Globals {
    int          sampleID;
    Camera       camera;
    vec2i        fbSize;
    small_vec3f *fbPointer;
    int         *tallies;
  };

  struct OptixRenderer : public Renderer {
    OptixRenderer(CommBackend *comm,
                  Model::SP model,
                  int numSPP);
    void render()  override;
    void resizeFrameBuffer(const vec2i &newSize)  override;
    void resetAccumulation()  override;
    void setCamera(const Camera &camera)  override;

    void composeRegion(const vec2i &ourRegionSize,
                       const small_vec3f *compInputs,
                       uint32_t *compOutputs);

    vec2i fbSize;
    CUDAArray<small_vec3f> localFB;
    CUDAArray<uint32_t>    compOutputs;
    CUDAArray<int>         tallies;
    Globals globals;
  };
  
}
