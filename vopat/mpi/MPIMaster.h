// ======================================================================== //
// Copyright 2018-2020 Ingo Wald                                            //
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

#include "vopat/mpi/MPICommon.h"
#include "vopat/render/Renderer.h"

namespace vopat {

#define USE_APP_FB 1
  
  struct MPIMaster : public MPICommon {
    MPIMaster(MPIBackend &mpi);

    void resize(const vec2i &newSize);
    void screenShot();
    void resetAccumulation();
    void terminate();
    void renderFrame(uint32_t *fbPointer);
    void resizeFrameBuffer(const vec2i &newSize);
    void setCamera(const Camera &camera);
    void setShadeMode(int shadeMode);
    void setNodeSelection(int nodeSelection);

    /*! collect the individual ranks partial (but fully composited)
        results and put them back together into a single frmae
        buffer */
    void collectRankResults(uint32_t *fbPointer);
  
    const uint32_t *getFB() const
    {
#if USE_APP_FB
      return appFB;
#else
      return fullyAssembledFrame.data();
#endif
    }

    MPIBackend &mpi;
    vec2i fbSize { -1,-1 };

#if USE_APP_FB
    uint32_t *appFB = nullptr;
#else
    std::vector<uint32_t> fullyAssembledFrame;
#endif
  };
  
} // ::vopat
