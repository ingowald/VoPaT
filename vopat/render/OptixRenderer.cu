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

#include "vopat/render/OptixRenderer.h"

namespace vopat {

  OptixRenderer::OptixRenderer(CommBackend *comm,
                               Model::SP model,
                               int numSPP)
    : AddWorkersRenderer(comm)
  {
    tallies.resize(comm->numWorkers());
    globals.tallies = tallies.get();
  }
  
  void OptixRenderer::resizeFrameBuffer(const vec2i &newSize)
  {
    if (isMaster()) {
    } else {
      // localFB.resize(newSize.x*newSize.y);
      globals.fbPointer = localFB.get();
      globals.fbSize    = fbSize;
    }
  }
  
  // void OptixRenderer::resetAccumulation()
  // {
  //   globals.sampleID = -1;
  // }
  
  // void OptixRenderer::setCamera(const Camera &camera)
  // {
  //   globals.camera = camera;
  // }


  __global__ void render(Globals &globals)
  {
  }
  
  void OptixRenderer::renderLocal()
  {
  }
  
  // void OptixRenderer::render(uint32_t *appFB)
  // {
  //   globals.sampleID++;
  //   if (isMaster()) {
  //     PING;
  //     PRINT(globals.fbPointer);
  //     PRINT(globals.fbSize);
  //   } else {
  //     workerRender();
  //   }
  // }

  void OptixRenderer::screenShot()
  {
    PING;
  }

  
}
