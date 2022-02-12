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
    PING; fflush(0);
    PRINT(comm);
    PRINT(comm->numWorkers());
    if (isMaster()) {
    } else {
      tallies.resize(comm->numWorkers());
      globals.tallies = tallies.get();
    }
  }
  
  void OptixRenderer::resizeFrameBuffer(const vec2i &newSize)
  {
    PING; fflush(0);
    AddWorkersRenderer::resizeFrameBuffer(newSize);
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


  __global__ void renderFrame(Globals globals)
  {
    int ix = threadIdx.x + blockIdx.x*blockDim.x;
    int iy = threadIdx.y + blockIdx.y*blockDim.y;
    if (ix == 0 && iy == 0)
      printf("render %i %i ptr %lx\n",
             globals.fbSize.x,globals.fbSize.y,
             globals.fbPointer);
  }
  
  void OptixRenderer::renderLocal()
  {
    vec2i blockSize(16);
    vec2i numBlocks = divRoundUp(islandFbSize,blockSize);
    renderFrame<<<numBlocks,blockSize>>>(globals);
    CUDA_SYNC_CHECK();
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
