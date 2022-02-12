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

#include "vopat/render/Renderer.h"

namespace vopat {
  
  std::string Renderer::screenShotFileName = "vopat";
  
  Renderer::Renderer(CommBackend *comm)
    : comm(comm)
  {
    cudaFree(0);
  }


  /*! the CUDA "parallel_for" variant */
  __global__ void cudaComposeRegion(const small_vec3f *compInputs,
                                    const vec2i ourRegionSize,
                                    const int numInputsPerPixel,
                                    // const int numSamplesPerPixel,
                                    uint32_t *compOutputs,
                                    bool doToneMapping)
  {
    const int our_y = threadIdx.y+blockIdx.y*blockDim.y;
    if (our_y >= ourRegionSize.y) return;
    const int our_x = threadIdx.x+blockIdx.x*blockDim.x;
    if (our_x >= ourRegionSize.x) return;

    vec3f sum = 0.f;
    for (int i=0;i<numInputsPerPixel;i++) {
      const int iy = our_y + i * ourRegionSize.y;
      vec3f in = from_half(compInputs[our_x + iy * ourRegionSize.x]);
      sum += in;
    }
    if (doToneMapping) {
      const float rate = .7f;
      sum.x = powf(sum.x,rate);
      sum.y = powf(sum.y,rate);
      sum.z = powf(sum.z,rate);
    }
              
    compOutputs[our_x+our_y*ourRegionSize.x]
      = make_rgba(sum);
  }
  
  
  void AddWorkersRenderer::composeRegion(uint32_t *results,
                                         const vec2i &ourRegionSize,
                                         const small_vec3f *inputs,
                                         int islandSize)
  {
    vec2i tileSize = 32;
    const vec2i numTiles = divRoundUp(ourRegionSize,tileSize);
    // const int islandSize = comm->worker.withinIsland->size;
    if (area(numTiles) > 0)
      cudaComposeRegion<<<tileSize,numTiles>>>
        (inputs,ourRegionSize,islandSize,
         results,
         false);
  }
  
  void AddWorkersRenderer::resizeFrameBuffer(const vec2i &newSize)
  {
    Renderer::resizeFrameBuffer(newSize);

    if (comm->isMaster) {
    } else {
      // ==================================================================
      // this upper part should be per node, shared among all threads
      // ==================================================================
      const int numIslands = comm->worker.numIslands;
      const int islandIdx  = comm->worker.islandIdx;
      const int islandRank = comm->worker.withinIsland->rank;
      const int islandSize = comm->worker.withinIsland->size;
    
      // ------------------------------------------------------------------
      // resize the full frame buffer - only the master needs the final
      // frame buffer, but we need it to compute our local island frame
      // buffer, as well as compositing buffer sizes
      // ------------------------------------------------------------------
      fullFbSize = newSize;
    
      // ------------------------------------------------------------------
      // compute size of our island's frame buffer sub-set, and allocate
      // local accum buffer of that size
      // ------------------------------------------------------------------
    
      islandFbSize.x = fullFbSize.x;
      islandFbSize.y
        = (fullFbSize.y / numIslands)
        + (islandIdx < (fullFbSize.y % numIslands));
    
      // ... and resize the local accum buffer
      // if (localAccumBuffer)
      //   CUDA_CALL(FreeMPI(localAccumBuffer));
      // CUDA_CALL(MallocMPI(&localAccumBuffer,1+area(islandFbSize)*sizeof(*localAccumBuffer)));
      // CUDA_CALL(Memset(localAccumBuffer,0,area(islandFbSize)*sizeof(*localAccumBuffer)));
      localFB.resize(islandFbSize.x*islandFbSize.y);
    
      // ------------------------------------------------------------------
      // compute mem required for compositing, and allocate
      // ------------------------------------------------------------------
      const int ourCompLineBegin
        = (islandFbSize.y * (islandRank+0)) / islandSize;
      const int ourCompLineEnd
        = (islandFbSize.y * (islandRank+1)) / islandSize;

      // ------------------------------------------------------------------
      // how many lines we'll *produce* during compositing, and mem for
      // it
      // ------------------------------------------------------------------
      const int ourCompLineCount = ourCompLineEnd-ourCompLineBegin;
      compResultMemory.resize(ourCompLineCount*islandFbSize.x);
      // if (compResultMemory)
      //   CUDA_CALL(FreeMPI(compResultMemory));
      // CUDA_CALL(MallocMPI(&compResultMemory,
      //                     1+ourCompLineCount*islandFbSize.x*sizeof(uint32_t)));
    
      // ------------------------------------------------------------------
      // how many lines we'll *receive* for compositing, and mem for it
      // ------------------------------------------------------------------
      const int numCompInputs = ourCompLineCount * islandSize;
      compInputsMemory.resize(numCompInputs*islandFbSize.x);
      // if (compInputsMemory)
      //   CUDA_CALL(FreeMPI(compInputsMemory));
      // CUDA_CALL(MallocMPI(&compInputsMemory,
      //                     1+numCompInputs*islandFbSize.x*sizeof(*compInputsMemory)));
    }
  }
  
  void AddWorkersRenderer::render(uint32_t *fbPointer)
  {
    Renderer::render(fbPointer);
    if (isMaster()) {
      /* nothing to do on master yet, wait for workers to render... */
      assert(fbPointer);
      assert(fbSize.x > 0);
      assert(fbSize.y > 0);
      comm->master.toWorkers->indexedGather
        (fbPointer,
         fbSize.x*sizeof(uint32_t),
         fbSize.y);
    } else {
      // ------------------------------------------------------------------
      // step 0: clients render into their owl local frame buffers
      // ------------------------------------------------------------------
      renderLocal();

      const int numIslands = comm->worker.numIslands;
      const int islandIdx  = comm->worker.islandIdx;
      const int islandRank = comm->worker.withinIsland->rank;
      const int islandSize = comm->worker.withinIsland->size;
      
      const int ourLineBegin
        = (islandFbSize.y * (islandRank+0)) / islandSize;
      const int ourLineEnd
        = (islandFbSize.y * (islandRank+1)) / islandSize;
      const int ourLineCount = ourLineEnd-ourLineBegin;
      // ------------------------------------------------------------------
      // step 1: exchage accum buffer regions w/ island peers
      // ------------------------------------------------------------------
      std::vector<int> sendCounts(islandSize);
      std::vector<int> sendOffsets(islandSize);
      std::vector<int> recvCounts(islandSize);
      std::vector<int> recvOffsets(islandSize);
      for (int i=0;i<islandSize;i++) {
        const size_t sizeOfLine = islandFbSize.x*sizeof(*localFB);
        
        // in:
        recvCounts[i] = ourLineCount*sizeOfLine;
        recvOffsets[i] = i*ourLineCount*sizeOfLine;
        
        // out:
        const int hisLineBegin
          = (islandFbSize.y * (i+0)) / islandSize;
        const int hisLineEnd
          = (islandFbSize.y * (i+1)) / islandSize;
        const int hisLineCount = hisLineEnd-hisLineBegin;
        sendCounts[i]  = hisLineCount*sizeOfLine;
        sendOffsets[i] = hisLineBegin*sizeOfLine;
      }

      comm->worker.withinIsland->allToAll
        (localFB.get(), //const void *sendBuf,
         sendCounts.data(),//const int *sendCounts,
         sendOffsets.data(),//const int *sendOffsets,
         compInputsMemory.get(),//void *recvBuf,
         recvCounts.data(),//const int *recvCounts,
         recvOffsets.data()//const int *recvOffsets) 
         );

      // ------------------------------------------------------------------
      // step 2: compose locally (optix backend uses cuda)
      // ------------------------------------------------------------------
      const vec2i ourRegionSize(islandFbSize.x,ourLineCount);
      // composeRegion(ourRegionSize,
      //               compInputsMemory.get(),// ,ourRegionSize
      //               // islandSize,
      //               // accumID,
      //               compResultMemory.get());
      composeRegion(compResultMemory.get(),
                    ourRegionSize,
                    compInputsMemory.get(),// ,ourRegionSize
                    // islandSize,
                    // accumID,
                    islandSize
                    );
      CUDA_SYNC_CHECK();
    
      // ------------------------------------------------------------------
      // step 3: send to master ...
      // ------------------------------------------------------------------
      std::vector<const void *> blockPointers(ourRegionSize.y);
      std::vector<int> blockTags(ourRegionSize.y);
      for (int iy=0;iy<ourRegionSize.y;iy++) {
        const size_t sizeOfLine = ourRegionSize.x*sizeof(int);
        int island_y = ourLineBegin+iy;
        int global_y = islandIdx + island_y * numIslands;
        blockTags[iy] = global_y;
        blockPointers[iy] = ((char *)compResultMemory.get())+iy*sizeOfLine;
      }
      comm->worker.toMaster->indexedGatherSend
        (ourRegionSize.y,
         ourRegionSize.x*sizeof(uint32_t),
         blockTags.data(),
         blockPointers.data());
    }
  }

}
