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

#include "vopat/render/RayForwardingRenderer.h"
#include "3rdParty/stb_image//stb/stb_image_write.h"
#include "3rdParty/stb_image//stb/stb_image.h"
#include "owl/owl_device.h"
#include <sstream>

namespace vopat {

  __global__ void writeLocalFB(vec2i fbSize,
                               small_vec3f *localFB,
                               vec3f *accumBuffer,
                               int numAccumFrames)
  {
    int ix = threadIdx.x + blockIdx.x*blockDim.x;
    int iy = threadIdx.y + blockIdx.y*blockDim.y;
    if (ix >= fbSize.x) return;
    if (iy >= fbSize.y) return;

    int i = ix + iy * fbSize.x;
    
    localFB[i] = to_half(accumBuffer[i] * (1.f/(numAccumFrames)));
  }
    
  void RayForwardingRenderer::createSendQueue(bool fishy)
  {
    computeCompactionOffsets<<<1,1>>>
      (globals,fishy);
    CUDA_SYNC_CHECK();
    int blockSize = 256;
    int numBlocks = divRoundUp(numRaysInQueue,blockSize);
    if (numBlocks)
      compactRays<<<numBlocks,blockSize>>>
        (globals,numRaysInQueue);
    CUDA_SYNC_CHECK();
  }

  int  RayForwardingRenderer::exchangeRays()
  {
    host_sendCounts = perRankSendCounts.download();
    const int numWorkers = globals.islandSize;//globals.numWorkers;
    const int myRank = globals.islandRank;//this->myRank();
    auto island = comm->worker.withinIsland;


// #if 1
//     int mySendCounts = 0;
//     for (auto i : host_sendCounts) mySendCounts += i;
//     checkRays(comm->myRank(),globals.rayQueueOut,mySendCounts,"sendqueue");
// #endif
      
        

    
    /* iw - TODO: change this to use alltoall instead of allgaher;
       with allgather each rank received N*N elements, which describes
       what every node is sending to any other node ... but it would
       actually only need to know how many _it_ receives */
    std::vector<int> allRankSendCounts(numWorkers*numWorkers);
    island->allGather(allRankSendCounts,host_sendCounts);

    // ------------------------------------------------------------------
    // compute SEND counts and offsets
    // ------------------------------------------------------------------
    std::vector<int> sendByteOffsets(numWorkers);
    std::vector<int> sendByteCounts(numWorkers);
    size_t ofs = 0;
    for (int i=0;i<numWorkers;i++) {
      int to_i = allRankSendCounts[myRank*numWorkers+i];
      sendByteOffsets[i] = ofs;
      sendByteCounts[i]  = to_i*sizeof(Ray);
      ofs += sendByteCounts[i];
    }

// #if 1
//     {
//       std::stringstream out;
//       out << "(" << myRank << ") self-send " << sendByteCounts[myRank] << std::endl;
//       std::cout << out.str();
//     }
// #endif


    // ------------------------------------------------------------------
    // compute RECEIVE counts and offsets
    // ------------------------------------------------------------------
    std::vector<int> recvByteOffsets(numWorkers);
    std::vector<int> recvByteCounts(numWorkers);
    ofs = 0;
    int numReceived = 0;
    for (int i=0;i<numWorkers;i++) {
      int from_i = allRankSendCounts[i*numWorkers+myRank];
      recvByteOffsets[i] = ofs;
      recvByteCounts[i]  = from_i*sizeof(Ray);
      ofs += recvByteCounts[i];
      numReceived += from_i;
    }

// #if 1
//     {
//       std::stringstream out;
//       out << "(" << myRank << ") self-recv " << recvByteCounts[myRank] << std::endl;
//       std::cout << out.str();
//     }
// #endif


    // {
    //   std::stringstream out;
    //   out << "[" << comm->myRank() << "] to:";
    //   for (int i=0;i<numWorkers;i++)
    //     out << " " << allRankSendCounts[myRank*numWorkers+i];
    //   out << " frm:";
    //   for (int i=0;i<numWorkers;i++)
    //     out << " " << allRankSendCounts[i*numWorkers+myRank];
    //   out << " -> numrecv " << numReceived << std::endl;
    //   std::cout << out.str();
    // }
    
    // ------------------------------------------------------------------
    // exeute the all2all
    // ------------------------------------------------------------------
    island->allToAll(rayQueueOut.get(),
                     sendByteCounts.data(),
                     sendByteOffsets.data(),
                     rayQueueIn.get(),
                     recvByteCounts.data(),
                     recvByteOffsets.data());
    CUDA_SYNC_CHECK();
    numRaysInQueue = numReceived;
    globals.numRaysInQueue = numRaysInQueue;


// #if 1
//     checkRays(comm->myRank(),globals.rayQueueIn,numRaysInQueue,"recved");
// #endif
    
    // ------------------------------------------------------------------
    // return how many we've exchanged ACROSS ALL ranks
    // ------------------------------------------------------------------
    int sumAllSends = 0;
    for (auto i : allRankSendCounts) sumAllSends += i;

    // for (int r=0;r<numWorkers;r++) {
    //   comm->worker.withinIsland->barrier();
    //   if (r == globals.myRank) {
    //     std::cout << "(" << r << ") IN:  ";
    //     for (int i=0;i<numWorkers;i++)
    //       std::cout << (recvByteCounts[i]/sizeof(RayType)) << " ";
    //     std::cout << std::endl;
    //     std::cout << "(" << r << ") OUT: ";
    //     for (int i=0;i<numWorkers;i++)
    //       std::cout << (sendByteCounts[i]/sizeof(RayType)) << " ";
    //     std::cout << std::endl;
    //     std::cout << "(" << r << ") num in queue " << numRaysInQueue << std::endl;
    //     if (r == 0)
    //       std::cout << "-------------------------------------------------------" << std::endl;
    //     fflush(0);
    //   }
    //   comm->worker.withinIsland->barrier();
    // }

    return sumAllSends;
  }
  
  void RayForwardingRenderer::renderLocal()
  {
    static Prof prof_renderLocal("renderLocal",comm->myRank());
    static Prof prof_genPrimary("genPrimary",comm->myRank());
    static Prof prof_traceLocally("traceLocally",comm->myRank());
    static Prof prof_exchangeRays("exchangeRays",comm->myRank());
    prof_renderLocal.enter();
    
    vec2i blockSize(16);
    vec2i numBlocks = divRoundUp(islandFbSize,blockSize);
    int sumRaysExchanged = 0;
    
    for (int s = 0; s < numSPP; s++) {
      globals.sampleID = numSPP * accumID + s;
    
      perRankSendCounts.bzero();
      // CUDA_SYNC_CHECK();
      if (numBlocks != vec2i(0)) {
        prof_genPrimary.enter();
        nodeRenderer->generatePrimaryWave(globals);
        prof_genPrimary.leave();
      }
      CUDA_SYNC_CHECK();
      host_sendCounts = perRankSendCounts.download();
      // {
      //   std::stringstream out;
      //   out << "(" << comm->myRank() << ") snd-pri ";
      //   for (auto i : host_sendCounts)
      //     out << " " << i;
      //   out << std::endl;
      //   std::cout << out.str();
      // }
      numRaysInQueue = host_sendCounts[myRank()];
      globals.numRaysInQueue = numRaysInQueue;

// #if 1
//       checkRays(comm->myRank(),globals.rayQueueIn,globals.numRaysInQueue,"after primary");
// #endif
      
//       CUDA_SYNC_CHECK();
      
      int numIterations = 0;
      bool fishy = false;
      while (true) {
        globals.fishy = fishy;
        if (++numIterations > 100)
          printf("loooots of iterations...\n");
        perRankSendCounts.bzero();
        CUDA_SYNC_CHECK();
        
        prof_traceLocally.enter();
        traceRaysLocally(fishy);
        CUDA_SYNC_CHECK();
        prof_traceLocally.leave();
        
        if (Prof::is_active) {
          comm->worker.withinIsland->barrier();
        }
        
        createSendQueue(fishy);
        CUDA_SYNC_CHECK();

        prof_exchangeRays.enter();
        int numRaysExchanged = exchangeRays();
        CUDA_SYNC_CHECK();
        sumRaysExchanged += numRaysExchanged;
        prof_exchangeRays.leave();

        if (numIterations >= 100 && numRaysExchanged == 1)
          fishy = true;
        
        if (numRaysExchanged == 0)
          break;
        
      }
    }

    CUDA_SYNC_CHECK();
    static Prof prof_addLocalFB("addLocalFB",comm->myRank());
    prof_addLocalFB.enter();
    if (numBlocks != vec2i(0)) {
      writeLocalFB<<<numBlocks,blockSize>>>(islandFbSize,
                                            localFB.get(),
                                            accumBuffer.get(),
                                            (globals.sampleID+1));
    }
    CUDA_SYNC_CHECK();
    prof_addLocalFB.leave();
    prof_renderLocal.leave();

    static int nextPing = 1;
    static int curPing = 0;
    curPing++;
    while (curPing >= nextPing) {
      std::cout << "(" << comm->myRank() << ") frame done; num rays exchanged is " << prettyNumber(sumRaysExchanged) << std::endl;
      nextPing *= 2;
     fflush(0);
    }
  }
  
  void RayForwardingRenderer::screenShot()
  {
    std::string fileName = Renderer::screenShotFileName;
    std::vector<uint32_t> pixels;
    vec2i fbSize;
    if (isMaster()) {
      fileName = fileName + "_master.png";
      pixels = masterFB.download();
      for (int iy=0;iy<worldFbSize.y/2;iy++) {
        uint32_t *top = pixels.data() + iy * worldFbSize.x;
        uint32_t *bot = pixels.data() + (worldFbSize.y-1-iy) * worldFbSize.x;
        for (int ix=0;ix<worldFbSize.x;ix++)
          std::swap(top[ix],bot[ix]);
      }
      fbSize = worldFbSize;
    } else {
      char suff[100];
      sprintf(suff,"_island%03i_rank%05i.png",
              comm->worker.islandIdx,comm->worker.withinIsland->rank);
      fileName = fileName + suff;
      
      std::vector<small_vec3f> hostFB;
      hostFB = localFB.download();
      for (int y=0;y<islandFbSize.y;y++) {
        const small_vec3f *line = hostFB.data() + (islandFbSize.y-1-y)*islandFbSize.x;
        for (int x=0;x<islandFbSize.x;x++) {
          vec3f col = from_half(line[x]);
          pixels.push_back(make_rgba(col) | (0xffu << 24));
        }
      }
      fbSize = islandFbSize;
    }
    
    stbi_write_png(fileName.c_str(),fbSize.x,fbSize.y,4,
                   pixels.data(),fbSize.x*sizeof(uint32_t));
    std::cout << "screenshot saved in '" << fileName << "'" << std::endl;

  }

  void RayForwardingRenderer::traceRaysLocally(bool fishy)
  {
    // const int myRank = globals.islandRank;//this->myRank();
    // printf("(%i) tracerays in\n",myRank);fflush(0);
    CUDA_SYNC_CHECK();
    nodeRenderer->traceLocally(globals,fishy);
    CUDA_SYNC_CHECK();
    // printf("(%i) tracerays OUT\n",myRank);fflush(0);
  }
 
  RayForwardingRenderer::RayForwardingRenderer(CommBackend *comm,
                                               NodeRenderer *nodeRenderer,
                                               int numSPP)
    : AddWorkersRenderer(comm),
      nodeRenderer(nodeRenderer),
      numSPP(numSPP)
  {
    if (isMaster()) {
    } else {
      globals.islandRank  = comm->islandRank();
      globals.islandSize  = comm->islandSize();
      globals.islandIndex = comm->islandIndex();
      globals.islandCount = comm->islandCount();
            
      perRankSendCounts.resize(globals.islandSize);
      perRankSendOffsets.resize(globals.islandSize);
      globals.perRankSendOffsets = perRankSendOffsets.get();
      globals.perRankSendCounts = perRankSendCounts.get();
    }
  }


  void RayForwardingRenderer::resizeFrameBuffer(const vec2i &newSize)
  {
    AddWorkersRenderer::resizeFrameBuffer(newSize);
    if (isMaster()) {
    } else {
      // PRINT(newSize);
      accumBuffer.resize(islandFbSize.x*islandFbSize.y);
      globals.accumBuffer  = accumBuffer.get();
      
      globals.islandFbSize = islandFbSize;
      globals.worldFbSize  = worldFbSize;
      
      rayQueueIn.resize(islandFbSize.x*islandFbSize.y);
      globals.rayQueueIn   = rayQueueIn.get();

      rayQueueOut.resize(islandFbSize.x*islandFbSize.y);
      globals.rayQueueOut  = rayQueueOut.get();

      rayNextNode.resize(islandFbSize.x*islandFbSize.y);
      globals.rayNextNode = rayNextNode.get();
    }
  }
   
}
