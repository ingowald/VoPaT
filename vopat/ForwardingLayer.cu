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

#include "vopat/ForwardingLayer.h"
#include "owl/owl_device.h"
#include <sstream>

namespace vopat {

  __global__ void computeCompactionOffsets(int rank,
                                           ForwardingLayer::DD globals)
  {
    if (threadIdx.x != 0) return;
    int ofs = 0;
    for (int i=0;i<globals.islandSize;i++) {
      globals.perRankSendOffsets[i] = ofs;
      // printf("(%i) compact offset %i = %i\n",rank,i,ofs);
      ofs += globals.perRankSendCounts[i];
    }
  }
  
  __global__ void compactRays(ForwardingLayer::DD globals, int numRays)
  {
    int tid = threadIdx.x+blockIdx.x*blockDim.x;
    if (tid >= numRays) return;

    int dest = globals.rayNextRank[tid];
    // if (globals.rayQueueIn[tid].dbg_destRank != dest)
    //   printf("bad dest in out queue\n");
    // if (dest < 0) return;

    int slot = atomicAdd(&globals.perRankSendOffsets[dest],1);
    auto ray = globals.rayQueueIn[tid];
    globals.rayQueueOut[slot] = ray;
  }


  void ForwardingLayer::createSendQueue()
  {
    // std::cout << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << std::endl;
    int numRaysOut;
    CUDA_CALL(Memcpy(&numRaysOut,dd.pNumRaysOut,sizeof(int),cudaMemcpyDefault));
    // printf("[%i] num out %i\n",myRank(),numRaysOut);
    computeCompactionOffsets<<<1,1>>>
      (myRank(),dd);
    // CUDA_SYNC_CHECK();
    int blockSize = 256;
    int numBlocks = divRoundUp(numRaysOut,blockSize);
    std::swap(rayQueueOut,rayQueueIn);
    std::swap(dd.rayQueueOut,dd.rayQueueIn);

    // CUDA_CALL(Memset(dd.rayQueueOut,-1,sizeof(Ray)*numRaysOut));
    if (numBlocks)
      compactRays<<<numBlocks,blockSize>>>
        (dd,numRaysOut);
    // CUDA_SYNC_CHECK();
  }

  // __global__
  // void checkRaysReceived(int *d_flag, Ray *rayQueueIn, int numRays, int rank)
  // {
  //   int tid = threadIdx.x+blockIdx.x*blockDim.x;
  //   if (tid >= numRays) return;

  //   if (rayQueueIn[tid].dbg_destRank != rank) {
  //     if (atomicAdd(d_flag,1) == 0)
  //       printf("NOT the right rank where this was supposed to go - on %i ray wanted %i!?\n",
  //              rank,rayQueueIn[tid].dbg_destRank);
  //   }
  // }

  // __global__
  // void checkRaysOut(int *d_flag, Ray *rayQueueOut, int ofs, int numRays, int dest, int rank)
  // {
  //   int tid = threadIdx.x+blockIdx.x*blockDim.x;
  //   if (tid >= numRays) return;

  //   if (rayQueueOut[ofs+tid].dbg_destRank != dest)
  //     if (atomicAdd(d_flag,1) == 0)
  //       printf("(%i) checkout - ofs %i: ray says %i list pos says %i\n",
  //              rank,
  //              ofs+tid,rayQueueOut[ofs+tid].dbg_destRank,dest
  //              );
  // }

  int  ForwardingLayer::exchangeRays()
  {
    createSendQueue();
    
    host_sendCounts = perRankSendCounts.download();

    // const int numWorkers = dd.islandSize;//dd.numWorkers;
    // const int myRank = dd.islandRank;//this->myRank();
    auto island    = comm->worker.withinIsland;
    int myRank     = comm->islandRank();
    int numWorkers = comm->islandSize();

    // for (int printRank=0;printRank<numWorkers;printRank++) {
    //   island->barrier();
    //   fflush(0);
    //   if (myRank != printRank) continue;
      
    //   for (int i=0;i<numWorkers;i++)
    //     printf("[%i] to rank %i = %i\n",myRank,i,host_sendCounts[i]);
    // } 
    
    /* iw - TODO: change this to use alltoall instead of allgaher;
       with allgather each rank received N*N elements, which describes
       what every node is sending to any other node ... but it would
       actually only need to know how many _it_ receives */
    std::vector<int> allRankSendCounts(numWorkers*numWorkers);
    island->allGather(allRankSendCounts,host_sendCounts);

    // for (int printRank=0;printRank<numWorkers;printRank++) {
    //   island->barrier();
    //   fflush(0);
    //   if (myRank != printRank) continue;
      
    //   for (int i=0;i<numWorkers*numWorkers;i++)
    //     printf("[%i] allSendCounts[%i] = %i\n",myRank,i,allRankSendCounts[i]);
    // }
    
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

    // for (int i=0;i<numWorkers;i++) {
    //   int numRays = sendByteCounts[i] / sizeof(Ray);
    //   if (numRays == 0) continue;
    //   int bs = 128;
    //   int nb = divRoundUp(numRays,bs);
    //   int *result;
    //   cudaMallocManaged(&result,sizeof(int));
    //   *result = 0;
    //   checkRaysOut<<<nb,bs>>>(result,rayQueueOut.get(),
    //                           sendByteOffsets[i]/sizeof(Ray),
    //                           numRays,i,myRank);
    //   CUDA_SYNC_CHECK();
    //   if (*result)
    //     throw std::runtime_error("invalid compaction queue");
    // }
    
    
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

    // for (int printRank=0;printRank<numWorkers;printRank++) {
    //   island->barrier();
    //   fflush(0);
    //   if (myRank != printRank) continue;
      
    //   for (int i=0;i<numWorkers;i++)
    //     printf("[%2i] from %2i = %9i/%9i ofs %9i    to %2i = %9i/%9i ofs %9i\n",
    //            myRank,
    //            i,
    //            recvByteCounts[i],
    //            recvByteCounts[i]/(int)sizeof(Ray), 
    //            recvByteOffsets[i]/(int)sizeof(Ray),
    //            i,
    //            sendByteCounts[i],
    //            sendByteCounts[i]/(int)sizeof(Ray),
    //            sendByteOffsets[i]/(int)sizeof(Ray));
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
    numRaysIn = numReceived;
    dd.numRaysIn = numRaysIn;

    // if (numRaysIn) {
    //   int bs = 128;
    //   int nb = divRoundUp(numRaysIn,bs);
    //   int *result;
    //   cudaMallocManaged(&result,sizeof(int));
    //   *result = 0;
    //   checkRaysReceived<<<nb,bs>>>(result,rayQueueIn.get(),numRaysIn,myRank);
    //   CUDA_SYNC_CHECK();
    //   if (*result) {
    //     printf("WRONG result!\n");
    //     sleep(1);
    //     exit(1);
    //   }
    // }

    
    // ------------------------------------------------------------------
    // return how many we've exchanged ACROSS ALL ranks
    // ------------------------------------------------------------------
    int sumAllSends = 0;
    for (auto i : allRankSendCounts) sumAllSends += i;

    return sumAllSends;
  }

// #if 0
//   void ForwardingLayer::renderLocal()
//   {
//     static Prof prof_renderLocal("renderLocal",comm->myRank());
//     static Prof prof_genPrimary("genPrimary",comm->myRank());
//     static Prof prof_traceLocally("traceLocally",comm->myRank());
//     static Prof prof_exchangeRays("exchangeRays",comm->myRank());
//     prof_renderLocal.enter();

//     vec2i blockSize(16);
//     vec2i numBlocks = divRoundUp(islandFbSize,blockSize);
//     int sumRaysExchanged = 0;
    
//     for (int s = 0; s < numSPP; s++) {
//       dd.sampleID = numSPP * accumID + s;
    
//       perRankSendCounts.bzero();
//       allSendCounts.bzero();
//       // CUDA_SYNC_CHECK();
//       if (numBlocks != vec2i(0)) {
//         prof_genPrimary.enter();
//         nodeRenderer->generatePrimaryWave(dd);
//         prof_genPrimary.leave();
//       }
//       CUDA_SYNC_CHECK();
//       host_sendCounts = perRankSendCounts.download();
//       // {
//       //   std::stringstream out;
//       //   out << "(" << comm->myRank() << ") snd-pri ";
//       //   for (auto i : host_sendCounts)
//       //     out << " " << i;
//       //   out << std::endl;
//       //   std::cout << out.str();
//       // }
//       numRaysInQueue = host_sendCounts[myRank()];
//       dd.numRaysInQueue = numRaysInQueue;

// // #if 1
// //       checkRays(comm->myRank(),dd.rayQueueIn,dd.numRaysInQueue,"after primary");
// // #endif
      
// //       CUDA_SYNC_CHECK();
      
//       int numIterations = 0;
//       while (true) {
//         if (++numIterations > 100)
//           printf("loooots of iterations...\n");
//         perRankSendCounts.bzero();
//         allSendCounts.bzero();
//         CUDA_SYNC_CHECK();
        
//         prof_traceLocally.enter();
//         traceRaysLocally();
//         CUDA_SYNC_CHECK();
//         prof_traceLocally.leave();
        
//         if (Prof::is_active) {
//           comm->worker.withinIsland->barrier();
//         }
        
//         createSendQueue();
//         CUDA_SYNC_CHECK();

//         prof_exchangeRays.enter();
//         int numRaysExchanged = exchangeRays();
//         CUDA_SYNC_CHECK();
//         sumRaysExchanged += numRaysExchanged;
//         prof_exchangeRays.leave();

//         if (numRaysExchanged == 0)
//           break;
        
//       }
//     }

//     CUDA_SYNC_CHECK();
//     static Prof prof_addLocalFB("addLocalFB",comm->myRank());
//     prof_addLocalFB.enter();
//     if (numBlocks != vec2i(0)) {
//       writeLocalFB<<<numBlocks,blockSize>>>(islandFbSize,
//                                             localFB.get(),
//                                             accumBuffer.get(),
//                                             (dd.sampleID+1));
//     }
//     CUDA_SYNC_CHECK();
//     prof_addLocalFB.leave();
//     prof_renderLocal.leave();

//     static int nextPing = 1;
//     static int curPing = 0;
//     curPing++;
//     while (curPing >= nextPing) {
//       std::cout << "(" << comm->myRank() << ") frame done; num rays exchanged is " << prettyNumber(sumRaysExchanged) << std::endl;
//       nextPing *= 2;
//      fflush(0);
//     }
//   }
// #endif
  
  // void ForwardingLayer::traceRaysLocally()
  // {
  //   // const int myRank = dd.islandRank;//this->myRank();
  //   // printf("(%i) tracerays in\n",myRank);fflush(0);
  //   CUDA_SYNC_CHECK();
  //   nodeRenderer->traceLocally(dd);
  //   CUDA_SYNC_CHECK();
  //   // printf("(%i) tracerays OUT\n",myRank);fflush(0);
  // }
 
  ForwardingLayer::ForwardingLayer(CommBackend *comm)
    : comm(comm)
  {
    if (isMaster()) {
    } else {
      dd.islandSize  = comm->islandSize();
      allSendCounts.resize(1);
      perRankSendCounts.resize(dd.islandSize);
      perRankSendOffsets.resize(dd.islandSize);
      dd.perRankSendOffsets = perRankSendOffsets.get();
      dd.perRankSendCounts = perRankSendCounts.get();
      dd.pNumRaysOut = allSendCounts.get();
    }
  }

  void ForwardingLayer::clearQueue()
  {
    if (isMaster()) {
    } else {
      perRankSendCounts.bzero();
      allSendCounts.bzero();
    }
  }

  void ForwardingLayer::resizeQueues(int maxRaysPerQueue)
  {
    if (isMaster()) {
    } else {
      rayQueueIn.resize(maxRaysPerQueue);
      dd.rayQueueIn   = rayQueueIn.get();

      rayQueueOut.resize(maxRaysPerQueue);
      dd.rayQueueOut  = rayQueueOut.get();

      rayNextRank.resize(maxRaysPerQueue);
      dd.rayNextRank = rayNextRank.get();
    }
  }
   
}
