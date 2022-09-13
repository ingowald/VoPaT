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

#include "common/CUDAArray.h"
#include "common/mpi/Comms.h"
#include "vopat/Ray.h"
#include <sstream>

namespace vopat {

  struct ForwardingLayer {
    
    struct DD {
      /*! mark ray at input queue position `rayID` to be forwarded to
          rank with given ID. This will overwrite the ray at input queue pos rayID */
      inline __device__ void forwardRay(const Ray &ray, int destRank) const;
      
      // int          myRank, numWorkers;
      // int          islandRank, islandSize, islandIndex, islandCount;
      // int          sampleID;
      // Camera       camera;

      // vec2i        worldFbSize, islandFbSize;
      // vec3f       *accumBuffer;

      /*! for compaction - where in the output queue to write rays for
          a given target rank */
      int         *perRankSendOffsets;
      
      /*! for compaction - how many rays will go to a given rank */
      int         *perRankSendCounts;

      /*! sum of ALL send counts, so we know where to append ray to the queue */
      int         *pNumRaysOut;
      
      /*! list of node IDs where each of the corresponding rays in
          rayQueueIn are supposed to be sent to (a nextnode of '-1'
          means the ray will die and not get sent anywhere) */
      int         *rayNextRank;

      /*! queue of rays received during ray exchange, and to be
          traced/shaded on this node */
      Ray         *rayQueueIn;
      
      /*! ray queue used for sending rays; generarting by
          sorting/compacting the input queue after ti has been traced
          locally */
      Ray         *rayQueueOut;
      
      /*! number of rays in the input queue */
      int         numRaysIn;
      /*! number of ranks in this island */
      int islandSize;
    };

    ForwardingLayer(CommBackend *comm);


    void resizeQueues(int maxRaysPerQueue);
    
    // void traceRaysLocally();
    void createSendQueue();
    int  exchangeRays();

    inline bool isMaster() const { return comm->isMaster; }
    
    /*! ray queues we are expected to trace in the next step */
    CUDAArray<Ray>         rayQueueIn;
    
    /*! ray queues for sending out; in this one rays are sorted by
        rank they are supposed to go to */
    CUDAArray<Ray>         rayQueueOut;
    
    /*! one int per ray, which - after tracing locally, says which
        rank to go to next; -1 meaning "done" */
    CUDAArray<int>         rayNextRank;
    
    /*! one entry per ray, telling how many rays _we_ want to send to
        that given rank */
    CUDAArray<int>         perRankSendCounts;
    CUDAArray<int>         perRankSendOffsets;
    std::vector<int>       host_sendCounts;
    /*! a single int, just to allow atomically counting where to next
        append to the queue */
    CUDAArray<int>         allSendCounts;
    
    DD dd;
    /*! number of rays currently in the "in" queue */
    int numRaysIn;
    CommBackend *comm;
  };
  

#ifdef __CUDA_ARCH__
  inline __device__
  void ForwardingLayer::DD::forwardRay
  (const Ray &ray, int nextRankForThisRay)
    const
  {
    atomicAdd(&perRankSendCounts[nextRankForThisRay],1);
    int outID = atomicAdd(pNumRaysOut,1);
    rayQueueOut[outID] = ray;
    rayNextRank[outID] = nextRankForThisRay;
  }
#endif
  
}
