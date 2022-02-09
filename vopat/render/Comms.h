// ======================================================================== //
// Copyright 2022-2022 Ingo Wald                                            //
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

namespace vopat {
  
  struct ToMasterComm {
    // void allGatherSend(const void *data, size_t size) = 0;
    virtual void indexedGatherSend(int numBlocks,
                                   size_t blockSize,
                                   const int *blockTags,
                                   const void **blockPtrs) = 0;
  };
  
  struct ToWorkersComm {
    virtual void broadcast(const void *data, size_t size) = 0;
    // void allGather(const std::vector<const void *> &destinationPointers);
    virtual void indexedGather(void *recvBuffer,
                               const size_t blockSize,
                               const size_t numBlocks) = 0;
  };

  struct IntraIslandComm {
    template<typename T>
    void allGather(std::vector<T> &result,
                   const T &ours)
    {
      allGather(result.data(),&ours,sizeof(T)); 
    }
    virtual void barrier() = 0;
    virtual void allGather(void *destArray,
                           const void *ours,
                           size_t elementSize) = 0;
    
    virtual void allToAll(const void *sendBuf,
                          const int *sendCounts,
                          const int *sendOffsets,
                          void *recvBuf,
                          const int *recvCounts,
                          const int *recvOffsets) = 0;
    int rank;
    int size;
  };


  struct CommBackend {
    
    virtual void barrierAll() = 0;

    int worldRank;
    int worldSize;
    int workersRank;
    int workersSize;
    bool isMaster, isWorker;
    int numRanksPerIsland;
    struct {
      ToMasterComm *toMaster = nullptr;
      IntraIslandComm *withinIsland = nullptr;
      int numIslands;
      int islandIdx;
      int gpuID;
      std::string gpuName;
    } worker;
    struct {
      ToWorkersComm *toWorkers = nullptr;
    } master;
    
  };
  
}

