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

#include <mpi.h>
#include "common/mpi/Comms.h"

namespace vopat {
  
  struct MPIToMasterComm : public ToMasterComm {
    void indexedGatherSend(int numBlocks,
                           size_t blockSize,
                           const int *blockTags,
                           const void **blockPtrs) override;
    void bc_recv(void *data, size_t size) override;
    
    MPI_Comm comm;
  };
  
  struct MPIToWorkersComm : public ToWorkersComm {
    void broadcast(const void *data, size_t size) override;
    void indexedGather(void *recvBuffer,
                       const size_t blockSize,
                       const size_t numBlocks) override;
    
    MPI_Comm comm;
  };

  struct MPIIntraIslandComm : public IntraIslandComm {
#if VOPAT_USE_RAFI
    MPI_Comm getMPI() override { return comm; }
#endif
    
    void barrier() override;
    void allGather(void *destArray,
                           const void *ours,
                           size_t elementSize) override;
    
    void allToAll(const void *sendBuf,
                  const int *sendCounts,
                  const int *sendOffsets,
                  void *recvBuf,
                  const int *recvCounts,
                  const int *recvOffsets) override;

    MPI_Comm comm;
  };


  /*! implements a "CommsBackend" through MPI; using specified number
      of ranks pre island (by default it creates a single island */
  struct MPIBackend : public CommBackend {

    
    MPIBackend(int &ac, char **&av,
               /*! number of ranks to use per island; implicitly
                   defines the number of islands. For
                   numRanksPerIsland <= 0 we create one island with
                   all ranks but rank 0 */
               int numRanksPerIsland=-1);
    
    void barrierAll() override;
    void finalize() override;

    std::string hostName;
    
    /*! the world comm split into two halves - the master (old rank
      0), and workres (all other ranks */
    MPI_Comm workersComm;
    
    MPI_Comm worldComm;
  };
  
}

