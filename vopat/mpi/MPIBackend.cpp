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

#include "MPIBackend.h"
#include "cuda_runtime_api.h"

#define MPI_CALL(a) MPI_##a;

namespace vopat {

  typedef enum { MPI_SPLIT_KEY_MASTER, MPI_SPLIT_KEY_ISLANDS } MPISplitKeyConstants;
    
  MPIBackend::MPIBackend(int &ac, char **&av, int numRanksPerIsland)
  {
    MPI_CALL(Init(&ac,&av));
    this->numRanksPerIsland = numRanksPerIsland;
    // ------------------------------------------------------------------
    // setup global comm among all ranks - we'll need this to exchange
    // further info on who does what
    // ------------------------------------------------------------------

    MPI_CALL(Comm_dup(MPI_COMM_WORLD,&worldComm));
    MPI_CALL(Comm_rank(worldComm,&worldRank));
    MPI_CALL(Comm_size(worldComm,&worldSize));
    isMaster = (worldRank == 0);
    isWorker = !isMaster;
    
    // ------------------------------------------------------------------
    // construct intercomm to master
    // ------------------------------------------------------------------

    MPIToMasterComm *toMaster = nullptr;
    MPIToWorkersComm *toWorkers = nullptr;
    MPIIntraIslandComm *intraIsland = nullptr;
    
    if (isMaster) {
      toWorkers = new MPIToWorkersComm;
      master.toWorkers = toWorkers;
      
      MPI_CALL(Comm_split(worldComm,0,MPI_SPLIT_KEY_MASTER,&workersComm));
      MPI_CALL(Intercomm_create(workersComm,0,worldComm,1,MPI_SPLIT_KEY_MASTER,
                                &toWorkers->comm));
      workersRank = -1;
      MPI_CALL(Comm_remote_size(toWorkers->comm,&workersSize));
      PRINT(workersSize);
    } else {
      toMaster = new MPIToMasterComm;
      worker.toMaster = toMaster;
      intraIsland = new MPIIntraIslandComm;
      worker.withinIsland = intraIsland;
      
      MPI_CALL(Comm_split(worldComm,1,MPI_SPLIT_KEY_MASTER,&workersComm));
      MPI_CALL(Intercomm_create(workersComm,0,worldComm,0,MPI_SPLIT_KEY_MASTER,
                                &toMaster->comm));
      MPI_CALL(Comm_rank(workersComm,&workersRank));
      MPI_CALL(Comm_size(workersComm,&workersSize));
    }

    if (isWorker) {
      // ------------------------------------------------------------------
      // determine which (world) rank lived on which host, and assign
      // GPUSs
      // ------------------------------------------------------------------
      
      std::vector<char> sendBuf(MPI_MAX_PROCESSOR_NAME);
      std::vector<char> recvBuf(MPI_MAX_PROCESSOR_NAME*workersSize);
      bzero(sendBuf.data(),sendBuf.size());
      bzero(recvBuf.data(),recvBuf.size());
      int hostNameLen;
      MPI_CALL(Get_processor_name(sendBuf.data(),&hostNameLen));
      this->hostName = sendBuf.data();
      MPI_CALL(Allgather(sendBuf.data(),sendBuf.size(),MPI_CHAR,
                         recvBuf.data(),/*yes, SENDbuf here*/sendBuf.size(),
                         MPI_CHAR,workersComm));
      std::vector<std::string> hostNames;
      for (int i=0;i<workersSize;i++) 
        hostNames.push_back(recvBuf.data()+i*MPI_MAX_PROCESSOR_NAME);
      hostName = sendBuf.data();
      
      // ------------------------------------------------------------------
      // count how many other ranks are already on this same node
      // ------------------------------------------------------------------
      MPI_CALL(Barrier(workersComm));
      int localDeviceID = 0;
      for (int i=0;i<workersRank;i++) 
        if (hostNames[i] == hostName)
          localDeviceID++;
      MPI_CALL(Barrier(workersComm));
      
      // ------------------------------------------------------------------
      // assign a GPU to this rank
      // ------------------------------------------------------------------
      int numGPUsOnThisNode;
      CUDA_CALL(GetDeviceCount(&numGPUsOnThisNode));
      if (numGPUsOnThisNode == 0)
        throw std::runtime_error("no GPU on this rank!");
      
      if (localDeviceID >= numGPUsOnThisNode) {
        printf("%s*********** WARNING: oversubscribing GPU on node %s ***********%s\n",
               OWL_TERMINAL_RED,
               hostName.c_str(),
               OWL_TERMINAL_DEFAULT);
      }
      worker.gpuID = localDeviceID % numGPUsOnThisNode;
      MPI_CALL(Barrier(workersComm));
      
      cudaDeviceProp prop;
      cudaGetDeviceProperties(&prop, worker.gpuID);
      worker.gpuName = prop.name;
      
      printf("#vopat.mpi: workers rank #%i on host %s GPU #%i (%s)\n",
             workersRank,hostName.c_str(),worker.gpuID,worker.gpuName.c_str());
      MPI_CALL(Barrier(workersComm));
    }
      
    // ------------------------------------------------------------------
    // assign island index ...
    // ------------------------------------------------------------------
    if (isWorker) {
      worker.numIslands = workersSize / numRanksPerIsland;
      if (worker.numIslands < 1)
        throw std::runtime_error("not enough workers for this model's number of partitions ...");
      if (workersSize % worker.numIslands)
        throw std::runtime_error("workers size not a multiple of num islands!");
      worker.islandIdx = workersRank/numRanksPerIsland;
    
      // ------------------------------------------------------------------
      // and construct island comm
      // ------------------------------------------------------------------
      MPI_CALL(Comm_split(workersComm,worker.islandIdx,0,&intraIsland->comm));
      MPI_CALL(Comm_rank(intraIsland->comm,&intraIsland->rank));
      MPI_CALL(Comm_size(intraIsland->comm,&intraIsland->size));

      printf("#vopat.mpi: workers rank #%i is local rank %i on island %i, local GPU id %i\n",
             workersRank,intraIsland->rank,worker.islandIdx,worker.gpuID);
      MPI_CALL(Barrier(workersComm));
    }
  }

  void MPIBackend::finalize()
  {
    MPI_CALL(Finalize());
  }

  void MPIBackend::barrierAll()
  {
    MPI_CALL(Barrier(worldComm));
  }


  void MPIIntraIslandComm::barrier() 
  {
    MPI_CALL(Barrier(comm));
  }

  void MPIIntraIslandComm::allGather(void *destArray,
                                     const void *ours,
                                     size_t elementSize) 
  {
    // PING;
    // int rank;
    // MPI_Comm_rank(comm,&rank);
    // PRINT(rank);
    // int size;
    // MPI_Comm_size(comm,&size);
    // PRINT(size);
    // PRINT(elementSize);
    // PRINT(destArray);
    // PRINT(ours);
    // PRINT(((int*)ours)[0]);
    // PRINT(((int*)ours)[1]);
    // PRINT(((int*)ours)[2]);
    // PRINT(((int*)ours)[3]);
    MPI_CALL(Allgather(ours,elementSize,MPI_CHAR,
                       destArray,elementSize,MPI_CHAR,
                       comm));
    // PRINT(((int*)destArray)[0]);
    // PRINT(((int*)destArray)[1]);
    // PRINT(((int*)destArray)[2]);
    // PRINT(((int*)destArray)[3]);
  }

  void MPIIntraIslandComm::allToAll(const void *sendBuf,
                                    const int *sendCounts,
                                    const int *sendOffsets,
                                    void *recvBuf,
                                    const int *recvCounts,
                                    const int *recvOffsets)
  {
    // CUDA_SYNC_CHECK();
    MPI_Alltoallv(sendBuf,sendCounts,sendOffsets,MPI_BYTE,
                  recvBuf,recvCounts,recvOffsets,MPI_BYTE,
                  comm);
    // CUDA_SYNC_CHECK();
  }

  void MPIToMasterComm::indexedGatherSend(int numBlocks,
                                          size_t blockSize,
                                          const int *blockTags,
                                          const void **blockPtrs)
  {
    if (numBlocks == 0)
      return;
    
    std::vector<MPI_Request> requests(numBlocks);
    for (int i=0;i<numBlocks;i++) {
      MPI_CALL(Isend(blockPtrs[i],blockSize,MPI_BYTE,0,blockTags[i],
                     comm,&requests[i]));
    }
    MPI_CALL(Waitall(numBlocks,requests.data(),MPI_STATUSES_IGNORE));
  }

  void MPIToWorkersComm::indexedGather(void *recvBuffer,
                                       const size_t blockSize,
                                       const size_t numBlocks)
  {
    std::cout << OWL_TERMINAL_RED;
    PING; PRINT(numBlocks); PRINT(blockSize);
    PRINT(recvBuffer);
    std::cout << OWL_TERMINAL_DEFAULT;
    fflush(0);
    
    if (numBlocks == 0)
      return;

    std::vector<MPI_Request> requests(numBlocks);
    for (int i=0;i<numBlocks;i++) {
      int tag = i;
      MPI_CALL(Irecv(((char*)recvBuffer)+i*blockSize,
                     blockSize,MPI_BYTE,MPI_ANY_SOURCE,tag,
                     comm,&requests[i]));
    }
    PING; fflush(0);
    MPI_CALL(Waitall(numBlocks,requests.data(),MPI_STATUSES_IGNORE));
    PING; fflush(0);
  }

  
  void MPIToWorkersComm::broadcast(const void *data, size_t size)
  {
  }
  

}
