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

#include "vopat/render/DistributedRendererBase.h"
#include "vopat/model/Model.h"
#include "3rdParty/stb_image//stb/stb_image_write.h"
#include "3rdParty/stb_image//stb/stb_image.h"
#include <sstream>

namespace vopat {

  template<typename _RayType>
  struct RayForwardingRenderer : public AddWorkersRenderer {
    using RayType = _RayType;
    
    struct Globals {
      /*! "kill" the ray in input queue position `rayID` - do nothing
          further with this ray, and do not forward it anywhere */
      inline __device__ void killRay(int rayID) const;
      
      /*! (atomically) add the given contribution to the specified pixel */
      inline __device__ void addPixelContribution(int pixelID, vec3f addtl) const;
      
      /*! mark ray at input queue position `rayID` to be forwarded to
          rank with given ID. This will overwrite the ray at input queue pos rayID */
      inline __device__ void forwardRay(int rayID, const RayType &ray, int nextNodeID) const;
      
      int          myRank, numWorkers;
      int          sampleID;
      Camera       camera;
      vec2i        fbSize;
      vec3f       *accumBuffer;

      /*! for compaction - where in the output queue to write rays for
          a given target rank */
      int         *perRankSendOffsets;
      
      /*! for compaction - how many rays will go to a given rank */
      int         *perRankSendCounts;

      /*! list of node IDs where each of the corresponding rays in
          rayQueueIn are supposed to be sent to (a nextnode of '-1'
          means the ray will die and not get sent anywhere) */
      int         *rayNextNode;

      /*! queue of rays received during ray exchange, and to be
          traced/shaded on this node */
      RayType     *rayQueueIn;
      
      /*! ray queue used for sending rays; generarting by
          sorting/compacting the input queue after ti has been traced
          locally */
      RayType     *rayQueueOut;
      
      /*! number of rays in the input queue */
      int          numRaysInQueue;
    };

    /*! abstraction for any sort of renderer that will generate and/or
        shade/modify/bounce rays within this ray forwardign context */
    struct NodeRenderer {
      virtual void generatePrimaryWave(const Globals &globals) = 0;
      virtual void traceLocally(const Globals &globals) = 0;
    };
    
    RayForwardingRenderer(CommBackend *comm,
                          NodeRenderer *nodeRenderer);

    // ==================================================================
    // main things we NEED to implement
    // ==================================================================
    
    /*! asks the _workers_ to render locally; will not get called on
        master, but assumes that all workers render their local frame
        buffer(s) that we'll then merge */
    void renderLocal() override;
    void screenShot() override;

    // ==================================================================
    // things we intercept to know what to do
    // ==================================================================
    
    void resizeFrameBuffer(const vec2i &newSize)  override;
    void setCamera(const vec3f &from,
                   const vec3f &at,
                   const vec3f &up,
                   const float fov) override;

    void resetAccumulation() override
    { AddWorkersRenderer::resetAccumulation(); accumBuffer.bzero(); }
    void traceRaysLocally();
    void createSendQueue();
    int  exchangeRays();
    
    
    /*! ray queues we are expected to trace in the next step */
    CUDAArray<RayType>         rayQueueIn;
    
    /*! ray queues for sending out; in this one rays are sorted by
        rank they are supposed to go to */
    CUDAArray<RayType>         rayQueueOut;
    
    /*! one int per ray, which - after tracing locally, says which
        rank to go to next; -1 meaning "done" */
    CUDAArray<int>         rayNextNode;
    
    /*! one entry per ray, telling how many rays _we_ want to send to
        that given rank */
    CUDAArray<int>         perRankSendCounts;
    CUDAArray<int>         perRankSendOffsets;
    std::vector<int>       host_sendCounts;
    
    CUDAArray<vec3f>       accumBuffer;
    
    Globals globals;
    int numRaysInQueue;
    NodeRenderer *nodeRenderer;
  };
  


  inline __both__ uint32_t make_8bit(const float f)
  {
    return min(255,max(0,int(f*256.f)));
  }

  inline __both__ uint32_t make_rgba(const vec3f color)
  {
    return
      (make_8bit(color.x) << 0) +
      (make_8bit(color.y) << 8) +
      (make_8bit(color.z) << 16) +
      (0xffU << 24);
  }
  inline __both__ uint32_t make_rgba(const vec4f color)
  {
    return
      (make_8bit(color.x) << 0) +
      (make_8bit(color.y) << 8) +
      (make_8bit(color.z) << 16) +
      (make_8bit(color.w) << 24);
  }

  inline __device__ void addToFB(vec3f *tgt, vec3f addtl)
  {
    atomicAdd(&tgt->x,addtl.x);
    atomicAdd(&tgt->y,addtl.y);
    atomicAdd(&tgt->z,addtl.z);
  }

  template<typename NodeRenderer>
  RayForwardingRenderer<NodeRenderer>::RayForwardingRenderer(CommBackend *comm,
                                             NodeRenderer *nodeRenderer)
    : AddWorkersRenderer(comm),
      nodeRenderer(nodeRenderer)
  {
    if (isMaster()) {
    } else {
      const int numWorkers = comm->numWorkers();

      globals.myRank   = myRank();
      globals.numWorkers = numWorkers;
      perRankSendCounts.resize(numWorkers);
      perRankSendOffsets.resize(numWorkers);
      globals.perRankSendOffsets = perRankSendOffsets.get();
      globals.perRankSendCounts = perRankSendCounts.get();
    }
  }


  template<typename RayType>
  void RayForwardingRenderer<RayType>::resizeFrameBuffer(const vec2i &newSize)
  {
    AddWorkersRenderer::resizeFrameBuffer(newSize);
    if (isMaster()) {
    } else {
      accumBuffer.resize(newSize.x*newSize.y);
      globals.accumBuffer = accumBuffer.get();
      globals.fbSize    = newSize;

      rayQueueIn.resize(fbSize.x*fbSize.y);
      globals.rayQueueIn = rayQueueIn.get();

      rayQueueOut.resize(fbSize.x*fbSize.y);
      globals.rayQueueOut = rayQueueOut.get();

      rayNextNode.resize(fbSize.x*fbSize.y);
      globals.rayNextNode = rayNextNode.get();
    }
  }
   
  template<typename T>
  void RayForwardingRenderer<T>::setCamera(const vec3f &from,
                                   const vec3f &at,
                                   const vec3f &up,
                                   const float fovy)
  {
    AddWorkersRenderer::setCamera(from,at,up,fovy);
    globals.camera = AddWorkersRenderer::camera;
  }

  __global__ void writeLocalFB(vec2i fbSize,
                               small_vec3f *localFB,
                               vec3f *accumBuffer,
                               int numAccumFrames);
    
  template<typename T>
  void RayForwardingRenderer<T>::renderLocal()
  {
    vec2i blockSize(16);
    vec2i numBlocks = divRoundUp(islandFbSize,blockSize);
    
    int numSPP = 16;
    for (int s = 0; s < numSPP; s++) {
      globals.sampleID = numSPP * accumID + s;
    
      perRankSendCounts.bzero();
      if (numBlocks != vec2i(0))
        nodeRenderer->generatePrimaryWave(globals);
      // generatePrimaryWave<<<numBlocks,blockSize>>>(globals);
      CUDA_SYNC_CHECK();
      host_sendCounts = perRankSendCounts.download();
      numRaysInQueue = host_sendCounts[myRank()];
      globals.numRaysInQueue = numRaysInQueue;
    
      CUDA_SYNC_CHECK();
      while (true) {
        perRankSendCounts.bzero();
        CUDA_SYNC_CHECK();
        traceRaysLocally();
        CUDA_SYNC_CHECK();
        createSendQueue();
        CUDA_SYNC_CHECK();
        int numRaysExchanged = exchangeRays();
        if (numRaysExchanged == 0)
          break;
      }
    }

    CUDA_SYNC_CHECK();
    if (numBlocks != vec2i(0))
      writeLocalFB<<<numBlocks,blockSize>>>(globals.fbSize,
                                            localFB.get(),
                                            accumBuffer.get(),
                                            (globals.sampleID+1));
    CUDA_SYNC_CHECK();
  }
  
  template<typename T>
  void RayForwardingRenderer<T>::screenShot()
  {
    std::string fileName = Renderer::screenShotFileName;
    std::vector<uint32_t> pixels;
    if (isMaster()) {
      fileName = fileName + "_master.png";
      pixels = masterFB.download();
      for (int iy=0;iy<fbSize.y/2;iy++) {
        uint32_t *top = pixels.data() + iy * fbSize.x;
        uint32_t *bot = pixels.data() + (fbSize.y-1-iy) * fbSize.x;
        for (int ix=0;ix<fbSize.x;ix++)
          std::swap(top[ix],bot[ix]);
      }
    } else {
      char suff[100];
      sprintf(suff,"_island%03i_rank%05i.png",
              comm->worker.islandIdx,comm->worker.withinIsland->rank);
      fileName = fileName + suff;
      
      std::vector<small_vec3f> hostFB;
      hostFB = localFB.download();
      for (int y=0;y<fbSize.y;y++) {
        const small_vec3f *line = hostFB.data() + (fbSize.y-1-y)*fbSize.x;
        for (int x=0;x<fbSize.x;x++) {
          vec3f col = from_half(line[x]);
          pixels.push_back(make_rgba(col) | (0xffu << 24));
        }
      }
    }
    
    stbi_write_png(fileName.c_str(),fbSize.x,fbSize.y,4,
                   pixels.data(),fbSize.x*sizeof(uint32_t));
    std::cout << "screenshot saved in '" << fileName << "'" << std::endl;

  }

  template<typename T>
  void RayForwardingRenderer<T>::traceRaysLocally()
  {
    CUDA_SYNC_CHECK();
    nodeRenderer->traceLocally(globals);
    CUDA_SYNC_CHECK();
  }
 
  template<typename Globals>
  __global__ void computeCompactionOffsets(Globals globals)
  {
    if (threadIdx.x != 0) return;
    int ofs = 0;
    for (int i=0;i<globals.numWorkers;i++) {
      globals.perRankSendOffsets[i] = ofs;
      ofs += globals.perRankSendCounts[i];
    }
  }
  
  template<typename Globals>
  __global__ void compactRays(Globals globals, int numRays)
  {
    int tid = threadIdx.x+blockIdx.x*blockDim.x;
    if (tid >= numRays) return;

    int dest = globals.rayNextNode[tid];
    if (dest < 0) return;

    int slot = atomicAdd(&globals.perRankSendOffsets[dest],1);
    globals.rayQueueOut[slot] = globals.rayQueueIn[tid];
  }

  template<typename T>
  void RayForwardingRenderer<T>::createSendQueue()
  {
    computeCompactionOffsets<<<1,1>>>
      (globals);
    CUDA_SYNC_CHECK();
    int blockSize = 1024;
    int numBlocks = divRoundUp(numRaysInQueue,blockSize);
    if (numBlocks)
      compactRays<<<numBlocks,blockSize>>>
        (globals,numRaysInQueue);
    CUDA_SYNC_CHECK();
  }

  template<typename T>
  int  RayForwardingRenderer<T>::exchangeRays()
  {
    host_sendCounts = perRankSendCounts.download();
    const int numWorkers = globals.numWorkers;
    const int myRank = this->myRank();
    auto island = comm->worker.withinIsland;

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
      sendByteCounts[i]  = to_i*sizeof(RayType);
      ofs += sendByteCounts[i];
    }

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
      recvByteCounts[i]  = from_i*sizeof(RayType);
      ofs += recvByteCounts[i];
      numReceived += from_i;
    }

    
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
  
  template<typename _RayType>
  inline __device__
  void RayForwardingRenderer<_RayType>::Globals::killRay(int rayID) const
  { rayNextNode[rayID] = -1; }
  

  template<typename _RayType>
  inline __device__
  void RayForwardingRenderer<_RayType>::Globals::addPixelContribution
  (int pixelID, vec3f addtl) const
  {
    vec3f *tgt = accumBuffer+pixelID;
    atomicAdd(&tgt->x,addtl.x);
    atomicAdd(&tgt->y,addtl.y);
    atomicAdd(&tgt->z,addtl.z);
  }

  template<typename _RayType>
  inline __device__
  void RayForwardingRenderer<_RayType>::Globals::forwardRay
  (int rayID, const RayType &ray, int nextNodeForThisRay)
    const
  {
    atomicAdd(&perRankSendCounts[nextNodeForThisRay],1);
    rayQueueIn[rayID]  = ray;
    rayNextNode[rayID] = nextNodeForThisRay;
  }
      
}
