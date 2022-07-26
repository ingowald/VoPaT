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
  
  inline __device__ bool checkOrigin(float x)
  {
    if (isnan(x) || fabsf(x) > 1e4f)
      return false;
    return true;
  }
  
  inline __device__ bool checkOrigin(vec3f org)
  { return checkOrigin(org.x) && checkOrigin(org.y) && checkOrigin(org.z); }

  template<typename Ray>
  inline __device__ bool checkOrigin(const Ray &ray)
  { return checkOrigin(ray.origin); }

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
      
      // int          myRank, numWorkers;
      int          islandRank, islandSize, islandIndex, islandCount;
      int          sampleID;
      Camera       camera;

      vec2i        worldFbSize, islandFbSize;
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

      bool fishy = false;
    };

    /*! abstraction for any sort of renderer that will generate and/or
        shade/modify/bounce rays within this ray forwardign context */
    struct NodeRenderer {
      virtual void setTransferFunction(const std::vector<vec4f> &cm,
                                       const interval<float> &range,
                                       const float density) {};
      virtual void setISO(int numActive,
                          const std::vector<int> &active,
                          const std::vector<float> &values,
                          const std::vector<vec3f> &colors) {};
      virtual void generatePrimaryWave(const Globals &globals) = 0;
      virtual void traceLocally(const Globals &globals, bool fishy) = 0;
      virtual void setLights(float ambient,
                             const std::vector<MPIRenderer::DirectionalLight> &dirLights) = 0;
    };
    
    RayForwardingRenderer(CommBackend *comm,
                          NodeRenderer *nodeRenderer,
                          int numSPP);

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
    void traceRaysLocally(bool fishy);
    void createSendQueue(bool fishy);
    int  exchangeRays();

     void setTransferFunction(const std::vector<vec4f> &cm,
                              const interval<float> &range,
                              const float density) override
    {
      nodeRenderer->setTransferFunction(cm,range,density);
      resetAccumulation();
    }
    void setISO(int numActive,
                const std::vector<int> &active,
                const std::vector<float> &values,
                const std::vector<vec3f> &colors) override
    {
      nodeRenderer->setISO(numActive,active,values,colors);
      resetAccumulation();
    }
    void setLights(float ambient,
                   const std::vector<MPIRenderer::DirectionalLight> &dirLights) override
    {
      nodeRenderer->setLights(ambient,dirLights); 
      resetAccumulation();
    }
    
    
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

    // num paths per pixel to be used
    int numSPP;
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


  template<typename RayType>
  void RayForwardingRenderer<RayType>::resizeFrameBuffer(const vec2i &newSize)
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
  
  template<typename T>
  void RayForwardingRenderer<T>::screenShot()
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

  template<typename T>
  void RayForwardingRenderer<T>::traceRaysLocally(bool fishy)
  {
    // const int myRank = globals.islandRank;//this->myRank();
    // printf("(%i) tracerays in\n",myRank);fflush(0);
    CUDA_SYNC_CHECK();
    nodeRenderer->traceLocally(globals,fishy);
    CUDA_SYNC_CHECK();
    // printf("(%i) tracerays OUT\n",myRank);fflush(0);
  }
 
  template<typename Globals>
  __global__ void computeCompactionOffsets(Globals globals, bool fishy)
  {
    if (threadIdx.x != 0) return;
    int ofs = 0;
    for (int i=0;i<globals.islandSize;i++) {
      globals.perRankSendOffsets[i] = ofs;
      ofs += globals.perRankSendCounts[i];
      // if (fishy) printf("(%i) sendCounts[%i] = %i\n",globals.islandRank,i,globals.perRankSendCounts[i]);
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
    auto ray = globals.rayQueueIn[tid];
    if (!checkOrigin(ray))
      printf("bad ray in compactrays %f %f %f\n",ray.origin.x,ray.origin.y,ray.origin.z);
    globals.rayQueueOut[slot] = ray;
  }


  template<typename Ray>
  __global__ void doCheckRays(int myRank, Ray *rays, int numRays, int *m_isBad)
  {
    int tid = threadIdx.x+blockIdx.x*blockDim.x;
    if (tid >= numRays) return;
    Ray ray = rays[tid];
    if (!checkOrigin(ray)) {
      printf("[%i] BAD RAY %i/%i :  %f %f %f\n",
             myRank,tid,numRays,
             ray.origin.x,
             ray.origin.y,
             ray.origin.z);
      *m_isBad = 1;
    }
  }
  
  
  template<typename Ray>
  void checkRays(int myRank, Ray *d_rays, int numRays, const char *text)
  {
    std::stringstream ss;
    ss << "(" << myRank << ") checking " << numRays << " rays (" << text << ")" << std::endl;
    std::cout << ss.str();
    int blockSize = 128;
    int numBlocks = divRoundUp(numRays,blockSize);
    int *is_bad = 0;
    if (!is_bad)
      CUDA_CALL(MallocManaged((void **)&is_bad,sizeof(int)));
    *is_bad = 0;
    if (numBlocks)
      doCheckRays<<<numBlocks,blockSize>>>(myRank,d_rays,numRays,is_bad);
    CUDA_SYNC_CHECK();
    if (*is_bad) {
      std::cout << "rank " << myRank << " contains at least one bad ray !" << std::endl;
      throw std::runtime_error("bad rays ....");
    }
  }

  
  template<typename T>
  void RayForwardingRenderer<T>::createSendQueue(bool fishy)
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

  template<typename T>
  int  RayForwardingRenderer<T>::exchangeRays()
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
      sendByteCounts[i]  = to_i*sizeof(RayType);
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
      recvByteCounts[i]  = from_i*sizeof(RayType);
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
  
  template<typename _RayType>
  inline __device__
  void RayForwardingRenderer<_RayType>::Globals::killRay(int rayID) const
  { rayNextNode[rayID] = -1; }
  

  template<typename _RayType>
  inline __device__
  void RayForwardingRenderer<_RayType>::Globals::addPixelContribution
  (int globalLinearPixelID, vec3f addtl) const
  {
    float fm = reduce_max(addtl);
    if (isnan(fm))
      printf("NAN IN CONTRIB\n");
    if (fm == 0.f)
      return;
    
    int global_iy = globalLinearPixelID / islandFbSize.x;
    int global_ix = globalLinearPixelID - global_iy * islandFbSize.x;
    int local_ix  = global_ix;
    int local_iy  = (global_iy - islandIndex) / islandCount;

    int ofs = local_iy*islandFbSize.x+local_ix;
    
    vec3f *tgt = accumBuffer+(ofs);

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
#if DEBUG_FORWARDS
    rayQueueIn[rayID].numFwds++;
    if (rayQueueIn[rayID].numFwds > 10)
      printf("ray got forwarded %i times...\n",rayQueueIn[rayID].numFwds);
#endif
    if (!checkOrigin(ray.origin))
      printf("Weird ray being forwarded here %f %f %f -> %i...\n",
             ray.origin.x,ray.origin.y,ray.origin.z,nextNodeForThisRay);
    rayNextNode[rayID] = nextNodeForThisRay;
  }

  inline __device__
  float fixDir(float f) { return (f==0.f)?1e-6f:f; }
  
  inline __device__
  vec3f fixDir(vec3f v)
  { return {fixDir(v.x),fixDir(v.y),fixDir(v.z)}; }
  
  
}
