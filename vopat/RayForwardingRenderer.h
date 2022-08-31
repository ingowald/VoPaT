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

#include "vopat/AddWorkersRenderer.h"
#include "vopat/Ray.h"
#include <sstream>

#ifndef VOPAT_MAX_BOUNCES
  // 0 bounces == direct illum only
# define VOPAT_MAX_BOUNCES 1
#endif

namespace vopat {

  struct RayForwardingRenderer;
  
  struct NodeRenderer {
    virtual void generatePrimaryWave(RayForwardingRenderer *rfr) = 0;
    virtual void traceLocally(RayForwardingRenderer *rfr) = 0;
  };
  
  struct RayForwardingRenderer : public AddWorkersRenderer {
    
    struct Globals {
      /*! "kill" the ray in input queue position `rayID` - do nothing
          further with this ray, and do not forward it anywhere */
      inline __device__ void killRay(int rayID) const;
      
      /*! (atomically) add the given contribution to the specified pixel */
      inline __device__ void addPixelContribution(int pixelID, vec3f addtl) const;
      
      /*! mark ray at input queue position `rayID` to be forwarded to
          rank with given ID. This will overwrite the ray at input queue pos rayID */
      inline __device__ void forwardRay(int rayID, const Ray &ray, int nextNodeID) const;
      
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
      Ray         *rayQueueIn;
      
      /*! ray queue used for sending rays; generarting by
          sorting/compacting the input queue after ti has been traced
          locally */
      Ray         *rayQueueOut;
      
      /*! number of rays in the input queue */
      int          numRaysInQueue;

      bool fishy = false;
    };

    // /*! abstraction for any sort of renderer that will generate and/or
    //     shade/modify/bounce rays within this ray forwardign context */
    // struct NodeRenderer {
    //   virtual void setTransferFunction(const std::vector<vec4f> &cm,
    //                                    const interval<float> &range,
    //                                    const float density) {};
    //   virtual void setISO(int numActive,
    //                       const std::vector<int> &active,
    //                       const std::vector<float> &values,
    //                       const std::vector<vec3f> &colors) {};
    //   virtual void generatePrimaryWave(const Globals &globals) = 0;
    //   virtual void traceLocally(const Globals &globals, bool fishy) = 0;
    //   virtual void setLights(float ambient,
    //                          const std::vector<MPIRenderer::DirectionalLight> &dirLights) = 0;
    // };
    
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
    // void screenShot() override;

    // ==================================================================
    // things we intercept to know what to do
    // ==================================================================
    
    void resizeFrameBuffer(const vec2i &newSize)  override;
    // void setCamera(const vec3f &from,
    //                const vec3f &at,
    //                const vec3f &up,
    //                const float fov) override;
    
    void resetAccumulation()
    { // AddWorkersRenderer::resetAccumulation(); 
      accumBuffer.bzero(); }
    void traceRaysLocally();
    void createSendQueue();
    int  exchangeRays();

    //  void setTransferFunction(const std::vector<vec4f> &cm,
    //                           const interval<float> &range,
    //                           const float density) override
    // {
    //   nodeRenderer->setTransferFunction(cm,range,density);
    //   resetAccumulation();
    // }
    // void setISO(int numActive,
    //             const std::vector<int> &active,
    //             const std::vector<float> &values,
    //             const std::vector<vec3f> &colors) override
    // {
    //   nodeRenderer->setISO(numActive,active,values,colors);
    //   resetAccumulation();
    // }
    // void setLights(float ambient,
    //                const std::vector<MPIRenderer::DirectionalLight> &dirLights) override
    // {
    //   nodeRenderer->setLights(ambient,dirLights); 
    //   resetAccumulation();
    // }
    
    
    /*! ray queues we are expected to trace in the next step */
    CUDAArray<Ray>         rayQueueIn;
    
    /*! ray queues for sending out; in this one rays are sorted by
        rank they are supposed to go to */
    CUDAArray<Ray>         rayQueueOut;
    
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

  inline void RayForwardingRenderer::setCamera(const vec3f &from,
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
    
  template<typename Globals>
  __global__ void computeCompactionOffsets(Globals globals, bool fishy)
  {
    if (threadIdx.x != 0) return;
    int ofs = 0;
    for (int i=0;i<globals.islandSize;i++) {
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

  
  inline __device__
  void RayForwardingRenderer::Globals::killRay(int rayID) const
  { rayNextNode[rayID] = -1; }
  

  inline __device__
  void RayForwardingRenderer::Globals::addPixelContribution
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


  inline __device__
  void RayForwardingRenderer::Globals::forwardRay
  (int rayID, const Ray &ray, int nextNodeForThisRay)
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

}
