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

#include "VolumeRendererBase.h"

namespace vopat {

  struct Vopat
  {
    struct Ray {
      struct {
        uint32_t    pixelID  : 29;
        uint32_t    dbg      :  1;
        uint32_t    crosshair:  1;
        uint32_t    isShadow :  1;
      };
      vec3f       origin;
#if 1
      inline __device__ void setDirection(vec3f v) { direction = to_half(fixDir(normalize(v))); }
      inline __device__ vec3f getDirection() const { return from_half(direction); }
      small_vec3f direction;
#else
      inline __device__ void setDirection(vec3f v) { direction = fixDir(normalize(v)); }
      inline __device__ vec3f getDirection() const { return direction; }
      vec3f direction;
#endif
      small_vec3f throughput;
    };
    
    using ForwardGlobals = typename RayForwardingRenderer<Ray>::Globals;
    using VolumeGlobals  = typename VolumeRenderer::Globals;
    
    static inline __device__
    Ray generateRay(const ForwardGlobals &globals,
                    vec2i pixelID,
                    vec2f pixelPos);

    static  inline __device__
    int computeNextNode(const VolumeGlobals &vopat,
                        const Ray &ray,
                        const float t_already_travelled,
                        bool dbg);
    static inline __device__
    int computeInitialRank(const VolumeGlobals &vopat,
                           Ray ray,
                           bool dbg=false);

    static inline __device__
    vec3f backgroundColor(const Vopat::Ray &ray,
                          const Vopat::ForwardGlobals &globals)
    {
      int iy = ray.pixelID / globals.fbSize.x;
      float t = iy / float(globals.fbSize.y);
      const vec3f c = (1.0f - t)*vec3f(1.0f, 1.0f, 1.0f) + t * vec3f(0.5f, 0.7f, 1.0f);
      return c;
    }
  
  };


  inline __device__
  bool boxTest(box3f box,
               Vopat::Ray ray,
               float &t0,
               float &t1,
               bool dbg=false)
  {
    vec3f dir = ray.getDirection();

    if (dbg)
      printf(" ray (%f %f %f)(%f %f %f) box (%f %f %f)(%f %f %f)\n",
             ray.origin.x,
             ray.origin.y,
             ray.origin.z,
             dir.x,
             dir.y,
             dir.z,
             box.lower.x,
             box.lower.y,
             box.lower.z,
             box.upper.x,
             box.upper.y,
             box.upper.z);
    
    vec3f t_lo = (box.lower - ray.origin) * rcp(dir);
    vec3f t_hi = (box.upper - ray.origin) * rcp(dir);
    
    vec3f t_nr = min(t_lo,t_hi);
    vec3f t_fr = max(t_lo,t_hi);

    t0 = max(t0,reduce_max(t_nr));
    t1 = min(t1,reduce_min(t_fr));
    if (dbg) printf("  -> t0 %f t1 %f\n",t0,t1);
    return (t0 <= t1);
  }

  
  inline __device__
  Vopat::Ray Vopat::generateRay(const Vopat::ForwardGlobals &globals,
                                                        vec2i pixelID,
                                                        vec2f pixelPos)
  {
    Ray ray;
    ray.pixelID  = pixelID.x + globals.fbSize.x*pixelID.y;
    ray.isShadow = false;
    ray.origin = globals.camera.lens_00;
    vec3f dir
      = globals.camera.dir_00
      + globals.camera.dir_du * (pixelID.x+pixelPos.x)
      + globals.camera.dir_dv * (pixelID.y+pixelPos.y);
    ray.setDirection(dir);
    ray.throughput = to_half(vec3f(1.f));
    return ray;
  }

  inline __device__
  int Vopat::computeNextNode(const Vopat::VolumeGlobals &vopat,
                             const Vopat::Ray &ray,
                             const float t_already_travelled,
                             bool dbg)
  {
    if (dbg) printf("finding next that's t >= %f and rank != %i\n",
                    t_already_travelled,vopat.myRank);
      
    int closest = -1;
    float t_closest = CUDART_INF;
    for (int i=0;i<vopat.numRanks;i++) {
      if (i == vopat.myRank) continue;
        
      float t0 = t_already_travelled * (1.f+1e-5f);
      float t1 = t_closest; 
      if (!boxTest(vopat.rankBoxes[i],ray,t0,t1,dbg))
        continue;
      if (dbg) printf("   accepted rank %i dist %f\n",i,t0);
      t_closest = t0;
      closest = i;
    }
    if (ray.dbg) printf("(%i) NEXT rank is %i\n",vopat.myRank,closest);
    return closest;
  }

  inline __device__
  int Vopat::computeInitialRank(const Vopat::VolumeGlobals &vopat,
                                Ray ray,
                                bool dbg)
  {
    int closest = -1;
    float t_closest = CUDART_INF;
    for (int i=0;i<vopat.numRanks;i++) {
      float t_min = 0.f;
      float t_max = t_closest;
      if (!boxTest(vopat.rankBoxes[i],ray,t_min,t_max))
        continue;
      closest = i;
      t_closest = t_min;
    }
    // if (ray.dbg) printf("(%i) INITIAL rank is %i\n",vopat.myRank,closest);
    return closest;
  }
  


  template<typename DeviceKernels>
  struct VopatNodeRenderer
    : public RayForwardingRenderer<typename DeviceKernels::Ray>::NodeRenderer,
      public VolumeRenderer
  {
    using inherited    = typename RayForwardingRenderer<typename DeviceKernels::Ray>::NodeRenderer;
    using Ray          = typename DeviceKernels::Ray;
    using ForwardGlobals = typename DeviceKernels::ForwardGlobals;
    using VolumeGlobals  = typename DeviceKernels::VolumeGlobals;
    
    VopatNodeRenderer(Model::SP model,
                      const std::string &baseFileName,
                      int myRank)
      : VolumeRenderer(model,baseFileName,myRank)
    // : inherited(model,baseFileName,myRank),
    //   VolumeRenderer(
    {}

    void generatePrimaryWave(const ForwardGlobals &forward) override;
    void traceLocally(const ForwardGlobals &forward) override;
    
    static inline __device__
    int computeInitialRank(const VolumeGlobals &vopat,
                           Ray ray, bool dbg = false);
    
    static inline __device__
    int computeNextNode(const ForwardGlobals &vopat,
                        const Ray &ray,
                        const float t_already_travelled,
                        bool dbg = false);

    void setTransferFunction(const std::vector<vec4f> &cm,
                             const interval<float> &range,
                             const float density) override
    { VolumeRenderer::setTransferFunction(cm,range,density); }

    void setLights(float ambient,
                   const std::vector<MPIRenderer::DirectionalLight> &dirLights) override
    { VolumeRenderer::setLights(ambient,dirLights); }
    
  };

  
  template<typename DeviceKernels>
  __global__
  void doTraceRaysLocally(typename VopatNodeRenderer<DeviceKernels>::ForwardGlobals forward,
                          typename VopatNodeRenderer<DeviceKernels>::VolumeGlobals  volume)
  {
    int tid = threadIdx.x+blockIdx.x*blockDim.x;
    if (tid >= forward.numRaysInQueue) return;

    DeviceKernels::traceRay(tid,forward,volume);
  }
  
  template<typename DeviceKernels>
  void VopatNodeRenderer<DeviceKernels>::traceLocally
  (const typename VopatNodeRenderer<DeviceKernels>::ForwardGlobals &forward)
  {
    // CUDA_SYNC_CHECK();
    int blockSize = 128;
    int numBlocks = divRoundUp(forward.numRaysInQueue,blockSize);
    if (numBlocks)
      doTraceRaysLocally<DeviceKernels><<<numBlocks,blockSize>>>
        (forward,VolumeRenderer::globals);
    // CUDA_SYNC_CHECK();
  }


  template<typename DeviceKernels>
  __global__
  void doGeneratePrimaryWave(typename VopatNodeRenderer<DeviceKernels>::ForwardGlobals vopat,
                             typename VopatNodeRenderer<DeviceKernels>::VolumeGlobals globals)
  {
    int ix = threadIdx.x + blockIdx.x*blockDim.x;
    int iy = threadIdx.y + blockIdx.y*blockDim.y;
    if (ix >= vopat.fbSize.x) return;
    if (iy >= vopat.fbSize.y) return;

    int myRank = vopat.myRank;
    typename DeviceKernels::Ray
      ray    = DeviceKernels::generateRay(vopat,vec2i(ix,iy),vec2f(.5f));
#if 0
    ray.dbg    = (vec2i(ix,iy) == vopat.fbSize/2);
#else
    ray.dbg    = false;
#endif
    ray.crosshair = (ix == vopat.fbSize.x/2) || (iy == vopat.fbSize.y/2);
    int dest   = DeviceKernels::computeInitialRank(globals,ray);

    if (dest < 0) {
      /* "nobody" owns this pixel, set to background on rank 0 */
      if (myRank == 0) {
        vopat.accumBuffer[ray.pixelID] += DeviceKernels::backgroundColor(ray,vopat);
      }
      return;
    }
    if (dest != myRank) {
      /* somebody else owns this pixel; we don't do anything */
      return;
    }
    int queuePos = atomicAdd(&vopat.perRankSendCounts[myRank],1);
    vopat.rayQueueIn[queuePos] = ray;
  }
  
  template<typename DeviceKernels>
  void VopatNodeRenderer<DeviceKernels>::generatePrimaryWave
  (const typename VopatNodeRenderer<DeviceKernels>::ForwardGlobals &vopat)
  {
    CUDA_SYNC_CHECK();
    vec2i blockSize(16);
    vec2i numBlocks = divRoundUp(vopat.fbSize,blockSize);
    // PRINT(numBlocks);
    doGeneratePrimaryWave<DeviceKernels><<<numBlocks,blockSize>>>(vopat,VolumeRenderer::globals);
    // CUDA_SYNC_CHECK();
  }


} // ::vopat


  
