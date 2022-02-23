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

#include "NodeRenderer.h"

namespace vopat {

  struct DeviceKernelsBase
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

    using VopatGlobals = typename RayForwardingRenderer<Ray>::Globals;
    
    struct OwnGlobals {
      box3f *rankBoxes;
      box3f  myRegion;
      float *myVoxels;
      vec3i  numVoxels;
    };

    static inline __device__
    vec3f albedo() {
      return vec3f(.8f, 1.f, .8f); // you'd normally sample this from the volume
    }
    
    static inline __device__
    float ambient() { return 0.1f; }

    static inline __device__ bool getVolume(float &f,
                                             const OwnGlobals &globals,
                                             vec3f P)
    {
      vec3ui cellID = vec3ui(floor(P - globals.myRegion.lower));
      if (// cellID.x < 0 || 
          (cellID.x >= globals.numVoxels.x-1) ||
          // cellID.y < 0 || 
          (cellID.y >= globals.numVoxels.y-1) ||
          // cellID.z < 0 || 
          (cellID.z >= globals.numVoxels.z-1))
        return false;
      
      f = globals.myVoxels[cellID.x
                                 +globals.numVoxels.x*(cellID.y
                                                       +globals.numVoxels.y*size_t(cellID.z))];
      return true;
    }
    
    static inline __device__ vec4f transferFunction(const VopatGlobals &vopat,
                                                    float f)
    {
      if (vopat.xf.numValues == 0)
        return f;
      if (vopat.xf.range.lower >= vopat.xf.range.upper)
        return f;

      f = (f - vopat.xf.range.lower) / (vopat.xf.range.upper - vopat.xf.range.lower);
      f = max(0.f,min(1.f,f));
      int i = min(vopat.xf.numValues-1,int(f * vopat.xf.numValues));
      return vopat.xf.values[i];
    }

    static inline __device__
    float floor(float f) { return ::floorf(f); }
  
    static inline __device__
    vec3f floor(vec3f v) { return { floor(v.x),floor(v.y),floor(v.z) }; }
  
  
    static inline __device__
    bool boxTest(box3f box,
                 Ray ray,
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

  
    static inline __device__
    Ray generateRay(const VopatGlobals &globals,
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

    static inline __device__
    vec3f backgroundColor(const Ray &ray,
                    const VopatGlobals &globals)
    {
      int iy = ray.pixelID / globals.fbSize.x;
      float t = iy / float(globals.fbSize.y);
      const vec3f c = (1.0f - t)*vec3f(1.0f, 1.0f, 1.0f) + t * vec3f(0.5f, 0.7f, 1.0f);
      return c;
    }

    static  inline __device__
    int computeNextNode(const VopatGlobals &vopat,
                        const OwnGlobals &globals,
                        const Ray &ray,
                        const float t_already_travelled,
                        bool dbg)
    {
      if (dbg) printf("finding next that's t >= %f and rank != %i\n",
                      t_already_travelled,vopat.myRank);
      
      int closest = -1;
      float t_closest = CUDART_INF;
      for (int i=0;i<vopat.numWorkers;i++) {
        if (i == vopat.myRank) continue;
        
        float t0 = t_already_travelled * (1.f+1e-5f);
        float t1 = t_closest; 
        if (!boxTest(globals.rankBoxes[i],ray,t0,t1,dbg))
          continue;
        if (dbg) printf("   accepted rank %i dist %f\n",i,t0);
        t_closest = t0;
        closest = i;
      }
      if (ray.dbg) printf("(%i) NEXT rank is %i\n",vopat.myRank,closest);
      return closest;
    }

    static inline __device__
    int computeInitialRank(const VopatGlobals &vopat,
                           const OwnGlobals &globals,
                           Ray ray,
                           bool dbg=false)
    {
      int closest = -1;
      float t_closest = CUDART_INF;
      for (int i=0;i<vopat.numWorkers;i++) {
        float t_min = 0.f;
        float t_max = t_closest;
        if (!boxTest(globals.rankBoxes[i],ray,t_min,t_max))
          continue;
        closest = i;
        t_closest = t_min;
      }
      if (ray.dbg) printf("(%i) INITIAL rank is %i\n",vopat.myRank,closest);
      return closest;
    }
  
  };
  

  template<typename DeviceKernels>
  struct LocalDeviceRenderer
    : public RayForwardingRenderer<typename DeviceKernels::Ray>::NodeRenderer
  {
    using Ray          = typename DeviceKernels::Ray;
    using OwnGlobals   = typename DeviceKernels::OwnGlobals;
    using VopatGlobals = typename DeviceKernels::VopatGlobals;
    
    LocalDeviceRenderer(Model::SP model,
                       const std::string &baseFileName,
                       int myRank)
      : model(model)
    {
      if (myRank < 0)
        return;
      
      // ------------------------------------------------------------------
      // upload per-rank boxes
      // ------------------------------------------------------------------
      std::vector<box3f> hostRankBoxes;
      for (auto brick : model->bricks)
        hostRankBoxes.push_back(brick->spaceRange);
      rankBoxes.upload(hostRankBoxes);
      globals.rankBoxes = rankBoxes.get();

      myBrick = model->bricks[myRank];
      const std::string fileName = Model::canonicalRankFileName(baseFileName,myRank);
      std::vector<float> loadedVoxels = myBrick->load(fileName);
      
      voxels.upload(loadedVoxels);
      globals.myVoxels  = voxels.get();
      globals.numVoxels = myBrick->numVoxels;//voxelRange.size();
      globals.myRegion  = myBrick->spaceRange;
    };

    /*! one box per rank, which rays can use to find neext rank to send to */
    CUDAArray<box3f> rankBoxes;
    OwnGlobals       globals;
    Brick::SP        myBrick;
    CUDAArray<float> voxels;

    void generatePrimaryWave(const VopatGlobals &globals) override;
    void traceLocally(const VopatGlobals &globals) override;

    static inline __device__
    int computeInitialRank(const VopatGlobals &vopat,
                           const OwnGlobals &globals,
                           Ray ray, bool dbg = false);
    
    static inline __device__
    int computeNextNode(const VopatGlobals &vopat,
                        const OwnGlobals &globals,
                        const Ray &ray,
                        const float t_already_travelled,
                        bool dbg = false);
    
    Model::SP model;
  };

  
  template<typename DeviceKernels>
  __global__
  void doTraceRaysLocally(typename LocalDeviceRenderer<DeviceKernels>::VopatGlobals vopat,
                          typename LocalDeviceRenderer<DeviceKernels>::OwnGlobals   globals)
  {
    int tid = threadIdx.x+blockIdx.x*blockDim.x;
    if (tid >= vopat.numRaysInQueue) return;

    DeviceKernels::traceRay(tid,vopat,globals);
  }
  
  template<typename DeviceKernels>
  void LocalDeviceRenderer<DeviceKernels>::traceLocally
  (const typename LocalDeviceRenderer<DeviceKernels>::VopatGlobals &vopat)
  {
    // CUDA_SYNC_CHECK();
    int blockSize = 512;
    int numBlocks = divRoundUp(vopat.numRaysInQueue,blockSize);
    if (numBlocks)
      doTraceRaysLocally<DeviceKernels><<<numBlocks,blockSize>>>
        (vopat,globals);
    // CUDA_SYNC_CHECK();
  }


  template<typename DeviceKernels>
  __global__
  void doGeneratePrimaryWave(typename LocalDeviceRenderer<DeviceKernels>::VopatGlobals vopat,
                             typename LocalDeviceRenderer<DeviceKernels>::OwnGlobals   globals)
  {
    int ix = threadIdx.x + blockIdx.x*blockDim.x;
    int iy = threadIdx.y + blockIdx.y*blockDim.y;
    if (ix >= vopat.fbSize.x) return;
    if (iy >= vopat.fbSize.y) return;

    int myRank = vopat.myRank;
    typename DeviceKernels::Ray
      ray    = DeviceKernels::generateRay(vopat,vec2i(ix,iy),vec2f(.5f));
    ray.dbg    = false;//(vec2i(ix,iy) == vopat.fbSize/2);
    ray.crosshair = (ix == vopat.fbSize.x/2) || (iy == vopat.fbSize.y/2);
    int dest   = DeviceKernels::computeInitialRank(vopat,globals,ray);

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
  void LocalDeviceRenderer<DeviceKernels>::generatePrimaryWave
  (const typename LocalDeviceRenderer<DeviceKernels>::VopatGlobals &vopat)
  {
    CUDA_SYNC_CHECK();
    vec2i blockSize(16);
    vec2i numBlocks = divRoundUp(vopat.fbSize,blockSize);
    // PRINT(numBlocks);
    doGeneratePrimaryWave<DeviceKernels><<<numBlocks,blockSize>>>(vopat,globals);
    // CUDA_SYNC_CHECK();
  }


} // ::vopat


  
