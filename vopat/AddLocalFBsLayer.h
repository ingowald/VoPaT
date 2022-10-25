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

#include "common/vopat.h"
#include "common/mpi/Comms.h"
#include "common/CUDAArray.h"

namespace vopat {

  /*! a renderer where worker nodes fill a *local* frame buffer, and
      the master collects the final *added* images; _how_ a client
      renders its local frame buffer is still virtual, but all the
      frame buffer handling, compositing, and collecting of pixels is
      done by this class */
  struct AddLocalFBsLayer
  {
    /*! "device data" (dd) for this class; ie, this provides a
        device-side interface to add pixel fragments to the local
        rank's local frame buffer */
    struct DD {
      /*! (atomically) add the given contribution to the specified pixel */
      inline __device__ void addPixelContribution(vec2i globalPixelID, vec3f value) const;
      
      /*! (atomically) add the given contribution to the specified pixel */
      inline __device__ void addPixelContribution(uint32_t linearIdx, vec3f value) const;
      
      /*! transforms a "global" pixel ID into local island-frame buffer coordinates */
      inline __device__ vec2i globalToLocal(vec2i globalPixelID) const
      { return vec2i{globalPixelID.x,
                       (islandScale == 1)
                       ?((globalPixelID.y-islandBias)/islandScale)
                       :globalPixelID.y};
      }
      
      /*! transforms a "global" pixel ID into linear index that we use w/ a ray */
      inline __device__ uint32_t globalToIndex(vec2i globalPixelID) const
      { return globalPixelID.x+fullFbSize.x*globalPixelID.y; }

      /*! transforms a "global" pixel ID into linear index that we use w/ a ray */
      inline __device__ vec2i indexToGlobal(uint32_t index) const
      {
        int iy = index / fullFbSize.x;
        int ix = index - iy * fullFbSize.x;
        return { ix,iy };
      }

      /*! transforms from a given *local* (smaller) frame buffer
        within an island into global image coordinates */
      inline __device__ vec2i localToGlobal(vec2i localPixelID) const
      { return vec2i{localPixelID.x,localPixelID.y*islandScale+islandBias}; }
      
      /*! device-size address to this rank's local frame buffer */
      vec3f       *accumBuffer { 0 };
      /*! size of this local frame buffer */
      vec2i        fullFbSize { 0,0 };
      
      int          islandBias { 0 };
      int          islandScale { 0 };
    };
    
    AddLocalFBsLayer(CommBackend *comm) : comm(comm) {}

    void resize(const vec2i &newSize);
    
    /*! add all ranks' local frame buffers into the current accum
      buffer; and on master, convert to RGBA8 format and write to
      final frame buffer pointer */
    void addLocalFBs(uint32_t *appFbPointer);
    
    /*! clear the accumulation buffer - by default this layer will not
      only always add differnet ranks' local FBs together, but will
      _also_ add the ranks' local FBs to the accum buffer before
      adding those */
    void resetAccumulation()
    { localAccumBuffer.bzero(); numAccumulatedFrames = 0; }

    /*! debugging tool - dumps all frame buffers, both master and all
        per-rank island accum buffers */
    void screenShot();

    static std::string screenShotFileName;

    static void composeRegion(uint32_t *results,
                              const vec2i &ourRegionSize,
                              const small_vec3f *inputs,
                              int numRanks);

    inline bool isMaster() const { return comm->isMaster; }
    
    // ==================================================================
    
    /*! "temporary" buffer where current node receives all the lines
      that it has to compose */
    // small_vec3f    *compInputsMemory = nullptr;

    /*! all of the other rank's lines that they send to us for compositing */
    CUDAArray<small_vec3f> compInputsMemory;//localAccumBuffer;
    /*! all of the lines that _we_ send to _others_ (in lower
        prec). This is basically a reduced-precision version of the
        local accum buffer that we're sending out */
    CUDAArray<small_vec3f> compSendMemory;//localAccumBuffer;

    /*! temporary memory where this node writes its composed lines
      to, and from where those can then be sent on to the master */
    // uint32_t       *compResultMemory = nullptr;
    CUDAArray<uint32_t> compResultMemory;
    vec2i           islandFbSize { 0,0 };
    vec2i           fullFbSize { 0,0 };

    /*! only on the master: the buffer we receive final composited
        pixels in, properly assembled across all islands */
    CUDAArray<uint32_t> masterFB;
    
    /*! where we accumulate the local frame buffers into - use float
      format to avoid extinction artifacts */
    CUDAArray<vec3f>    localAccumBuffer;
    
    int numAccumulatedFrames = 0;
    
    DD dd;
    
    CommBackend *comm;
  };


  // __global__ void encodeAccumBufferForSending(vec2i fbSize,
  //                                             small_vec3f *lowerPrec,
  //                                             vec3f *accumBuffer,
  //                                             int numAccumFrames);
    

  // ==================================================================
  // IMPLEMENTATOIN
  // ==================================================================

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

  // inline __device__ void addToFB(vec3f *tgt, vec3f addtl)
  // {
  //   atomicAdd(&tgt->x,addtl.x);
  //   atomicAdd(&tgt->y,addtl.y);
  //   atomicAdd(&tgt->z,addtl.z);
  // }
  
#ifdef __CUDA_ARCH__
  inline __device__
  void AddLocalFBsLayer::DD::addPixelContribution(vec2i pixelID, vec3f fragment) const
  {
    float fm = reduce_max(fragment);
    if (fm == 0.f)
      return;

    pixelID = globalToLocal(pixelID);
    // pixelID.y = 
    // int global_iy = globalLinearPixelID / islandFbSize.x;
    // int global_ix = globalLinearPixelID - global_iy * islandFbSize.x;
    // int local_ix  = global_ix;
    // int local_iy  = (global_iy - islandIndex) / islandCount;

    vec3f *tgt = accumBuffer+(pixelID.y*fullFbSize.x+pixelID.x);

    atomicAdd(&tgt->x,fragment.x);
    atomicAdd(&tgt->y,fragment.y);
    atomicAdd(&tgt->z,fragment.z);
  }
  
  inline __device__
  void AddLocalFBsLayer::DD::addPixelContribution(uint32_t linearIdx, vec3f value) const
  {
    addPixelContribution(indexToGlobal(linearIdx),value);
  }
#endif

  
}

