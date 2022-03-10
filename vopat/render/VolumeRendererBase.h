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

#include "owl/owl_device.h"
#include "vopat/render/RayForwardingRenderer.h"
#include "vopat/render/MacroCell.h"

namespace vopat {

  struct VolumeRenderer {
    enum { MAX_DIR_LIGHTS = 2 };
    
    struct Globals {
      /*! hardcoded these, for now */
      inline __device__ float ambient()        const { return lights.ambient; }
      inline __device__ int   numDirLights()   const { return lights.numDirectional; }
      inline __device__ vec3f lightRadiance(int which)     const {
        return lights.directional[which].rad;
      }
      inline __device__ vec3f lightDirection(int which) const {
        return lights.directional[which].dir; }

      /*! put a scalar field throught he transfer function, and reutnr
        RGBA result */
      inline __device__ vec4f transferFunction(float f, bool dbg = false) const;

      /*! look up the given (world-space) 3D point in the volume, and
        return interpolated scalar value */
      inline __device__ bool getVolume(float &f, vec3f P, bool dbg = false) const;
      
      
      /* transfer function */
      struct {
        /*! the range of input data values that the transfer function is defined over */
        interval<float> domain;
        float           density = 1.f;
        vec4f          *values  = 0;
        int             numValues = 0;
      } xf;
      /*! my *lcoal* per-rank data */
      struct {
#if VOPAT_VOXELS_AS_TEXTURE
        cudaTextureObject_t texObj;
#else
        float *voxels;
#endif
        vec3i  dims;
      } volume;
      /* macro cells */
      struct {
        MacroCell *data;
        vec3i      dims;
        int        width;
      } mc;

      struct {
        float ambient = .1f;
        int numDirectional = 0;
        struct {
          vec3f dir = { .1f, 1.f, .1f };
          vec3f rad = { 1.f, 1.f, 1.f };
        } directional[MAX_DIR_LIGHTS];
      } lights;
      
      int    myRank;
      box3f *rankBoxes;
      int    numRanks;
      box3f  myRegion;
    };

    VolumeRenderer(Model::SP model,
                   const std::string &baseFileName,
                   int myRank);
    void initMacroCells();
    void setTransferFunction(const std::vector<vec4f> &cm,
                             const interval<float> &xfDomain,
                             const float density);
    
    void setLights(float ambient,
                   const std::vector<MPIRenderer::DirectionalLight> &dirLights)
    {
      globals.lights.ambient = ambient;
      globals.lights.numDirectional = dirLights.size();
      for (int i=0;i<min((int)dirLights.size(),(int)MAX_DIR_LIGHTS);i++) {
        globals.lights.directional[i].dir = normalize(dirLights[i].dir);
        globals.lights.directional[i].rad = dirLights[i].rad;
      }
    }


    Globals              globals;
    Model::SP            model;
    CUDAArray<box3f>     rankBoxes;
    Brick::SP            myBrick;
    CUDAArray<MacroCell> mcData;
#if VOPAT_VOXELS_AS_TEXTURE
    cudaArray_t          voxelArray;
#endif
    CUDAArray<float>     voxels;
    CUDAArray<vec4f>     colorMap;
    int mcWidth = 8;
    int myRank = -1;
  };


  // ##################################################################
  // ##################################################################
  // ##################################################################
  
  /*! put a scalar field throught he transfer function, and reutnr
    RGBA result */
  inline __device__ vec4f VolumeRenderer::Globals::transferFunction(float f, bool dbg) const
  {
    if (dbg)
      printf("mapping %f domain %f %f numvals %i ptr %lx...\n",
             f,xf.domain.lower,xf.domain.upper,
             this->xf.numValues,this->xf.values
             );
    if (this->xf.numValues == 0)
      return vec4f(0.f);
    if (this->xf.domain.lower >= this->xf.domain.upper)
      return vec4f(0.f);

    f = (f - this->xf.domain.lower) / (this->xf.domain.upper - this->xf.domain.lower);
    f = max(0.f,min(1.f,f));
    int i = min(this->xf.numValues-1,int(f * this->xf.numValues));
    return this->xf.values[i];
  }
  
  /*! look up the given 3D (world-space) point in the volume, and return interpolated scalar value */
  inline __device__ bool VolumeRenderer::Globals::getVolume(float &f, vec3f P, bool dbg) const
  {
#if VOPAT_VOXELS_AS_TEXTURE
    vec3ui cellID = vec3ui(floor(P) - this->myRegion.lower);
    if (// cellID.x < 0 || 
        (cellID.x >= this->volume.dims.x-1) ||
        // cellID.y < 0 || 
        (cellID.y >= this->volume.dims.y-1) ||
        // cellID.z < 0 || 
        (cellID.z >= this->volume.dims.z-1)) {
      f = 0.f;
      return false;
    }
    vec3f pos = vec3f(cellID) + vec3f(.5f); // Transform to CUDA texture cell-centric
    tex3D(&f,this->volume.texObj,pos.x,pos.y,pos.z);
    return true;
#else
#if 1
    // tri-lerp:
    vec3ui cellID = vec3ui(floor(P) - this->myRegion.lower);
    if (dbg) printf("cell %i %i %i\n",cellID.x,cellID.y,cellID.z);
    if (// cellID.x < 0 || 
        (cellID.x >= this->volume.dims.x-1) ||
        // cellID.y < 0 || 
        (cellID.y >= this->volume.dims.y-1) ||
        // cellID.z < 0 || 
        (cellID.z >= this->volume.dims.z-1)) {
      f = 0.f;
      return false;
    }

    vec3f  frac   = P - floor(P);

    size_t cx0 = cellID.x;
    size_t cy0 = cellID.y * size_t(this->volume.dims.x);
    size_t cz0 = cellID.z * (size_t(this->volume.dims.x) * size_t(this->volume.dims.y));
    size_t cx1 = cx0 + 1;
    size_t cy1 = cy0 + size_t(this->volume.dims.x);
    size_t cz1 = cz0 + (size_t(this->volume.dims.x) * size_t(this->volume.dims.y));
      
    float f000 = this->volume.voxels[cx0+cy0+cz0];
    float f001 = this->volume.voxels[cx1+cy0+cz0];
    float f010 = this->volume.voxels[cx0+cy1+cz0];
    float f011 = this->volume.voxels[cx1+cy1+cz0];
    float f100 = this->volume.voxels[cx0+cy0+cz1];
    float f101 = this->volume.voxels[cx1+cy0+cz1];
    float f110 = this->volume.voxels[cx0+cy1+cz1];
    float f111 = this->volume.voxels[cx1+cy1+cz1];

    float f00x = (1.f-frac.x)*f000 + frac.x*f001;
    float f01x = (1.f-frac.x)*f010 + frac.x*f011;
    float f10x = (1.f-frac.x)*f100 + frac.x*f101;
    float f11x = (1.f-frac.x)*f110 + frac.x*f111;

    float f0y = (1.f-frac.y)*f00x + frac.y*f01x;
    float f1y = (1.f-frac.y)*f10x + frac.y*f11x;

    float fz = (1.f-frac.z)*f0y + frac.z*f1y;
    f = fz;

    if (isnan(f)) {
      printf("f is NAN! P %f %f %f lerp %f %f %f\n",P.x,P.y,P.z,frac.x,frac.y,frac.z);
    }
    return true;
#else
    // nearest 
    vec3ui cellID = vec3ui(floor(P - this->myRegion.lower));
      
    if (// cellID.x < 0 || 
        (cellID.x >= this->volume.dims.x-1) ||
        // cellID.y < 0 || 
        (cellID.y >= this->volume.dims.y-1) ||
        // cellID.z < 0 || 
        (cellID.z >= this->volume.dims.z-1))
      return false;
      
    f = this->volume.voxels[cellID.x
                            +this->volume.dims.x*(cellID.y
                                                  +this->volume.dims.y*size_t(cellID.z))];
    return true;
#endif
#endif
  }

} // :vopat

