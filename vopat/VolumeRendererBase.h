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

// #include "owl/owl_device.h"
#include "model/Model.h"
#include "owl/owl.h"
// #include "vopat/RayForwardingRenderer.h"
#include "vopat/MacroCell.h"
#include "vopat/VoxelData.h"

#ifdef VOPAT_UMESH
# define VOPAT_UMESH_OPTIX 1
#endif

namespace vopat {

  struct VolumeRenderer {

    struct Globals {
      /*! put a scalar field throught he transfer function, and reutnr
        RGBA result */
      inline __device__ vec4f transferFunction(float f, bool dbg = false) const;

      /*! look up the given (world-space) 3D point in the volume, and
        return interpolated scalar value */
      inline __device__ bool getVolume(float &f, vec3f P, bool dbg = false) const;
      
      /*! look up the given (world-space) 3D point in the volume, and
        return the gradient */
      inline __device__ bool getGradient(vec3f &g, vec3f P, bool dbg = false) const;
      
      
      /* transfer function */
      struct {
        /*! the range of input data values that the transfer function is defined over */
        interval<float> domain;
        float           density = 1.f;
        vec4f          *values  = 0;
        int             numValues = 0;
      } xf;
      /*! my *lcoal* per-rank data */
#if VOPAT_UMESH
      UMeshData umesh;
#else
      VoxelData volume;
#endif
      /* macro cells */
      struct {
        MacroCell *data;
        vec3i      dims;
        int        width;
      } mc;
    
      vec3f gradientDelta;

      // box3f *rankBoxes;
      // box3f  myRegion;
    };

    VolumeRenderer(Model::SP model,
                   const std::string &baseFileName,
                   int islandRank,
                   /*! local rank's linear GPU ID on which to create the owl device */
                   int gpuID);
    void initMacroCells();
    void setTransferFunction(const std::vector<vec4f> &cm,
                             const interval<float> &xfDomain,
                             const float density);
    
    void setLights(float ambient,
                   const std::vector<MPIRenderer::DirectionalLight> &dirLights)
    {
      globals.lights.ambientTerm = ambient;
      // globals.lights.ambientEnvLight = ambient;
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
#if VOPAT_UMESH
    OWLBuffer /*float*/umeshScalarsBuffer;
    OWLBuffer /*vec3f*/umeshVerticesBuffer;
    OWLBuffer /*vec4i*/umeshTetsBuffer;
    OWLBuffer sharedFaceIndicesBuffer;
    OWLBuffer sharedFaceNeighborsBuffer;
    OWLGeomType umeshGT;
    OWLGeom     umeshGeom;
    OWLGroup    umeshAccel;
#else
# if VOPAT_VOXELS_AS_TEXTURE
    cudaArray_t          voxelArray;
# endif
    CUDAArray<float>     voxels;
#endif
    CUDAArray<vec4f>     colorMap;
    int mcWidth = 8;
    int islandRank = -1;

    OWLContext owl = 0;
    OWLModule  owlDevCode;
  };


  // ##################################################################
  // ##################################################################
  // ##################################################################
  
  /*! put a scalar field throught he transfer function, and reutnr
    RGBA result */
  inline __device__ vec4f VolumeRenderer::Globals::transferFunction(float f, bool dbg) const
  {
    if (this->xf.numValues == 0)
      return vec4f(0.f);
    if (this->xf.domain.lower >= this->xf.domain.upper)
      return vec4f(0.f);

    f = (f - this->xf.domain.lower) / (this->xf.domain.upper - this->xf.domain.lower);
    f = max(0.f,min(1.f,f));
    int i = min(this->xf.numValues-1,int(f * this->xf.numValues));
    return this->xf.values[i];
  }

#if VOPAT_UMESH
  inline __device__ bool VolumeRenderer::Globals::getVolume(float &f, vec3f P, bool dbg) const
  {
    if (!myRegion.contains(P)) { f = 0.f; return false; }
    return umesh.sample(f,P,dbg);
  }
  /*! look up the given 3D (world-space) point in the volume, and return the gradient */
  inline __device__ bool VolumeRenderer::Globals::getGradient(vec3f &g, vec3f P, bool dbg) const
  {
    if (!myRegion.contains(P)) {
      g = vec3f(0.f);
      return false;
    }

    const vec3f delta = gradientDelta;
    return umesh.gradient(g,P,delta,dbg);
  }
#else  
  /*! look up the given 3D (world-space) point in the volume, and return interpolated scalar value */
  inline __device__ bool VolumeRenderer::Globals::getVolume(float &f, vec3f P, bool dbg) const
  {
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

    return volume.sample(f,P-this->myRegion.lower,dbg);
  }

  /*! look up the given 3D (world-space) point in the volume, and return the gradient */
  inline __device__ bool VolumeRenderer::Globals::getGradient(vec3f &g, vec3f P, bool dbg) const
  {
    vec3ui cellID = vec3ui(floor(P) - this->myRegion.lower);
    if (// cellID.x < 0 || 
        (cellID.x >= this->volume.dims.x-1) ||
        // cellID.y < 0 || 
        (cellID.y >= this->volume.dims.y-1) ||
        // cellID.z < 0 || 
        (cellID.z >= this->volume.dims.z-1)) {
      g = vec3f(0.f);
      return false;
    }

    const vec3f delta = gradientDelta;
    return volume.gradient(g,P-this->myRegion.lower,delta,dbg);
  }
#endif
  
} // :vopat

