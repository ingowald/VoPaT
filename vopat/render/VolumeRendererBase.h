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
#include "vopat/render/VoxelData.h"

namespace vopat {

  /*! misc helpers, might eventually move somewhere else */
  inline __device__
  void makeOrthoBasis(vec3f& u, vec3f& v, const vec3f& w)
  {
    v = abs(w.x) > abs(w.y)?normalize(vec3f(-w.z,0,w.x)):normalize(vec3f(0,w.z,-w.y));
    u = cross(v, w);
  }

  inline __device__ vec3f uniformSampleCone(const vec2f &u, float cosThetaMax)
  {
    float cosTheta = (1.f - u.x) + u.x * cosThetaMax;
    float sinTheta = sqrtf(1.f - cosTheta * cosTheta);
    float phi = u.y * 2.f * float(M_PI);
    return {cosf(phi) * sinTheta, sinf(phi) * sinTheta, cosTheta};
  }

  struct VolumeRenderer {
    enum { MAX_DIR_LIGHTS = 2 };
  

    struct Globals {
      /*! hardcoded these, for now */
      inline __device__ float ambient()         const { return lights.ambientTerm; }
      inline __device__ float ambientEnvLight() const { return lights.ambientEnvLight; }
      inline __device__ int   numDirLights()    const { return lights.numDirectional; }
      inline __device__ vec3f lightRadiance(int which)     const {
        return lights.directional[which].rad;
      }
      inline __device__ vec3f lightDirection(int which) const {
        return lights.directional[which].dir;
      }
      // inline __device__ vec3f sampleLightDirection(int which, Random &rnd, float &pdf) const {
      //   vec3f lightDir = lights.directional[which].dir;
      //   vec3f u,v;
      //   makeOrthoBasis(u,v,lightDir);
      //   const float cosThetaMax = .001f;
      //   vec3f coneSample = uniformSampleCone({rnd(),rnd()},cosThetaMax);
      //   pdf = 1.f / (2.f*float(M_PI) * (1.0f - cosThetaMax));
      //   return u*coneSample.x + v*coneSample.y + lightDir*coneSample.z;
      // }


      /*! sample a light, return light sample in lDir/lRad, and reutrn
          pdf of this sample */
      inline __device__ float sampleLight(Random &rnd,
                                         const vec3f surfP,
                                         const vec3f surfN,
                                         vec3f &lDir,
                                         vec3f &lRad) const
      {


        float pdf = 1.f;
        const int numLights = numDirLights();
        if (numLights == 0) return 0.f;
        
        int which = int(rnd() * numLights);
        if (which < 0 || which >= numLights) which = 0;
        pdf *= 1.f / numLights;

        lDir = lights.directional[which].dir;
        lRad = lights.directional[which].rad;

#if 0
        // HACK to create a giant light in the middle...
        if (length(surfP-vec3f(256.f)) > 50.f) return 0.f;
        else lRad *= 500.f;
#endif
        
        return pdf;
        // vec3f u,v;
        // makeOrthoBasis(u,v,lightDir);
        // const float cosThetaMax = .001f;
        // vec3f coneSample = uniformSampleCone({rnd(),rnd()},cosThetaMax);
        // pdf = 1.f / (2.f*float(M_PI) * (1.0f - cosThetaMax));
        // return u*coneSample.x + v*coneSample.y + lightDir*coneSample.z;
        // return 1.f/numLights;
      }
      
      // inline __device__ int uniformSampleOneLight(Random &rnd) const {
      //   const int numLights = numDirLights();
      //   int which = int(rnd() * numLights); if (which == numLights) which = 0;
      //   return which;
      // }

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

      struct {
        // ambient term that applies to every sample, w/o shadowing
        float ambientTerm = .0f;
        // ambient environment light, only added if paths get lost to env
        float ambientEnvLight = .2f;
        int numDirectional = 0;
        struct {
          vec3f dir = { .1f, 1.f, .1f };
          vec3f rad = { 1.f, 1.f, 1.f };
        } directional[MAX_DIR_LIGHTS];
      } lights;
    
      vec3f gradientDelta;

      int    islandRank;
      int    islandSize;
      box3f *rankBoxes;
      box3f  myRegion;
    };

    VolumeRenderer(Model::SP model,
                   const std::string &baseFileName,
                   int islandRank);
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
    CUDAArray<BVHNode> myBVHNodes;
    CUDAArray<float> myScalars;
    CUDAArray<vec3f> myVertices;
    CUDAArray<umesh::UMesh::Tet> myTets;
#else
# if VOPAT_VOXELS_AS_TEXTURE
    cudaArray_t          voxelArray;
# endif
    CUDAArray<float>     voxels;
#endif
    CUDAArray<vec4f>     colorMap;
    int mcWidth = 8;
    int islandRank = -1;
  };


  // ##################################################################
  // ##################################################################
  // ##################################################################
  
  /*! put a scalar field throught he transfer function, and reutnr
    RGBA result */
  inline __device__ vec4f VolumeRenderer::Globals::transferFunction(float f, bool dbg) const
  {
    // if (dbg)
    //   printf("mapping %f domain %f %f numvals %i ptr %lx...\n",
    //          f,xf.domain.lower,xf.domain.upper,
    //          this->xf.numValues,this->xf.values
    //          );
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

