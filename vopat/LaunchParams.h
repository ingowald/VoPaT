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

#pragma once

#include "common/vopat.h"
#include "vopat/ForwardingLayer.h"
#include "vopat/AddLocalFBsLayer.h"
// #include "vopat/volume/StructuredVolume.h"
// #include "vopat/volume/UMeshVolume.h"
// #include "vopat/SurfaceIntersector.h"
#include "vopat/NextDomainKernel.h"

#include "vopat/volume/UMeshVolume.h"
#include "vopat/volume/StructuredVolume.h"
#include "vopat/Camera.h"

namespace vopat {

  using namespace owl;
  using namespace owl::common;
  using Random = owl::common::LCG<8>;

#if VOPAT_USE_RAFI
  using ForwardGlobals = rafi::DeviceInterface<vopat::Ray>;
#else
  using ForwardGlobals = typename ForwardingLayer::DD;
#endif
  // using VolumeGlobals  = typename VolumeRenderer::Globals;
  // using SurfaceGlobals = typename SurfaceIntersector::Globals;
    
  /*! "triangle mesh" geometry type for shared-faces method */
  // struct UMeshGeom {
  //   vec4i *tets;
  //   vec3f *vertices;
  //   float *scalars;
  //   vec2i *tetsOnFace;
  // };

  struct LaunchParams {
    static inline __device__ const LaunchParams &get();

    MCGrid::DD mcGrid;
    AddLocalFBsLayer::DD     fbLayer;
    ForwardGlobals           forwardGlobals;
    // VolumeGlobals            volumeGlobals;
    // SurfaceGlobals           surfaceGlobals;
    NextDomainKernel::LPData nextDomainKernel;
    Camera                   camera;
    struct{
      int type;
      union {
        UMeshVolume::DD      umesh;
        StructuredVolume::DD structured;
      };
#if __CUDA_ARCH__
      inline __device__ bool sample(float &f, vec3f P, bool dbg) const
      {
        if (type == VolumeSamplerType_Structured)
          return structured.sample(f,P,dbg);
        else
          return umesh.sample(f,P,dbg);
      }
#endif
      Volume::DD xf;
    } volumeSampler;

    OptixTraversableHandle replicatedSurfaceBVH;

    int emergency;
    int rank;
    int sampleID;
  };
  
}
