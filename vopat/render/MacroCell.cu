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

#include "vopat/render/MacroCell.h"

namespace vopat {

#if VOPAT_UMESH
  // this gets done in VolumeRendererBase.cu:rasterTets
#else
/*! computes initial *input* range of the macrocells; ie, min/max of
    raw data values *excluding* any transfer fucntion */
  __global__ void initMacroCell(MacroCell *mcData,
                                vec3i mcDims,
                                int mcWidth,
                                VoxelData volume)
  {
    vec3i mcID(threadIdx.x+blockIdx.x*blockDim.x,
               threadIdx.y+blockIdx.y*blockDim.y,
               threadIdx.z+blockIdx.z*blockDim.z);
    
    if (mcID.x >= mcDims.x) return;
    if (mcID.y >= mcDims.y) return;
    if (mcID.z >= mcDims.z) return;
    
    int mcIdx = mcID.x + mcDims.x*(mcID.y + mcDims.y*mcID.z);
    auto &mc = mcData[mcIdx];

    /* compute begin/end of VOXELS for this macro-cell */
    vec3i begin = mcID*mcWidth;
    vec3i end = min(begin + mcWidth + /* plus one for tri-lerp!*/1,
                    volume.dims);
    interval<float> valueRange;
    for (int iz=begin.z;iz<end.z;iz++)
      for (int iy=begin.y;iy<end.y;iy++)
        for (int ix=begin.x;ix<end.x;ix++) {
#if VOPAT_VOXELS_AS_TEXTURE
          float f;
          tex3D(&f,volume.texObjNN,ix,iy,iz);
          valueRange.extend(f);
#else
          valueRange.extend(volume.voxels[ix+volume.dims.x*(iy+volume.dims.y*size_t(iz))]);
#endif
        }
    mc.inputRange = valueRange;
    mc.maxOpacity = 1.f;
  }
#endif
  
  /*! assuming the min/max of the raw data values are already set in a
    macrocell, this updates the *mapped* min/amx values from a given
    transfer function */
  __global__ void mapMacroCell(MacroCell *mcData,
                               vec3i mcDims,
                               vec4f *xfValues,
                               int numXfValues,
                               interval<float> xfDomain)
  {
    vec3i mcID(threadIdx.x+blockIdx.x*blockDim.x,
               threadIdx.y+blockIdx.y*blockDim.y,
               threadIdx.z+blockIdx.z*blockDim.z);

    if (0 && mcID == vec3i(0))
      printf("mapping macrocells to transfer function; input value range supposedly %f %f\n",
             xfDomain.lower,xfDomain.upper);
    
    if (mcID.x >= mcDims.x) return;
    if (mcID.y >= mcDims.y) return;
    if (mcID.z >= mcDims.z) return;
    
    int mcIdx = mcID.x + mcDims.x*(mcID.y + mcDims.y*mcID.z);
    auto &mc = mcData[mcIdx];

    float lo = max(mc.inputRange.lower,xfDomain.lower);
    float hi = min(mc.inputRange.upper,xfDomain.upper);
    if (lo > hi) {
      mc.maxOpacity = 0.f;
      return;
    }

    lo = (lo - xfDomain.lower) / (xfDomain.upper - xfDomain.lower);
    hi = (hi - xfDomain.lower) / (xfDomain.upper - xfDomain.lower);

    int lo_idx = max(0,int(lo*numXfValues));
    int hi_idx = min(numXfValues-1,int(ceil(hi*numXfValues)));
    float maxOpacity = 0.f;
    for (int i=lo_idx;i<=hi_idx;i++)
      maxOpacity = max(maxOpacity,xfValues[i].w);
    mc.maxOpacity = maxOpacity;
    
    if (0 && mcID == mcDims/2)
      printf("center macrocell at %i %i %i: input range %f %f, max opacity %f\n",
             mcID.x,mcID.y,mcID.z,
             mc.inputRange.lower,mc.inputRange.upper,mc.maxOpacity
             );
  }

} // ::vopat
