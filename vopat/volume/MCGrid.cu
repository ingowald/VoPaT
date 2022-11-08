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

#include "vopat/volume/MCGrid.h"

namespace vopat {

  /*! assuming the min/max of the raw data values are already set in a
    macrocell, this updates the *mapped* min/amx values from a given
    transfer function */
  __global__ void mapMacroCell(MacroCell *mcData,
                               vec3i mcDims,
                               const vec4f *xfValues,
                               int numXfValues,
                               interval<float> xfDomain)
  {
    vec3i mcID(threadIdx.x+blockIdx.x*blockDim.x,
               threadIdx.y+blockIdx.y*blockDim.y,
               threadIdx.z+blockIdx.z*blockDim.z);

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
  }
  
  /*! recompute all macro cells' majorant value by remap each such
    cell's value range through the given transfer function */
  void MCGrid::mapXF(const vec4f *d_xfValues,
                     int xfSize,
                     range1f xfDomain)
  {
    // cuda block size:
    const vec3i bs = 4;
    // cuda num blocks
    const vec3i nb = divRoundUp(dd.dims,bs);
    mapMacroCell
      <<<(dim3)nb,(dim3)bs>>>
      (dd.cells,dd.dims,d_xfValues,xfSize,xfDomain);
  }
  
} // ::vopat
