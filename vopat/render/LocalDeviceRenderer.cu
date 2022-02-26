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

#include "LocalDeviceRenderer.h"

namespace vopat {

  __global__ void initMacroCell(DeviceKernelsBase::MacroCell *mcData,
                                vec3i mcDims,
                                int mcWidth,
                                float *voxelData,
                                vec3i voxelDims)
  {
    int ix = threadIdx.x+blockIdx.x*blockDim.x; if (ix >= mcDims.x) return;
    int iy = threadIdx.y+blockIdx.y*blockDim.y; if (iy >= mcDims.y) return;
    int iz = threadIdx.z+blockIdx.z*blockDim.z; if (iz >= mcDims.z) return;
    
    int mcIdx = ix + mcDims.x*(iy + mcDims.y*iz);
    auto &mc = mcData[mcIdx];

    /* compute begin/end of VOXELS for this macro-cell */
    vec3i begin = vec3i(ix,iy,iz)*mcWidth;
    vec3i end = min(begin + mcWidth + /* plus one for tri-lerp!*/1,
                    voxelDims);
    interval<float> valueRange;
    for (int iz=begin.z;iz<end.z;iz++)
      for (int iy=begin.y;iy<end.y;iy++)
        for (int ix=begin.x;ix<end.x;ix++)
          valueRange.extend(voxelData[ix+voxelDims.x*(iy+voxelDims.y*size_t(iz))]);
    mc.inputRange = valueRange;
    mc.mappedRange = valueRange;
  }

} // ::vopat
