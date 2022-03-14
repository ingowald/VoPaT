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

#include "vopat/common.h"

namespace vopat {

  struct MacroCell {
    /*! input values _before_ transfer function */
    interval<float> inputRange;
    /*! max opacity value _after_ transfer function */
    float maxOpacity;
  };

  struct VoxelData {
#if VOPAT_VOXELS_AS_TEXTURE
    cudaTextureObject_t texObj;
    cudaTextureObject_t texObjNN;
#else
    float *devPtr;
#endif
    vec3i dims;
  };

  /*! computes initial *input* range of the macrocells; ie, min/max of
    raw data values *excluding* any transfer fucntion */
  __global__ void initMacroCell(MacroCell *mcData,
                                vec3i mcDims,
                                int mcWidth,
                                VoxelData volume);
  
  /*! assuming the min/max of the raw data values are already set in a
    macrocell, this updates the *mapped* min/amx values from a given
    transfer function */
  __global__ void mapMacroCell(MacroCell *mcData,
                               vec3i mcDims,
                               vec4f *xfValues,
                               int numXfValues,
                               interval<float> xfDomain);

}
  
