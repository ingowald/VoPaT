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

#include "common/CUDAArray.h"

namespace vopat {

  struct MacroCell {
    /*! input values _before_ transfer function */
    interval<float> inputRange;
    /*! max opacity value _after_ transfer function */
    float maxOpacity;
  };

  struct MCGrid {
    struct DD {
      inline __device__ float getMajorant(const vec3i coord) const
      {
        if (uint32_t(coord.x) >= uint32_t(dims.x) |
            uint32_t(coord.y) >= uint32_t(dims.y) |
            uint32_t(coord.z) >= uint32_t(dims.z))
          { printf("invalid coord!\n"); return 0.f; }
        return cells[coord.x+dims.x*(coord.y+dims.y*(coord.z))].maxOpacity;
      }
      
      MacroCell *cells;
      vec3i      dims;
      affine3f   worldToMcSpace;
      
      // template<typename Lambda>
      // inline __device__ void march(Ray &ray, const Lambda &lambda);
    };
    
    // void addLPVars(std::vector<OWLVarDecl> &lpVars,
    //                // offset of this kernel's vars within LP
    //                uint32_t kernelOffset);
    // void setLPVars(OWLLaunchParams lp);

    CUDAArray<MacroCell> cells;
    vec3i                dims;
    
    DD dd;
  };
  
// #if VOPAT_UMESH
// #else
//   /*! computes initial *input* range of the macrocells; ie, min/max of
//     raw data values *excluding* any transfer fucntion */
//   __global__ void initMacroCell(MacroCell *mcData,
//                                 vec3i mcDims,
//                                 int mcWidth,
//                                 VoxelData volume);
// #endif
  
  /*! assuming the min/max of the raw data values are already set in a
    macrocell, this updates the *mapped* min/amx values from a given
    transfer function */
  __global__ void mapMacroCell(MacroCell *mcData,
                               vec3i mcDims,
                               vec4f *xfValues,
                               int numXfValues,
                               interval<float> xfDomain);

}
  
