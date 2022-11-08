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

  using range1f = interval<float>;
  
  struct MacroCell {
    /*! input values _before_ transfer function */
    interval<float> inputRange;
    /*! max opacity value _after_ transfer function */
    float maxOpacity;
  };

  struct MCGrid {
    struct DD {
      inline __device__ float getMajorant(const vec3i cellID) const
      {
        if (uint32_t(cellID.x) >= uint32_t(dims.x) ||
            uint32_t(cellID.y) >= uint32_t(dims.y) ||
            uint32_t(cellID.z) >= uint32_t(dims.z))
          { printf("invalid cellID!\n"); return 0.f; }
        const float f = cells[cellID.x+dims.x*(cellID.y+dims.y*(cellID.z))].maxOpacity;
        return f;
      }
      
      MacroCell *cells;
      vec3i      dims;
      affine3f   worldToMcSpace;
    };

    MCGrid()
    { dd.dims = vec3i(0); }

    CUDAArray<MacroCell> cells;

    /*! recompute all macro cells' majorant value by remap each such
        cell's value range through the given transfer function */
    void mapXF(const vec4f *d_xfValues,
               int xfSize,
               range1f xfDomain);
    
    DD dd;
  };
  
  /*! assuming the min/max of the raw data values are already set in a
    macrocell, this updates the *mapped* min/amx values from a given
    transfer function */
  __global__ void mapMacroCell(MacroCell *mcData,
                               vec3i mcDims,
                               const vec4f *xfValues,
                               int numXfValues,
                               interval<float> xfDomain);

}
  
