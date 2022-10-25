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

#include "common/vopat.h"
#include <owl/owl.h>
#include "model/Brick.h"
#include "model/Model.h"
#include "vopat/volume/MCGrid.h"

namespace vopat {

  struct Volume {
    typedef std::shared_ptr<Volume> SP;

    Volume(Brick::SP brick) : brick(brick) {}

    static Volume::SP createFrom(Brick::SP brick);
    
    struct DD {
      //      cudaTextureObject_t xfTexture;
      inline __device__ vec4f map(float f) const
      {
        if (domain.lower >= domain.upper ||
            f < domain.lower ||
            f > domain.upper)
          return vec4f(0.f);
        f = (f-domain.lower)*(numValues-1)/(domain.upper-domain.lower);
        int s0 = int(f);
        float ff = f - s0;
        int s1 = min(s0,numValues-1);
        return (1.f-ff)*values[s0]+ff*values[s1];
      }
      // struct {

      vec4f          *values;
      int             numValues;
      interval<float> domain;
      float           density;
      // } xf;
    };
    
    virtual void build(OWLContext owl,
                          OWLModule owlDevCode) = 0;
    virtual void setDD(OWLLaunchParams lp) = 0;
    virtual void addLPVars(std::vector<OWLVarDecl> &lpVars) = 0;
    
    virtual void buildMCs(MCGrid &mcGrid) = 0;
    
    void setTransferFunction(const std::vector<vec4f> &cm,
                             const interval<float> &domain,
                             const float density);

    DD xfGlobals;
    struct {
      CUDAArray<vec4f> colorMap;
      interval<float> domain;
      float density;
    } xf;
    
    Brick::SP brick;
  };
  
} // ::vopat

