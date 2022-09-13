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

namespace vopat {

  struct Volume {
    typedef std::shared_ptr<Volume> SP;

    Volume(Brick::SP brick) : brick(brick) {}

    static Volume::SP createFrom(Brick::SP brick);
    
    struct DD {
      //      cudaTextureObject_t xfTexture;
      struct {
        vec4f          *values;
        int             numValues;
        interval<float> domain;
        float           density;
      } xf;
    };
    
    virtual void build(OWLContext owl,
                          OWLModule owlDevCode) = 0;
    virtual void setDD(OWLLaunchParams lp) = 0;
    virtual void addLPVars(std::vector<OWLVarDecl> &lpVars) = 0;

    void setTransferFunction(const std::vector<vec4f> &cm,
                             const interval<float> &domain,
                             const float density);

    struct {
      CUDAArray<vec4f> colorMap;
      interval<float> domain;
      float density;
    } xf;
    
    Brick::SP brick;
  };
  
} // ::vopat

