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

#include "vopat/volume/Volume.h"
#include "model/StructuredModel.h"

namespace vopat {
  
  struct StructuredVolume : public Volume {
    typedef std::shared_ptr<StructuredVolume> SP;

    static SP create(StructuredBrick::SP brick)
    { return std::make_shared<StructuredVolume>(brick); }
    
    StructuredVolume(StructuredBrick::SP brick)
      : Volume(brick), myBrick(brick)
    {}

    struct DD {
      inline __device__ bool sample(float &f, vec3f P, bool dbg) const;
      inline __device__ bool gradient(vec3f &g, vec3f P, vec3f delta, bool dbg) const;
      
      cudaTextureObject_t texObj;
      // cudaTextureObject_t texObjNN;
      vec3i dims;
    };
    
    void build(OWLContext owl,
               OWLModule owlDevCode) override;
    void setDD(OWLLaunchParams lp) override;
    void addLPVars(std::vector<OWLVarDecl> &lpVars) override;

    StructuredBrick::SP myBrick;
    DD globals;
  };


#ifdef __CUDA_ARCH__
  inline __device__
  bool StructuredVolume::DD::sample(float &f, vec3f P, bool dbg) const
  {
    P += vec3f(.5f); // Transform to CUDA texture cell-centric
    tex3D(&f,this->texObj,P.x,P.y,P.z);
    return true;
  }

  inline __device__
  bool StructuredVolume::DD::gradient(vec3f &g, vec3f P, vec3f delta, bool dbg) const
  {
    float right,left,top,bottom,front,back;
    sample(right, P+vec3f(delta.x,0.f,0.f),dbg);
    sample(left,  P-vec3f(delta.x,0.f,0.f),dbg);
    sample(top,   P+vec3f(0.f,delta.y,0.f),dbg);
    sample(bottom,P-vec3f(0.f,delta.y,0.f),dbg);
    sample(front, P+vec3f(0.f,0.f,delta.z),dbg);
    sample(back,  P-vec3f(0.f,0.f,delta.z),dbg);
    g = vec3f(right-left,top-bottom,front-back);
    return true;
  }
#endif
  
}


