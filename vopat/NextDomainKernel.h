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
#include "owl/owl.h"
#include "Ray.h"

namespace vopat {

  struct VopatNodeRenderer;
  
  struct NextDomainKernel {
    enum { RETRY = 1<<30 };
    
    struct Proxy {
      box3f domain;
      int   rankID;
      float majorant;
    };

    struct PRD {
      uint32_t   skipCurrentRank:1;
      uint32_t   dbg:1;
      int   closestRank:30;
      float closestDist;
    };
    
    struct LPData {
      int                    myRank;
      Proxy                 *proxies;
      OptixTraversableHandle proxyBVH;
    };

    struct Geom {
      Proxy                 *proxies;
    };
    
    void create(VopatNodeRenderer *vopat);
    
    void addLPVars(std::vector<OWLVarDecl> &lpVars);
    void setLPVars(OWLLaunchParams lp);
    
    std::vector<Proxy> proxies;
    OWLBuffer proxiesBuffer;
    OWLGeomType gt;
    OWLGeom    geom;
    OWLGroup   blas, tlas;
    int        myRank;
  };

} // ::vopat
