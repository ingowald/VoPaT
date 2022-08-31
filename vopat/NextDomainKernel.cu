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

#include "NextDomainKernel.h"
#include "NodeRenderer.h"

namespace vopat {

  void NextDomainKernel::create(VopatNodeRenderer *vopat)
  {
    PING;
    myRank = vopat->volume.islandRank;
    auto &owl = vopat->volume.owl;
    auto &owlDevCode = vopat->volume.owlDevCode;

    OWLVarDecl vars[]
      = {
         { "proxies",OWL_BUFPTR,OWL_OFFSETOF(NextDomainKernel::Geom,proxies) },
         { nullptr }
    };
    gt = owlGeomTypeCreate(owl,OWL_GEOMETRY_USER,sizeof(Geom),vars,-1);
    owlGeomTypeSetBoundsProg(gt,owlDevCode,"proxyBounds");
    owlGeomTypeSetIntersectProg(gt,0,owlDevCode,"proxyIsec");
    
    geom = owlGeomCreate(owl,gt);
    owlGeomSetPrimCount(geom,proxies.size());
    proxiesBuffer
      = owlDeviceBufferCreate(owl,OWL_USER_TYPE(box3f),
                              proxies.size(),proxies.data());
    CUDA_SYNC_CHECK();
    owlBuildPrograms(owl);
    CUDA_SYNC_CHECK();

    blas = owlUserGeomGroupCreate(owl,1,&geom,OPTIX_BUILD_FLAG_ALLOW_UPDATE);
    owlGroupBuildAccel(blas);
    CUDA_SYNC_CHECK();
    
    tlas = owlInstanceGroupCreate(owl,1,&blas,0,0,
                                  OWL_MATRIX_FORMAT_OWL,
                                  OPTIX_BUILD_FLAG_ALLOW_UPDATE);
    owlGroupBuildAccel(tlas);
    CUDA_SYNC_CHECK();

    owlGeomSetBuffer(geom,"proxies",proxiesBuffer);
  }

  void NextDomainKernel::addLPVars(std::vector<OWLVarDecl> &lpVars)
  {
    lpVars.push_back({"proxies",OWL_BUFPTR,OWL_OFFSETOF(NextDomainKernel::LPData,proxies)});
    lpVars.push_back({"proxyBVH",OWL_GROUP,OWL_OFFSETOF(NextDomainKernel::LPData,proxyBVH)});
    lpVars.push_back({"myRank",OWL_INT,OWL_OFFSETOF(NextDomainKernel::LPData,myRank)});
  }
  
  void NextDomainKernel::setLPVars(OWLLaunchParams lp)
  {
    owlParamsSetBuffer(lp,"proxies",proxiesBuffer);
    owlParamsSetGroup(lp,"proxyBVH",tlas);
    owlParamsSet1i(lp,"myRank",myRank);
  }
    
  
}

