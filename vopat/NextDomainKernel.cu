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

#include "vopat/NextDomainKernel.h"
#include "vopat/VopatRenderer.h"

namespace vopat {

  int numShardsPerRank = 10;
  
  /*! exhanges shards across all range, build *all* ranks' proxies
    and value ranges, and upload to the proxiesBuffer and
    valueRangesBuffer */
  void NextDomainKernel::createProxies(VopatRenderer *vopat)
  {
    auto comm = vopat->comm;
    
    PING;fflush(0);
    std::vector<Shard> myShards
      = vopat->volume->brick->makeShards(numShardsPerRank);
    PING;fflush(0);
    PRINT(myShards.size());
    
    const int islandSize = comm->islandSize();
    auto island = comm->worker.withinIsland;

    // ------------------------------------------------------------------
    // compute how *many* shards each rank has
    // ------------------------------------------------------------------
    std::vector<int>   shardsOnRank(islandSize);
    const int myNumShards = (int)myShards.size();
    island->allGather(shardsOnRank,myNumShards);

    // ------------------------------------------------------------------
    // compute *total* number of shards, and allocate
    // ------------------------------------------------------------------
    int allNumShards = 0;
    for (auto sor : shardsOnRank) allNumShards += sor;
    PRINT(allNumShards);
    std::vector<Shard> allShards(allNumShards);
    
    
    // ------------------------------------------------------------------
    // *exchange all* shards across all ranks
    // ------------------------------------------------------------------
    island->allGather(allShards,myShards);
    
    // ------------------------------------------------------------------
    // compute proxies and value ranges
    // ------------------------------------------------------------------
    std::vector<Proxy>   allProxies(allNumShards);
    std::vector<range1f> allValueRanges(allNumShards);
    int numShardsDone = 0;
    for (int rank=0;rank<islandSize;rank++) {
      int shardID = numShardsDone++;
      auto shard = allShards[shardID];
      Proxy proxy;
      proxy.domain = shard.domain;
      proxy.rankID = rank;
      // majorant will be set later, once we have XF - just cannot be
      // 0.f or bounds prog will axe this
      proxy.majorant = -1.f;
      allProxies.push_back(proxy);
      allValueRanges.push_back(shard.valueRange);
    }
    this->numProxies = allProxies.size();
    
    proxiesBuffer = owlDeviceBufferCreate
      (vopat->owl,OWL_USER_TYPE(Proxy),allProxies.size(),allProxies.data());
    valueRangesBuffer = owlDeviceBufferCreate
      (vopat->owl,OWL_USER_TYPE(range1f),allValueRanges.size(),allValueRanges.data());
  }

  void NextDomainKernel::create(VopatRenderer *vopat)
  {
    PING;
    auto &owl = vopat->owl;
    if (!owl) return;
    
    auto &owlDevCode = vopat->owlDevCode;
    PRINT(owlDevCode);

    OWLVarDecl vars[]
      = {
         { "proxies",OWL_BUFPTR,OWL_OFFSETOF(NextDomainKernel::Geom,proxies) },
         { nullptr }
    };
    PING; fflush(0);
    gt = owlGeomTypeCreate(owl,OWL_GEOMETRY_USER,sizeof(Geom),vars,-1);
    owlGeomTypeSetBoundsProg(gt,owlDevCode,"proxyBounds");
    owlGeomTypeSetIntersectProg(gt,0,owlDevCode,"proxyIsec");

    PING;fflush(0);
    createProxies(vopat);
    PING;fflush(0);

    geom = owlGeomCreate(owl,gt);
    owlGeomSetPrimCount(geom,numProxies);
    PING;
    PRINT(proxiesBuffer);
    fflush(0);
    owlGeomSetBuffer(geom,"proxies",proxiesBuffer);

    CUDA_SYNC_CHECK();
    owlBuildPrograms(owl);
    CUDA_SYNC_CHECK();

    PING; fflush(0);
    
    blas = owlUserGeomGroupCreate(owl,1,&geom,OPTIX_BUILD_FLAG_ALLOW_UPDATE);
    owlGroupBuildAccel(blas);
    CUDA_SYNC_CHECK();
    
    PING; fflush(0);
    tlas = owlInstanceGroupCreate(owl,1,&blas,0,0,
                                  OWL_MATRIX_FORMAT_OWL,
                                  OPTIX_BUILD_FLAG_ALLOW_UPDATE);
    owlGroupBuildAccel(tlas);
    CUDA_SYNC_CHECK();

  }

  void NextDomainKernel::addLPVars(std::vector<OWLVarDecl> &lpVars)
  {
    lpVars.push_back({"proxies",OWL_BUFPTR,OWL_OFFSETOF(NextDomainKernel::LPData,proxies)});
    lpVars.push_back({"proxyBVH",OWL_GROUP,OWL_OFFSETOF(NextDomainKernel::LPData,proxyBVH)});
    lpVars.push_back({"myRank",OWL_INT,OWL_OFFSETOF(NextDomainKernel::LPData,myRank)});
  }
  
  void NextDomainKernel::setLPVars(OWLLaunchParams lp)
  {
    PING;
    PRINT(proxiesBuffer);
    owlParamsSetBuffer(lp,"proxies",proxiesBuffer);
    owlParamsSetGroup(lp,"proxyBVH",tlas);
    owlParamsSet1i(lp,"myRank",myRank);
  }
    
  
}

