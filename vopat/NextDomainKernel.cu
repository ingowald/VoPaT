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
#include <sstream>

namespace vopat {

  int numVolumeProxiesPerRank = 10; //64;
  
  /*! exhanges volumeProxies across all range, build *all* ranks' proxies
    and value ranges, and upload to the proxiesBuffer and
    valueRangesBuffer */
  void NextDomainKernel::createProxies(VopatRenderer *vopat)
  {
    auto comm = vopat->comm;

    std::vector<VolumeProxy> myVolumeProxies
      = vopat->volume->brick->makeVolumeProxies(numVolumeProxiesPerRank);
    
    const int islandSize = comm->islandSize();
    auto island = comm->worker.withinIsland;
    this->myRank = comm->myRank();
    
    // ------------------------------------------------------------------
    // compute how *many* volumeProxies each rank has
    // ------------------------------------------------------------------
    std::vector<int>   volumeProxiesOnRank(islandSize);
    const int myNumVolumeProxies = (int)myVolumeProxies.size();
    island->allGather(volumeProxiesOnRank,myNumVolumeProxies);
    // ------------------------------------------------------------------
    // compute *total* number of volumeProxies, and allocate
    // ------------------------------------------------------------------
    int allNumVolumeProxies = 0;
    for (auto sor : volumeProxiesOnRank) allNumVolumeProxies += sor;
    std::vector<VolumeProxy> allVolumeProxies(allNumVolumeProxies);
    // for (int i=0;i<volumeProxiesOnRank.size();i++)
    //   ss << "  volumeProxies from rank " << i << " : " << volumeProxiesOnRank[i] << std::endl;
    // ss << "Sum all volumeProxies: " << allNumVolumeProxies << std::endl;
    // ------------------------------------------------------------------
    // *exchange all* volumeProxies across all ranks
    // ------------------------------------------------------------------
    island->allGather(allVolumeProxies,myVolumeProxies);

    // ------------------------------------------------------------------
    // compute proxies and value ranges
    // ------------------------------------------------------------------
    std::vector<Proxy>   allProxies;
    std::vector<range1f> allValueRanges;
    int numVolumeProxiesDone = 0;
    for (int rank=0;rank<islandSize;rank++)
      for (int _i=0;_i<volumeProxiesOnRank[rank];_i++) {
        int volumeProxyID = numVolumeProxiesDone++;
        auto volumeProxy = allVolumeProxies[volumeProxyID];
        Proxy proxy;
        proxy.domain = volumeProxy.domain;
        proxy.rankID = rank;
        // majorant will be set later, once we have XF - just cannot be
        // 0.f or bounds prog will axe this
        proxy.majorant = -1.f;
        allProxies.push_back(proxy);
        allValueRanges.push_back(volumeProxy.valueRange);
      }
    this->numProxies = allProxies.size();

    proxiesBuffer = owlDeviceBufferCreate
      (vopat->owl,OWL_USER_TYPE(Proxy),allProxies.size(),allProxies.data());
    valueRangesBuffer = owlDeviceBufferCreate
      (vopat->owl,OWL_USER_TYPE(range1f),allValueRanges.size(),allValueRanges.data());
  }

  void NextDomainKernel::create(VopatRenderer *vopat)
  {
    auto &owl = vopat->owl;
    if (!owl) return;

    auto &owlDevCode = vopat->owlDevCode;
    this->myRank = vopat->comm->myRank();

    OWLVarDecl vars[]
      = {
         { "proxies",OWL_BUFPTR,OWL_OFFSETOF(NextDomainKernel::Geom,proxies) },
         { nullptr }
    };
    gt = owlGeomTypeCreate(owl,OWL_GEOMETRY_USER,sizeof(Geom),vars,-1);
    owlGeomTypeSetAnyHit(gt,0,owlDevCode,"proxyAH");
    owlGeomTypeSetClosestHit(gt,0,owlDevCode,"proxyCH");
    owlGeomTypeSetBoundsProg(gt,owlDevCode,"proxyBounds");
    owlGeomTypeSetIntersectProg(gt,0,owlDevCode,"proxyIsec");

    createProxies(vopat);

    geom = owlGeomCreate(owl,gt);
    owlGeomSetPrimCount(geom,numProxies);
    owlGeomSetBuffer(geom,"proxies",proxiesBuffer);

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
  }

  void NextDomainKernel::addLPVars(std::vector<OWLVarDecl> &lpVars, uint32_t kernelOffset)
  {
    lpVars.push_back({"proxies",OWL_BUFPTR,
                      kernelOffset+OWL_OFFSETOF(NextDomainKernel::LPData,proxies)});
    lpVars.push_back({"proxyBVH",OWL_GROUP,
                      kernelOffset+OWL_OFFSETOF(NextDomainKernel::LPData,proxyBVH)});
    lpVars.push_back({"myRank",OWL_INT,
                      kernelOffset+OWL_OFFSETOF(NextDomainKernel::LPData,myRank)});
  }
  
  void NextDomainKernel::setLPVars(OWLLaunchParams lp)
  {
    owlParamsSetBuffer(lp,"proxies",proxiesBuffer);
    owlParamsSetGroup(lp,"proxyBVH",tlas);
    owlParamsSet1i(lp,"myRank",myRank);
  }


  inline __device__
  float remap(const float f, const range1f &range)
  {
    return (f - range.lower) / (range.upper - range.lower);
  }

  __global__ void updateProxies(const vec4f *xfValues,
                                int xfSize,
                                range1f xfDomain,
                                NextDomainKernel::Proxy *proxies,
                                range1f *valueRanges,
                                int numProxies)
  {
    int tid = threadIdx.x+blockIdx.x*blockDim.x;
    if (tid >= numProxies) return;

    range1f valueRange = valueRanges[tid];
    float majorant = 0.f;
    if (xfDomain.lower != xfDomain.upper) {
    
      valueRange.lower = remap(valueRange.lower,xfDomain);
      valueRange.upper = remap(valueRange.upper,xfDomain);

      if (valueRange.upper >= 0.f && valueRange.lower <= 1.f) {
        int numCMIntervals = xfSize-1;
        int idx_lo = clamp(int(valueRange.lower*numCMIntervals),0,numCMIntervals);
        int idx_hi = clamp(int(valueRange.upper*numCMIntervals),0,numCMIntervals);
        
        for (int i=idx_lo;i<=idx_hi;i++) {
          majorant = max(majorant,xfValues[i].w);
        }
      }
    }
    
    proxies[tid].majorant = majorant;
  }



  
  void NextDomainKernel::mapXF(const vec4f *xfValues,
                               int xfSize,
                               range1f xfDomain)
  {
    int bs = 128;
    int nb = divRoundUp(numProxies,bs);
    updateProxies<<<nb,bs>>>(xfValues,xfSize,xfDomain,
                             (Proxy*)owlBufferGetPointer(proxiesBuffer,0),
                             (range1f*)owlBufferGetPointer(valueRangesBuffer,0),
                             numProxies);
    CUDA_SYNC_CHECK();
    owlGroupRefitAccel(blas);
    owlGroupRefitAccel(tlas);
  }
    
  
}

