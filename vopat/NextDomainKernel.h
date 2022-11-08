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

  struct VopatRenderer;

  using range1f = interval<float>;
  
  
  struct NextDomainKernel {
    enum { MAX_RANKS = 64 };
    
    // enum { RETRY = 1<<30 };
    enum { Phase_FindFirst=0, Phase_FindSelf, Phase_FindOthers, Phase_FindNext };
          
    struct Proxy {
      box3f domain;
      int   rankID;
      float majorant;
    };

    struct PRD {
      struct {
        inline __device__ void clearBits()
        { for (int i=0;i<(MAX_RANKS+15)/16;i++) words[i] = 0; }
        
        inline __device__ void setBit(int rank)
        { words[rank/16] |= (1<<(rank%16)); }
        
        inline __device__ bool hasBitSet(int rank)
        { return words[rank/16] & (1<<(rank%16)); }
        
        uint16_t   words[(MAX_RANKS+15)/16];
      } alreadyTravedMask;
      
      float      closestDist;
      struct {
        uint32_t   dbg:1;
        uint32_t   phase:4;
        int32_t    closestRank:20;
      };
    };
    
    struct LPData {
      inline __device__ int computeFirstRank(Ray &path) const;
      inline __device__ int computeNextRank(Ray &path) const;
      
      int                    myRank;
      Proxy                 *proxies;
      OptixTraversableHandle proxyBVH;
    };

    struct Geom {
      Proxy                 *proxies;
    };
    
    void create(VopatRenderer *);
    
    void addLPVars(std::vector<OWLVarDecl> &lpVars,
                   // offset of this kernel's vars within LP
                   uint32_t kernelOffset);
    void setLPVars(OWLLaunchParams lp);

    /*! exhanges volumeProxies across all range, build *all* ranks' proxies
        and value ranges, and upload to the proxiesBuffer and
        valueRangesBuffer */
    void createProxies(VopatRenderer *vopat);

    void mapXF(const vec4f *xfValues,
               int xfSize,
               range1f xfDomain);
    
    /*! total number of proxies gathered across all ranks - NOT only the active ones */
    int numProxies = -1;
    
    /*! for each logical volumeProxy created across all ranks, this stores
        the corresponding proxy */
    OWLBuffer   proxiesBuffer;
    
    /*! for each logical volumeProxy created across all ranks, this stores
        the (pre-transfer function) value range for this volumeProxy (so the
        proxy's majorant can be recomputed if the xf changes) */
    OWLBuffer   valueRangesBuffer;
    
    OWLGeomType gt;
    OWLGeom     geom;
    OWLGroup    blas, tlas;
    int         myRank;
  };

} // ::vopat
