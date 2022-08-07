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

#include "VopatBase.h"
#include "DeviceKernels.h"

namespace vopat
{
  
  inline __device__
  void generatePrimaryWaveKernel(const vec2i launchIdx,
                                 const ForwardGlobals &vopat,
                                 const VolumeGlobals &globals)
  {
    int ix = launchIdx.x;
    int iy = launchIdx.y;

    if (vec2i(ix,iy) == vec2i(0))
      printf("launch 0 : islandsize %i %i \n",
             vopat.islandFbSize.x,vopat.islandFbSize.y);
             
    if (ix >= vopat.islandFbSize.x) return;
    if (iy >= vopat.islandFbSize.y) return;

    bool dbg = vec2i(ix,iy) == vopat.islandFbSize/2;
    if (dbg) printf("=======================================================\ngeneratePrimaryWaveKernel %i %i\n",ix,iy);
    
    int myRank = vopat.islandRank;//myRank;
    int world_iy
      = vopat.islandIndex
      + iy * vopat.islandCount;
    Ray ray    = DeviceKernels::generateRay(vopat,vec2i(ix,world_iy),vec2f(.5f));
#if 0
    ray.dbg    = (vec2i(ix,world_iy) == vopat.worldFbSize/2);
    if (ray.dbg) printf("----------- NEW RAY -----------\n");
#else
    ray.dbg    = false;
#endif

    ray.numBounces = 0;
#if DEBUG_FORWARDS
    ray.numFwds = 0;
#endif
    ray.crosshair = (ix == vopat.worldFbSize.x/2) || (world_iy == vopat.worldFbSize.y/2);
    int dest   = DeviceKernels::computeInitialRank(globals,ray);

    if (dest < 0) {
      /* "nobody" owns this pixel, set to background on rank 0 */
      if (myRank == 0) {
        // vopat.accumBuffer[islandPixelID(vopat,ray.pixelID)] += DeviceKernels::backgroundColor(ray,vopat);
        vopat.addPixelContribution(ray.pixelID,DeviceKernels::backgroundColor(ray,vopat));
      }
      return;
    }
    if (dest != myRank) {
      /* somebody else owns this pixel; we don't do anything */
      return;
    }
    int queuePos = atomicAdd(&vopat.perRankSendCounts[myRank],1);
    
    if (queuePos >= vopat.islandFbSize.x*vopat.islandFbSize.y)
      printf("FISHY PRIMARY RAY POS!\n");
    
    vopat.rayQueueIn[queuePos] = ray;
    if (!checkOrigin(ray))
      printf("fishy primary ray!\n");
  }
  
}
