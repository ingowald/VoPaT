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

#include "NodeRenderer.h"

namespace vopat {

  VopatNodeRenderer::VopatNodeRenderer(Model::SP model,
                                       const std::string &baseFileName,
                                       int islandRank,
                                       int gpuID)
    : volume(model,baseFileName,islandRank,gpuID)
  {
    auto &owl = volume.owl;
    auto &owlDevCode = volume.owlDevCode;
    if (islandRank >= 0) {
      traceLocallyRG = owlRayGenCreate(owl,owlDevCode,"traceLocallyRG",0,0,0);
      generatePrimaryWaveRG = owlRayGenCreate(owl,owlDevCode,"generatePrimaryWaveRG",0,0,0);
      OWLVarDecl lpArgs[]
        = {
           {"forwardGlobals",OWL_USER_TYPE(ForwardGlobals),OWL_OFFSETOF(LaunchParams,forwardGlobals)},
           {"volumeGlobals",OWL_USER_TYPE(VolumeGlobals),OWL_OFFSETOF(LaunchParams,volumeGlobals)},
           {"surfaceGlobals",OWL_USER_TYPE(SurfaceGlobals),OWL_OFFSETOF(LaunchParams,surfaceGlobals)},
           // {"umeshSampleBVH",OWL_GROUP,OWL_OFFSETOF(LaunchParams,umeshSampleBVH)},
           {nullptr}
      };
      lp = owlParamsCreate(owl,sizeof(LaunchParams),
                           lpArgs,-1);
      owlBuildPrograms(owl);
      owlBuildPipeline(owl);
      owlBuildSBT(owl);
    }
                           
    // Reuse for ISOs
    surface.globals.gradientDelta = volume.globals.gradientDelta;
#if VOPAT_UMESH
    surface.globals.umesh        = volume.globals.umesh;
#else
    surface.globals.volume        = volume.globals.volume;
#endif
    surface.globals.islandRank    = volume.globals.islandRank;
    surface.globals.rankBoxes     = volume.globals.rankBoxes;
    // surface.globals.numRanks      = volume.globals.numRanks;
    surface.globals.myRegion      = volume.globals.myRegion;

    std::vector<int> hIsoActive({0,0,0,0});
    std::vector<float> hIsoValues({0.f,0.f,0.f,0.f});
    std::vector<vec3f> hIsoColors({{.8f,.8f,.8f},{.8f,.8f,.8f},{.8f,.8f,.8f},{.8f,.8f,.8f}});
    surface.isoActive.upload(hIsoActive);
    surface.isoValues.upload(hIsoValues);
    surface.isoColors.upload(hIsoColors);

    surface.globals.iso.numActive = 0;
    surface.globals.iso.active    = surface.isoActive.get();
    surface.globals.iso.values    = surface.isoValues.get();
    surface.globals.iso.colors    = surface.isoColors.get();
  }

  // __global__
  // void doTraceRaysLocally(ForwardGlobals forward,
  //                         VolumeGlobals  volume,
  //                         SurfaceGlobals surf)
  // {
  //   int tid = threadIdx.x+blockIdx.x*blockDim.x;
  //   if (tid >= forward.numRaysInQueue) return;

  //   traceRaysKernel(tid,forward,volume,surf);
  // }
  
  void VopatNodeRenderer::traceLocally
  (const ForwardGlobals &forward,
   bool fishy)
  {
    PING;
    PRINT(forward.numRaysInQueue);
    if (forward.numRaysInQueue == 0) return;
#if VOPAT_UMESH_OPTIX
    printf(" -> tracing numRaysInQueue %i\n",forward.numRaysInQueue);
    owlParamsSetRaw(lp,"forwardGlobals",&forward);
    owlParamsSetRaw(lp,"volumeGlobals",&volume.globals);
    owlParamsSetRaw(lp,"surfaceGlobals",&surface.globals);
    // owlParamsSetGroup(lp,"umeshSampleBVH",volume.umeshAccel);
    owlLaunch2D(traceLocallyRG,forward.numRaysInQueue,1,lp);
#else
    // CUDA_SYNC_CHECK();
    int blockSize = 64;
    int numBlocks = divRoundUp(forward.numRaysInQueue,blockSize);
    if (fishy) printf(" -> tracing numRaysInQueue %i\n",forward.numRaysInQueue);
    if (numBlocks)
      doTraceRaysLocally<<<numBlocks,blockSize>>>
        (forward,volume.globals,surface.globals);
    // CUDA_SYNC_CHECK();
#endif
  }

  // __global__
  // void doGeneratePrimaryWave(ForwardGlobals forward,
  //                            VolumeGlobals globals)
  // {
  //   int ix = threadIdx.x + blockIdx.x*blockDim.x;
  //   int iy = threadIdx.y + blockIdx.y*blockDim.y;
  //   generatePrimaryWaveKernel(vec2i(ix,iy),forward,globals);
  // }
  
  void VopatNodeRenderer::generatePrimaryWave
  (const ForwardGlobals &forwardGlobals)
  {
    CUDA_SYNC_CHECK();
    PING;
    PRINT(forwardGlobals.islandFbSize);
    
#if VOPAT_UMESH_OPTIX
    if (forwardGlobals.islandFbSize.y <= 0)
      return;
    
    auto volumeGlobals = volume.globals;
    PRINT(forwardGlobals.islandFbSize);
    owlParamsSetRaw(lp,"forwardGlobals",&forwardGlobals);
    owlParamsSetRaw(lp,"volumeGlobals",&volumeGlobals);

    std::cout << "##################################################################" << std::endl;
    fflush(0);
    owlLaunch2D(generatePrimaryWaveRG,
                forwardGlobals.islandFbSize.x,
                forwardGlobals.islandFbSize.y,
                lp);
    owlLaunchSync(lp);
    std::cout << "##################################################################" << std::endl;
    fflush(0);
#else
    vec2i blockSize(16);
    vec2i numBlocks = divRoundUp(vopat.islandFbSize,blockSize);
    doGeneratePrimaryWave<<<numBlocks,blockSize>>>(vopat,volume.globals);
#endif
    CUDA_SYNC_CHECK();
  }

} // ::vopat
