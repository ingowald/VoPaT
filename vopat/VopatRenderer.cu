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

#include "vopat/VopatRenderer.h"

namespace vopat {

  extern "C" char deviceCode_ptx[];
  
  VopatRenderer::VopatRenderer(CommBackend *comm,
                               Volume::SP volume//,
                               // Model::SP model,
                               // const std::string &baseFileName
                               )
    : comm(comm),
      volume(volume),
      forwardingLayer(comm),
      fbLayer(comm)
  {
    CUDA_SYNC_CHECK();
    printf("#vopat(%i.%i): initializing OWL\n",
           comm->islandIndex(),comm->islandRank());
    
    if (comm->islandRank() >= 0) {
      owl = owlContextCreate(&comm->worker.gpuID,1);
      owlDevCode = owlModuleCreate(owl,deviceCode_ptx);

      createNextDomainKernel();
      
      CUDA_SYNC_CHECK();
      traceLocallyRG = owlRayGenCreate(owl,owlDevCode,"traceLocallyRG",0,0,0);
      generatePrimaryWaveRG = owlRayGenCreate(owl,owlDevCode,"generatePrimaryWaveRG",0,0,0);
      
      std::vector<OWLVarDecl> lpVars;

      nextDomainKernel.addLPVars(lpVars,OWL_OFFSETOF(LaunchParams,nextDomainKernel));
      volume->addLPVars(lpVars);
      lpVars.push_back
        ({"forwardGlobals",OWL_USER_TYPE(ForwardGlobals),OWL_OFFSETOF(LaunchParams,forwardGlobals)});
      lpVars.push_back
        ({"fbLayer",OWL_USER_TYPE(AddLocalFBsLayer::DD),OWL_OFFSETOF(LaunchParams,fbLayer)});
      lpVars.push_back
        ({"camera",OWL_USER_TYPE(Camera),OWL_OFFSETOF(LaunchParams,camera)});
      lpVars.push_back
        ({"rank",OWL_INT,OWL_OFFSETOF(LaunchParams,rank)});
      // lpVars.push_back
      //   ({"volumeGlobals",OWL_USER_TYPE(VolumeGlobals),OWL_OFFSETOF(LaunchParams,volumeGlobals)});
      // lpVars.push_back
      //   ({"surfaceGlobals",OWL_USER_TYPE(SurfaceGlobals),OWL_OFFSETOF(LaunchParams,surfaceGlobals)});

      
      lp = owlParamsCreate(owl,sizeof(LaunchParams),
                           lpVars.data(),lpVars.size());
      owlParamsSet1i(lp,"rank",myRank());
      
      volume->build(owl,owlDevCode);
      volume->setDD(lp);

      CUDA_SYNC_CHECK();
      owlBuildPrograms(owl);
      CUDA_SYNC_CHECK();
      owlBuildPipeline(owl);
      CUDA_SYNC_CHECK();
      owlBuildSBT(owl);
      CUDA_SYNC_CHECK();
      
      nextDomainKernel.setLPVars(lp);
    }
                           
    // volume = std::make_shared<Volume>(model,baseFileName,islandRank,gpuID);
      
    CUDA_SYNC_CHECK();
    // Reuse for ISOs
//     surface.globals.gradientDelta = volume.globals.gradientDelta;
// #if VOPAT_UMESH
//     surface.globals.umesh        = volume.globals.umesh;
// #else
//     surface.globals.volume        = volume.globals.volume;
// #endif
//     surface.globals.islandRank    = volume.globals.islandRank;
//     surface.globals.rankBoxes     = volume.globals.rankBoxes;
//     // surface.globals.numRanks      = volume.globals.numRanks;
//     surface.globals.myRegion      = volume.globals.myRegion;

    // std::vector<int> hIsoActive({0,0,0,0});
    // std::vector<float> hIsoValues({0.f,0.f,0.f,0.f});
    // std::vector<vec3f> hIsoColors({{.8f,.8f,.8f},{.8f,.8f,.8f},{.8f,.8f,.8f},{.8f,.8f,.8f}});
    // PING;
    // surface.isoActive.upload(hIsoActive);
    // PING;
    // surface.isoValues.upload(hIsoValues);
    // PING;
    // surface.isoColors.upload(hIsoColors);
    // PING;

    // surface.globals.iso.numActive = 0;
    // surface.globals.iso.active    = surface.isoActive.get();
    // surface.globals.iso.values    = surface.isoValues.get();
    // surface.globals.iso.colors    = surface.isoColors.get();
    // PING;
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
  
  void VopatRenderer::traceLocally()
  {
    if (forwardingLayer.numRaysIn == 0)
      return;

    auto &forward = forwardingLayer.dd;
    owlParamsSetRaw(lp,"forwardGlobals",&forward);

    volume->setDD(lp);
    
// #if VOPAT_UMESH_OPTIX
    // printf(" -> tracing numRaysInQueue %i\n",forward.numRaysInQueue);
    // owlParamsSetRaw(lp,"volumeGlobals",&volume.globals);
    // owlParamsSetRaw(lp,"surfaceGlobals",&surface.globals);
    // owlParamsSetGroup(lp,"umeshSampleBVH",volume.umeshAccel);
    owlLaunch2D(traceLocallyRG,forward.numRaysIn,1,lp);
// #else
//     // CUDA_SYNC_CHECK();
//     int blockSize = 64;
//     int numBlocks = divRoundUp(forward.numRaysIn,blockSize);
//     if (fishy) printf(" -> tracing numRaysIn %i\n",forward.numRaysIn);
//     if (numBlocks)
//       doTraceRaysLocally<<<numBlocks,blockSize>>>
//         (forward,volume.globals,surface.globals);
//     // CUDA_SYNC_CHECK();
// #endif
  }

  // __global__
  // void doGeneratePrimaryWave(ForwardGlobals forward,
  //                            VolumeGlobals globals)
  // {
  //   int ix = threadIdx.x + blockIdx.x*blockDim.x;
  //   int iy = threadIdx.y + blockIdx.y*blockDim.y;
  //   generatePrimaryWaveKernel(vec2i(ix,iy),forward,globals);
  // }
  
  void VopatRenderer::generatePrimaryWave()
  {
    // auto &forwardGlobals = forwardingLayer.dd;
    CUDA_SYNC_CHECK();
    // PING;
    // PRINT(forwardGlobals.islandFbSize);
    
// #if VOPAT_UMESH_OPTIX
    auto &fbSize = fbLayer.islandFbSize;
    if (fbSize.y <= 0)
      return;
    
    auto &forward = forwardingLayer.dd;
    owlParamsSetRaw(lp,"forwardGlobals",&forward);
    AddLocalFBsLayer::DD &fbLayerDD = fbLayer.dd;
    owlParamsSetRaw(lp,"fbLayer",&fbLayerDD);
    owlParamsSetRaw(lp,"camera",&camera.dd);

    volume->setDD(lp);
    // auto volumeGlobals = volume.globals;
    // PRINT(forwardGlobals.islandFbSize);
    // owlParamsSetRaw(lp,"forwardGlobals",&forwardGlobals);
    // owlParamsSetRaw(lp,"volumeGlobals",&volumeGlobals);

    // std::cout << "##################################################################" << std::endl;
    // fflush(0);
    owlLaunch2D(generatePrimaryWaveRG,
                fbSize.x,
                fbSize.y,
                lp);
    owlLaunchSync(lp);
    // std::cout << "##################################################################" << std::endl;
    // fflush(0);
// #else
//     vec2i blockSize(16);
//     vec2i numBlocks = divRoundUp(vopat.islandFbSize,blockSize);
//     doGeneratePrimaryWave<<<numBlocks,blockSize>>>(vopat,volume.globals);
// #endif
    CUDA_SYNC_CHECK();
  }


  void VopatRenderer::createNextDomainKernel()
  {
    // NextDomainKernel &ndk = nextDomainKernel;
    // std::cout << "building proxies" << std::endl;
    // for (int rankID=0;rankID<volume.model->bricks.size();rankID++) {
    //   NextDomainKernel::Proxy proxy;
    //   proxy.rankID = rankID;
    //   proxy.majorant = 1e20f;
    //   proxy.domain = volume.model->bricks[rankID]->getDomain();
    //   ndk.proxies.push_back(proxy);
    // }

    CUDA_SYNC_CHECK();
    nextDomainKernel.create(this);
    CUDA_SYNC_CHECK();
  }


  void VopatRenderer::resizeFrameBuffer(const vec2i &newSize)
  {
    printf("#(%i.%i) resize(%i %i)\n",
           comm->islandIndex(),comm->islandRank(),
           newSize.x,newSize.y);
    
    fbLayer.resize(newSize);
    camera.dd = Camera(fbLayer.fullFbSize,
                       camera.from,
                       camera.at,
                       camera.up,
                       camera.fovy);

    if (isMaster()) {
      islandFbSize = -1;
    } else {
      islandFbSize = fbLayer.islandFbSize;
      int maxRaysPerPixel = 1+2*VOPAT_MAX_BOUNCES;
      forwardingLayer.resizeQueues(islandFbSize.x*islandFbSize.y*maxRaysPerPixel);
    }
  }


  void VopatRenderer::setTransferFunction(const std::vector<vec4f> &cm,
                                          const interval<float> &range,
                                          const float density)
  {
    // printf("(%i.%i) setting transfer function num %i range %f %f\n",
    //        comm->islandIndex(),
    //        comm->islandRank(),
    //        int(cm.size()),range.lower,range.upper);

    if (isMaster()) return;
    
    volume->setTransferFunction(cm,range,density);
    nextDomainKernel.mapXF(volume->xf.colorMap.get(),volume->xf.colorMap.N,
                           volume->xf.domain);
    nextDomainKernel.setLPVars(lp);
    // printf("todo - update macro cells; todo - update shards/proxies\n"); 
  }
  

  void VopatRenderer::renderFrame(uint32_t *fbPointer)
  {
    resetAccumulation();
    generatePrimaryWave();
    fbLayer.addLocalFBs(fbPointer);
  }

} // ::vopat
