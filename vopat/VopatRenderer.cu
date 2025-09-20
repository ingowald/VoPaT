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
                               Volume::SP volume,
                               mini::Scene::SP replicatedGeom//,
                               // Model::SP model,
                               // const std::string &baseFileName
                               )
    : comm(comm),
      volume(volume),
      replicatedGeom(replicatedGeom),
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
        ({"forwardGlobals",OWL_USER_TYPE(ForwardGlobals),
          OWL_OFFSETOF(LaunchParams,forwardGlobals)});
      lpVars.push_back
        ({"fbLayer",OWL_USER_TYPE(AddLocalFBsLayer::DD),
          OWL_OFFSETOF(LaunchParams,fbLayer)});
      lpVars.push_back
        ({"camera",OWL_USER_TYPE(Camera),
          OWL_OFFSETOF(LaunchParams,camera)});
      lpVars.push_back
        ({"rank",OWL_INT,
          OWL_OFFSETOF(LaunchParams,rank)});
      lpVars.push_back
        ({"sampleID",OWL_INT,
          OWL_OFFSETOF(LaunchParams,sampleID)});
      lpVars.push_back
        ({"emergency",OWL_INT,
          OWL_OFFSETOF(LaunchParams,emergency)});
      lpVars.push_back
        ({"mcGrid",OWL_USER_TYPE(MCGrid::DD),
          OWL_OFFSETOF(LaunchParams,mcGrid)});
      lpVars.push_back
        ({"volumeSampler.xf",OWL_USER_TYPE(Volume::DD),
          OWL_OFFSETOF(LaunchParams,volumeSampler.xf)});
      lpVars.push_back
        ({"volumeSampler.type",OWL_INT,
          OWL_OFFSETOF(LaunchParams,volumeSampler.type)});
      lpVars.push_back
        ({"replicatedSurfaceBVH",OWL_GROUP,
          OWL_OFFSETOF(LaunchParams,replicatedSurfaceBVH)});
      
      lp = owlParamsCreate(owl,sizeof(LaunchParams),
                           lpVars.data(),lpVars.size());
      owlParamsSet1i(lp,"rank",myRank());
      
      volume->build(owl,owlDevCode);
      volume->setDD(lp);

      volume->buildMCs(mcGrid);
      owlParamsSetRaw(lp,"mcGrid",&mcGrid.dd);

      buildReplicatedGeometry();
      
      CUDA_SYNC_CHECK();
      owlBuildPrograms(owl);
      CUDA_SYNC_CHECK();
      owlBuildPipeline(owl);
      CUDA_SYNC_CHECK();
      owlBuildSBT(owl);
      CUDA_SYNC_CHECK();
      
      nextDomainKernel.setLPVars(lp);
    }
    CUDA_SYNC_CHECK();
  }

  /*! builds OWL accel structure(s) for all replicated geometry, and
    sets accel to launch params */
  void VopatRenderer::buildReplicatedGeometry()
  {
    if (!replicatedGeom) return;

    MeshGeom::defineGeometryType(owl,owlDevCode);
    
    auto scene = replicatedGeom;
    std::map<mini::Object::SP,OWLGroup> objectGroups;
    for (auto inst : scene->instances)
      objectGroups[inst->object] = 0;
    int meshID = 0;
    for (auto &og : objectGroups) {
      auto obj = og.first;
      std::vector<OWLGeom> geoms;

      for (auto mesh : obj->meshes) {
        OWLGeom geom = MeshGeom::createGeom(owl,mesh);
        vec3f color = owl::common::randomColor(meshID++);
        owlGeomSet3f(geom,"diffuseColor",color.x,color.y,color.z);
        geoms.push_back(geom);
      }
      
      OWLGroup group = owlTrianglesGeomGroupCreate(owl,geoms.size(),geoms.data());
      owlGroupBuildAccel(group);
      og.second = group;
    }

    std::vector<affine3f> transforms;
    std::vector<OWLGroup> groups;
    for (auto inst : scene->instances) {
      transforms.push_back(inst->xfm);
      groups.push_back(objectGroups[inst->object]);
    }
    OWLGroup world = owlInstanceGroupCreate(owl,groups.size(),
                                            groups.data(),0,
                                            (const float *)transforms.data());
    owlGroupBuildAccel(world);

    owlParamsSetGroup(lp,"replicatedSurfaceBVH",world);
  }

  void VopatRenderer::traceLocally()
  {
    if (forwardingLayer.numRaysIn == 0)
      return;

#if VOPAT_USE_RAFI
    auto forward = forwardingLayer.rafi->getDeviceInterface();
#else
    auto &forward = forwardingLayer.dd;
#endif
    owlParamsSetRaw(lp,"forwardGlobals",&forward);

    volume->setDD(lp);

#if VOPAT_USE_RAFI
    owlLaunch2D(traceLocallyRG,forwardingLayer.numRaysIn,1,lp);
#else
    owlLaunch2D(traceLocallyRG,forward.numRaysIn,1,lp);
#endif
    CUDA_SYNC_CHECK();
  }

  void VopatRenderer::generatePrimaryWave()
  {
    CUDA_SYNC_CHECK();
    auto &fbSize = fbLayer.islandFbSize;
    if (fbSize.y <= 0)
      return;

    int thisFrameID = accumID++;
    owlParamsSet1i(lp,"sampleID",thisFrameID);
    
#if VOPAT_USE_RAFI
    auto forward = forwardingLayer.rafi->getDeviceInterface();
#else
    auto &forward = forwardingLayer.dd;
#endif
    owlParamsSetRaw(lp,"forwardGlobals",&forward);
    AddLocalFBsLayer::DD &fbLayerDD = fbLayer.dd;
    owlParamsSetRaw(lp,"fbLayer",&fbLayerDD);
    owlParamsSetRaw(lp,"camera",&camera.dd);
    volume->setDD(lp);

    forwardingLayer.clearQueue();
    CUDA_SYNC_CHECK();
    owlLaunch2D(generatePrimaryWaveRG,
                fbSize.x,
                fbSize.y,
                lp);
    owlLaunchSync(lp);
    CUDA_SYNC_CHECK();
  }


  void VopatRenderer::createNextDomainKernel()
  {
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
    resetAccumulation();

    if (isMaster()) return;

    volume->setTransferFunction(cm,range,density);
    nextDomainKernel.mapXF(volume->xf.colorMap.get(),volume->xf.colorMap.N,
                           volume->xf.domain);
    nextDomainKernel.setLPVars(lp);
    owlParamsSetRaw(lp,"volumeSampler.xf",&volume->xfGlobals);
    mcGrid.mapXF(volume->xf.colorMap.get(),volume->xf.colorMap.N,
                 volume->xf.domain);
    
    // printf("todo - update macro cells; todo - update shards/proxies\n"); 
  }
  

  void VopatRenderer::renderFrame(uint32_t *fbPointer)
  {
    // for debugging:
    // resetAccumulation();
    
    if (!isMaster()) {
      PING;
      owlParamsSet1i(lp,"emergency",0);
      PING;
      forwardingLayer.clearQueue();
      PING;
      generatePrimaryWave();
      PING;
      
      int numExchanged;
      int numIts = 0;
      PING;
      while (1) {
        PING; PRINT(numIts);
        numExchanged = forwardingLayer.exchangeRays();
        PRINT(numExchanged);
        if (numExchanged == 0) {
          PING;
          break;
        }
        if (++numIts > 20) {
          if (myRank() == 0)
            std::cout << "fishy num forwards - done " << numIts << " rounds of forwarding already, now doing another with " << numExchanged << "rays ..." << std::endl;

          owlParamsSet1i(lp,"emergency",numIts);
          comm->worker.withinIsland->barrier();
          usleep(20);
          if (myRank() == 0)
            std::cout << "==================================================================" << std::endl; fflush(0);
          usleep(20);
          
          if (numIts > 30)
            {
            // break;
              comm->worker.withinIsland->barrier();
              sleep(1);
              comm->worker.withinIsland->barrier();
              exit(1);
            }
        }
#if 0
        std::cout << "==================================================================" << std::endl; fflush(0);
        comm->worker.withinIsland->barrier();
        usleep(50);
        // sleep(1);
#endif
        forwardingLayer.clearQueue();
        traceLocally();
      }
    }
    fbLayer.addLocalFBs(fbPointer);
    comm->barrierAll();
  }

} // ::vopat
