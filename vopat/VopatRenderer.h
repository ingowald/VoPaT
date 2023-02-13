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

#include "vopat/LaunchParams.h"
#include "volume/Volume.h"
#include "surface/MeshGeom.h"
#include "model/Model.h"
#include "miniScene/Scene.h"

#ifndef VOPAT_MAX_BOUNCES
/*! bounces==0 means primary rays only, no shadow ray; bouncs==1 means
  primray ray, one shadow ray, and one bounce ray, etc */

# define VOPAT_MAX_BOUNCES 1
#endif

namespace vopat {

  typedef ForwardingLayer::DD ForwardGlobals;
  
  struct VopatRenderer
  {
    typedef std::shared_ptr<VopatRenderer> SP;

    static SP create(CommBackend *comm,
                     Volume::SP volume,
                     mini::Scene::SP replicatedGeom)
    { return std::make_shared<VopatRenderer>(comm,volume,replicatedGeom); }
    
    VopatRenderer(CommBackend *comm,
                  Volume::SP volume,
                  mini::Scene::SP replicatedGeom
                  //,
                  // Model::SP model,
                  // const std::string &baseFileName
                  );

    bool isMaster() const { return comm->isMaster; }
    
    void generatePrimaryWave();
    void traceLocally();
    void resizeFrameBuffer(const vec2i &newSize);
    
    void screenShot() { fbLayer.screenShot(); }
    void resetAccumulation() { accumID = 0; fbLayer.resetAccumulation(); }

    void setTransferFunction(const std::vector<vec4f> &cm,
                             const interval<float> &range,
                             const float density);

    void setLights(float ambient,
                   const std::vector<DirectionalLight> &dirLights) 
    {
      volume.setLights(ambient,dirLights);
      PING;
    }

    int myRank() const { return comm->islandRank(); }

    /*! builds OWL accel structure(s) for all replicated geometry, and
        sets accel to launch params */
    void buildReplicatedGeometry();
      
    void setCamera(const vec3f &from,
                   const vec3f &at,
                   const vec3f &up,
                   const float fovy)
    {
      camera.from = from;
      camera.at = at;
      camera.up = up;
      camera.fovy = fovy;
      camera.dd = Camera(fbLayer.fullFbSize,from,at,up,fovy);
      // PING;
    }
    void createNextDomainKernel();
    /*! render frame to given frame buffer pointer. fbPointer will be
        null on the workers, and on the master has to be preallocated
        to hold all pixels that this renderer has last been resized
        to */
    void renderFrame(uint32_t *fbPointer);

    CommBackend *comm;
    Volume::SP volume;
    mini::Scene::SP replicatedGeom;

    OWLContext owl;
    OWLModule  owlDevCode;

    OWLLaunchParams lp = 0;
    OWLRayGen traceLocallyRG;
    OWLRayGen generatePrimaryWaveRG;

    //    OWLGroup  nextDomainGroup;
    NextDomainKernel nextDomainKernel;

    struct {
      vec3f from{0.f,0.f,0.f}, at{0.f,0.f,1.f}, up{0.f,1.f,0.f};
      float fovy = 30.f;
      Camera dd;
    } camera;
    ForwardingLayer  forwardingLayer;
    AddLocalFBsLayer fbLayer;
    MCGrid           mcGrid;
    vec2i islandFbSize;
    int accumID;
  };

  
}


