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
#include "model/Model.h"

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
                     Volume::SP volume)
    { return std::make_shared<VopatRenderer>(comm,volume); }
    
    VopatRenderer(CommBackend *comm,
                  Volume::SP volume//,
                  // Model::SP model,
                  // const std::string &baseFileName
                  );

    bool isMaster() const { return comm->isMaster; }
    
    void generatePrimaryWave();
    void traceLocally();
    void resizeFrameBuffer(const vec2i &newSize);
    
    void screenShot() { addLocalFBsLayer.screenShot(); }
    void resetAccumulation() { addLocalFBsLayer.resetAccumulation(); }
    // void generatePrimaryWave(const ForwardGlobals &forward);
    // void traceLocally(const ForwardGlobals &forward, bool fishy);
    
    // static inline __device__
    // int computeInitialRank(const VolumeGlobals &vopat,
    //                        Ray ray, bool dbg = false);
    
    // static inline __device__
    // int computeNextNode(const ForwardGlobals &vopat,
    //                     const Ray &ray,
    //                     const float t_already_travelled,
    //                     bool dbg = false);

    void setTransferFunction(const std::vector<vec4f> &cm,
                             const interval<float> &range,
                             const float density);

    // void setISO(int numActive,
    //             const std::vector<int> &active,
    //             const std::vector<float> &value,
    //             const std::vector<vec3f> &colors)
    // { surface.setISO(numActive,active,value,colors); }

    void setLights(float ambient,
                   const std::vector<DirectionalLight> &dirLights) 
    {// volume.setLights(ambient,dirLights);
      PING;
    }

    void setCamera(const vec3f &from,
                   const vec3f &at,
                   const vec3f &up,
                   const float fovy)
    {
      PING;
    }
    void createNextDomainKernel();
    /*! render frame to given frame buffer pointer. fbPointer will be
        null on the workers, and on the master has to be preallocated
        to hold all pixels that this renderer has last been resized
        to */
    void renderFrame(uint32_t *fbPointer);
    
    CommBackend *comm;
    Volume::SP volume;
    // SurfaceIntersector surface;

    OWLContext owl;
    OWLModule  owlDevCode;

    OWLLaunchParams lp;
    OWLRayGen traceLocallyRG;
    OWLRayGen generatePrimaryWaveRG;

    //    OWLGroup  nextDomainGroup;
    NextDomainKernel nextDomainKernel;
    
    ForwardingLayer  forwardingLayer;
    AddLocalFBsLayer addLocalFBsLayer;
    vec2i islandFbSize;
  };

  
}


