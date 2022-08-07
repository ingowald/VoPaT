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

#include "genPrimaryWave.h"
#include "traceRays.h"

namespace vopat {
  
  struct VopatNodeRenderer
    : public RayForwardingRenderer::NodeRenderer
    // ,
    //   public VolumeRenderer,
    //   public SurfaceIntersector
  {
    using inherited    = typename RayForwardingRenderer::NodeRenderer;
    
    VopatNodeRenderer(Model::SP model,
                      const std::string &baseFileName,
                      int islandRank,
                      int gpuID);
    void generatePrimaryWave(const ForwardGlobals &forward) override;
    void traceLocally(const ForwardGlobals &forward, bool fishy) override;
    
    static inline __device__
    int computeInitialRank(const VolumeGlobals &vopat,
                           Ray ray, bool dbg = false);
    
    static inline __device__
    int computeNextNode(const ForwardGlobals &vopat,
                        const Ray &ray,
                        const float t_already_travelled,
                        bool dbg = false);

    void setTransferFunction(const std::vector<vec4f> &cm,
                             const interval<float> &range,
                             const float density) override
    { volume.setTransferFunction(cm,range,density); }

    void setISO(int numActive,
                const std::vector<int> &active,
                const std::vector<float> &value,
                const std::vector<vec3f> &colors)
    { surface.setISO(numActive,active,value,colors); }

    void setLights(float ambient,
                   const std::vector<MPIRenderer::DirectionalLight> &dirLights) override
    { volume.setLights(ambient,dirLights); }

    VolumeRenderer volume;
    SurfaceIntersector surface;

      OWLLaunchParams lp;
    OWLRayGen traceLocallyRG;
    OWLRayGen generatePrimaryWaveRG;
  };

  
}


