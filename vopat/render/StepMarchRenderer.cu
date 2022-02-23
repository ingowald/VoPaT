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

#include "LocalDeviceRenderer.h"

namespace vopat {

  /*! a version of the data parallel ray forwarding renderer that uses
    fixed step size ray marching throught he volume */
  struct StepMarchKernels : public DeviceKernelsBase
  {
    static inline __device__
    void traceRay(int tid,
                  const VopatGlobals &vopat,
                  const OwnGlobals &globals);
  };


  inline __device__
  void StepMarchKernels::traceRay(int tid,
                                  const VopatGlobals &vopat,
                                  const OwnGlobals &globals)
  {
    Ray ray = vopat.rayQueueIn[tid];

    vec3f throughput = from_half(ray.throughput);
    vec3f org = ray.origin;
    vec3f dir = ray.getDirection();
    
    const box3f myBox = globals.rankBoxes[vopat.myRank];
    float t0 = 0.f, t1 = CUDART_INF;
    boxTest(myBox,ray,t0,t1);

      
    // Random rnd((int)ray.pixelID+vopat.myRank+ray.age,vopat.sampleID);
    Random rnd((int)ray.pixelID,vopat.sampleID+vopat.myRank*0x123456);

    // maximum possible voxel density
    const float dt = 1.f; // relative to voxels
    const float DENSITY = vopat.xf.density;//.03f;
    float t = t0 + dt * rnd();
    bool killThisRay = false;
    while (true) {
      if (t >= t1) break;

      // Update current position
      vec3f P = org + t * dir;
      float f;
      if (!getVolume(f,globals,P)) { t += dt; continue; }
      vec4f xf = transferFunction(vopat,f);
      f = xf.w;
      f *= (DENSITY * dt);
      if (rnd() >= f) {
        t += dt;
        continue;
      }

      if (ray.isShadow) {
        killThisRay = true;
        // vopat.killRay(tid);
        break;//return;
      } else {
        org = P; 
        ray.origin = org;
        ray.setDirection(lightDirection());
        dir = ray.getDirection();
        
        t0 = 0.f;
        t1 = CUDART_INF;
        boxTest(myBox,ray,t0,t1,ray.dbg);
        t = dt * rnd();
        ray.isShadow = true;
        throughput *= vec3f(xf);
        ray.throughput = to_half(throughput);
        continue;
      }
    }
    int nextNode
      = killThisRay
      ? -1
      : computeNextNode(vopat,globals,ray,t1,ray.dbg);

    if (nextNode == -1) {
      vec3f color
        = (ray.isShadow)
        /* shadow ray that did reach the light (shadow rays that got
           blocked got terminated above) */
        ? lightColor() // * albedo()
        /* primary ray going straight through */
        : backgroundColor(ray,vopat);

      if (killThisRay)
        color *= ambient();
      color *= throughput;
      
      if (ray.crosshair) color = vec3f(1.f)-color;
      vopat.addPixelContribution(ray.pixelID,color);
      vopat.killRay(tid);
    } else {
      // ray has another node to go to - add to queue
      ray.throughput = to_half(throughput);
      vopat.forwardRay(tid,ray,nextNode);
    }
  }

  
  Renderer *createRenderer_StepMarch(CommBackend *comm,
                                    Model::SP model,
                                    const std::string &fileNameBase,
                                    int rank)
  {
    LocalDeviceRenderer<StepMarchKernels> *nodeRenderer
      = new LocalDeviceRenderer<StepMarchKernels>
      (model,fileNameBase,rank);
    return new RayForwardingRenderer<StepMarchKernels::Ray>(comm,nodeRenderer);
  }

} // ::vopat
