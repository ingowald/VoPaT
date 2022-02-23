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
#include "DDA.h"

namespace vopat {

  /*! a version of the data parallel ray forwarding renderer that uses
    DDA to step through all cells of the local volume, sampling each one */
  struct CellMarchKernels : public DeviceKernelsBase
  {
    static inline __device__
    void traceRay(int tid,
                  const VopatGlobals &vopat,
                  const OwnGlobals &globals);
  };

  inline __device__
  void CellMarchKernels::traceRay(int tid,
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

    Random rnd((int)ray.pixelID,vopat.sampleID+vopat.myRank*0x123456);
    vec3i numVoxels = globals.numVoxels;
    vec3i numCells  = numVoxels - 1;
    const float DENSITY = vopat.xf.density;//.03f;
    bool skipFirstStep = false;
    if (!ray.isShadow)
      dda::dda3(org - myBox.lower,dir,CUDART_INF,
                vec3ui(numCells),
                [&](vec3i cellIdx)
                {
                  // Update current position - for now (in absence of
                  // tin/tout of the cell - let's just do sample in
                  // center of cell
                  vec3f P = myBox.lower + vec3f(cellIdx)+0.5f;
                  float f;
                  if (!getVolume(f,globals,P))
                    // something fishy with this sample pos - keep on going
                    return true;
                  vec4f xf = transferFunction(vopat,f);
                  f = xf.w;
                  f *= (DENSITY);
                  if (rnd() >= f) 
                    // did not sample this density; keep on going
                    return true;

                  // this cell WAS sampled - turn this into a shadow ray
                  org = P; 
                  ray.origin = org;
                  ray.setDirection(lightDirection());
                  dir = ray.getDirection();
                  
                  t0 = 0.f;
                  t1 = CUDART_INF;
                  boxTest(myBox,ray,t0,t1,ray.dbg);
                  ray.isShadow = true;
                  skipFirstStep = true;
                  return false;
                },
                false);
    // note ray may also just have BECOME a shadow ray
    bool killThisRay = false;
    if (ray.isShadow)
      dda::dda3(org - myBox.lower,dir,CUDART_INF,
                vec3ui(numCells),
                [&](vec3i cellIdx)
                {
                  if (skipFirstStep) {
                    skipFirstStep = false;
                    return true;
                  }
                  // Update current position - for now (in absence of
                  // tin/tout of the cell - let's just do sample in
                  // center of cell
                  vec3f P = myBox.lower + vec3f(cellIdx)+0.5f;
                  float f;
                  if (!getVolume(f,globals,P))
                    // something fishy with this sample pos - keep on going
                    return true;
                  vec4f xf = transferFunction(vopat,f);
                  f = xf.w;
                  f *= (DENSITY);
                  if (rnd() >= f) 
                    // did not sample this density; keep on going
                    return true;

                  // kill ray and terminate traversal
                  killThisRay = true;
                  return false;
                },
                false);
    
    int nextNode
      = killThisRay
      ? -1
      : computeNextNode(vopat,globals,ray,t1,ray.dbg);
    
    if (nextNode == -1) {
      vec3f color
        = (ray.isShadow)
        /* shadow ray that did reach the light (shadow rays that got
           blocked got terminated above) */
        ? lightColor() //albedo()
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
      // ray.throughput = to_half(throughput);
      vopat.forwardRay(tid,ray,nextNode);
    }
  }

  
  Renderer *createRenderer_CellMarch(CommBackend *comm,
                                     Model::SP model,
                                     const std::string &fileNameBase,
                                     int rank)
  {
    LocalDeviceRenderer<CellMarchKernels> *nodeRenderer
      = new LocalDeviceRenderer<CellMarchKernels>
      (model,fileNameBase,rank);
    return new RayForwardingRenderer<CellMarchKernels::Ray>(comm,nodeRenderer);
  }

} // ::vopat
