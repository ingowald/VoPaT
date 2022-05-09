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

/*! this file implements a local device renderer thta is
    "intentionally" wrong, in the sense that it computes the same type
    of image you would getif oyu used woodcock _locally_, but relied
    on compositing rather than ray forwarding */
#include "vopat/render/VopatBase.h"
#include "DDA.h"

namespace vopat {

  struct WrongShadowsKernels : public Vopat
  {
    static inline __device__
    void traceRay(int tid,
                  const typename Vopat::ForwardGlobals &vopat,
                  const typename Vopat::VolumeGlobals  &dvr,
                  const typename Vopat::SurfaceGlobals &surf)
    {
      Ray ray = vopat.rayQueueIn[tid];

      vec3f throughput = from_half(ray.throughput);
      vec3f org = ray.origin;
      vec3f dir = ray.getDirection();
    
      const box3f myBox = dvr.rankBoxes[vopat.islandRank];
      float t0 = 0.f, t1 = CUDART_INF;
      boxTest(myBox,ray,t0,t1);

      Random rnd((int)ray.pixelID,vopat.sampleID+vopat.islandRank*0x123456);
#if VOPAT_UMESH
      vec3i numVoxels = 1024;
#else
      vec3i numVoxels = dvr.mc.dims;
#endif
      vec3i numCells  = numVoxels - 1;
      vec3i singleCell = vec3i(1); // just testing...

      // maximum possible voxel density
      const float dt = 1.f; // relative to voxels
      // const float DENSITY = .03f / ((vopat.xf.density == 0.f) ? 1.f : vopat.xf.density);//.03f;
      const float DENSITY = ((dvr.xf.density == 0.f) ? 1.f : dvr.xf.density);//.03f;
      float majorant = 1.f; // must be larger than the max voxel density
      float t = t0;
      while (true) {
        // Sample a distance
        t = t - (log(1.0f - rnd()) / (majorant*DENSITY)) * dt; 

        // A boundary has been hit
        if (t >= t1) {
          break;
        }

        // Update current position
        vec3f P = org + t * dir;

        // Sample heterogeneous media
        float f;
        if (!dvr.getVolume(f,P)) { continue; }
        vec4f xf = dvr.transferFunction(f);
        f = xf.w;
      
        // Check if a collision occurred (real particles / real + fake particles)
        if (rnd() >= f / majorant)
          continue;
        
        if (ray.isShadow) {
          // vec3f color = throughput * dvr.ambient();
          // if (ray.crosshair) color = vec3f(1.f)-color;
          // vopat.addPixelContribution(ray.pixelID,color);
          vopat.killRay(tid);            
          return;
        }
        
        org = P; 
        ray.origin = org;
        throughput *= vec3f(xf.x,xf.y,xf.z);
        {
          // add ambient illumination 
          vec3f color = throughput * dvr.ambient();
          if (ray.crosshair) color = vec3f(1.f)-color;
          vopat.addPixelContribution(ray.pixelID,color);
        }
          
        const int numLights = dvr.numDirLights();
        if (numLights == 0) {
          vopat.killRay(tid);            
          return;
        }
          
        int which = int(rnd() * numLights); if (which == numLights) which = 0;
        throughput *= ((float)numLights * dvr.lightRadiance(which));
        ray.throughput = to_half(throughput);
        
        ray.setDirection(dvr.lightDirection(which));
        dir = ray.getDirection();
            
        t0 = 0.f;
        t1 = CUDART_INF;
        boxTest(myBox,ray,t0,t1);
        t = 0.f; // reset t to the origin
        ray.isShadow = true;

        continue;
      }

      int nextNode = ray.isShadow ? -1 : computeNextNode(dvr,ray,t1,/*ray.dbg*/false);

      if (nextNode == -1) {
        vec3f color
          = (ray.isShadow)
          /* shadow ray that did reach the light (shadow rays that got
             blocked got terminated above) */
          ? throughput //albedo()
          /* primary ray going straight through */
          : Vopat::backgroundColor(ray,vopat);

        if (ray.crosshair) color = vec3f(1.f)-color;
        vopat.addPixelContribution(ray.pixelID,color);
        vopat.killRay(tid);
      } else {
        // ray has another node to go to - add to queue
        // ray.throughput = to_half(throughput);
        vopat.forwardRay(tid,ray,nextNode);
      }
    }
  };

  struct EmissionAbsorptionKernels : public Vopat
  {
    static inline __device__
    void traceRay(int tid,
                  const typename Vopat::ForwardGlobals &vopat,
                  const typename Vopat::VolumeGlobals  &dvr,
                  const typename Vopat::SurfaceGlobals &surf)
    {
      Ray ray = vopat.rayQueueIn[tid];

      vec3f throughput = from_half(ray.throughput);
      vec3f org = ray.origin;
      vec3f dir = ray.getDirection();
    
      const box3f myBox = dvr.rankBoxes[vopat.islandRank];
      float t0 = 0.f, t1 = CUDART_INF;
      boxTest(myBox,ray,t0,t1);

      Random rnd((int)ray.pixelID,vopat.sampleID+vopat.islandRank*0x123456);
#if VOPAT_UMESH
      vec3i numVoxels = 1024;
#else
      vec3i numVoxels = dvr.volume.dims;
#endif
      vec3i numCells  = numVoxels - 1;
      vec3i singleCell = vec3i(1); // just testing...

      // maximum possible voxel density
      const float dt = 1.f; // relative to voxels
      // const float DENSITY = .03f / ((vopat.xf.density == 0.f) ? 1.f : vopat.xf.density);//.03f;
      const float DENSITY = ((dvr.xf.density == 0.f) ? 1.f : dvr.xf.density);//.03f;
      float majorant = 1.f; // must be larger than the max voxel density
      float t = t0;
      while (true) {
        // Sample a distance
        t = t - (log(1.0f - rnd()) / (majorant*DENSITY)) * dt; 

        // A boundary has been hit
        if (t >= t1) {
          break;
        }

        // Update current position
        vec3f P = org + t * dir;

        // Sample heterogeneous media
        float f;
        if (!dvr.getVolume(f,P)) { t += dt; continue; }
        vec4f xf = dvr.transferFunction(f);
        f = xf.w;
      
        // Check if a collision occurred (real particles / real + fake particles)
        if (rnd() < f / majorant) {
          // if (ray.isShadow) {
          //   vec3f color = throughput * dvr.ambient();
          //   if (ray.crosshair) color = vec3f(1.f)-color;
          //   vopat.addPixelContribution(ray.pixelID,color);
          //   vopat.killRay(tid);            
          //   return;
          // } else {

          throughput *= vec3f(xf);
          
          const int numLights = dvr.numDirLights();
          if (numLights == 0.f) {
            throughput *= dvr.ambient();
            return;
          } else {
            int which = int(rnd() * numLights); if (which == numLights) which = 0;
            throughput *= dvr.lightRadiance(which);
          }

          vec3f color = throughput;
          if (ray.crosshair) color = vec3f(1.f)-color;
          vopat.addPixelContribution(ray.pixelID,color);
          vopat.killRay(tid);            
          return;
          
          // org = P; 
          //   ray.origin = org;
          //   ray.setDirection(dvr.lightDirection());
          //   dir = ray.getDirection();
            
          //   throughput *= vec3f(xf.x,xf.y,xf.z);
          //   ray.throughput = to_half(throughput);
            
          //   t0 = 0.f;
          //   t1 = CUDART_INF;
          //   boxTest(myBox,ray,t0,t1);
          //   t = 0.f; // reset t to the origin
          //   ray.isShadow = true;

          //   continue;
          // }
        }
      }

      int nextNode = computeNextNode(dvr,ray,t1,/*ray.dbg*/false);

      if (nextNode == -1) {
        vec3f color = throughput * Vopat::backgroundColor(ray,vopat);

        if (ray.crosshair) color = vec3f(1.f)-color;
        vopat.addPixelContribution(ray.pixelID,color);
        vopat.killRay(tid);
      } else {
        // ray has another node to go to - add to queue
        // ray.throughput = to_half(throughput);
        vopat.forwardRay(tid,ray,nextNode);
      }
    }
  };

  Renderer *createRenderer_WrongShadows(CommBackend *comm,
                                    Model::SP model,
                                    const std::string &fileNameBase,
                                        int rank, int numSPP)
  {
    VopatNodeRenderer<WrongShadowsKernels> *nodeRenderer
      = new VopatNodeRenderer<WrongShadowsKernels>
      (model,fileNameBase,rank);
    return new RayForwardingRenderer<WrongShadowsKernels::Ray>(comm,nodeRenderer,numSPP);
  }

  Renderer *createRenderer_NoShadows(CommBackend *comm,
                                     Model::SP model,
                                     const std::string &fileNameBase,
                                     int rank, int numSPP)
  {
    VopatNodeRenderer<EmissionAbsorptionKernels> *nodeRenderer
      = new VopatNodeRenderer<EmissionAbsorptionKernels>
      (model,fileNameBase,rank);
    return new RayForwardingRenderer<EmissionAbsorptionKernels::Ray>(comm,nodeRenderer,numSPP);
  }

} // ::vopat
