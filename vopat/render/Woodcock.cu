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

#include "vopat/render/VopatBase.h"
#include "DDA.h"

namespace vopat {

  struct WoodcockKernels : public Vopat
  {
    static inline __device__
    void traceRay(int tid,
                  const typename Vopat::ForwardGlobals &vopat,
                  const typename Vopat::VolumeGlobals  &dvr)
    {
      Ray ray = vopat.rayQueueIn[tid];

      vec3f throughput = from_half(ray.throughput);
      vec3f org = ray.origin;
      vec3f dir = ray.getDirection();
    
      const box3f myBox = dvr.rankBoxes[vopat.myRank];
      float t0 = 0.f, t1 = CUDART_INF;
      boxTest(myBox,ray,t0,t1);

      Random rnd((int)ray.pixelID,vopat.sampleID+vopat.myRank*0x123456);
      vec3i numVoxels = dvr.volume.dims;
      vec3i numCells  = numVoxels - 1;

      vec3i singleCell = vec3i(1); // just testing...
      vec3i numMacrocells = dvr.mc.dims;

      if (ray.dbg) printf("Woodcock (%f %f %f) mc (%i %i %i)!\n"
                          ,myBox.upper.x
                          ,myBox.upper.y
                          ,myBox.upper.z
                          ,numMacrocells.x
                          ,numMacrocells.y
                          ,numMacrocells.z
                          );
      
      auto worldToUnit = affine3f(
        linear3f(
          vec3f((myBox.upper.x - myBox.lower.x), 0.f, 0.f),
          vec3f(0.f, (myBox.upper.y - myBox.lower.y), 0.f),
          vec3f(0.f, 0.f, (myBox.upper.z - myBox.lower.z))
        ).inverse(),
        vec3f(0.f, 0.f, 0.f)
      );
      auto unitToWorld = affine3f(
        linear3f(
          vec3f((myBox.upper.x - myBox.lower.x), 0.f, 0.f),
          vec3f(0.f, (myBox.upper.y - myBox.lower.y), 0.f),
          vec3f(0.f, 0.f, (myBox.upper.z - myBox.lower.z))
        ),
        vec3f(0.f, 0.f, 0.f)
      );      
      auto gridToUnit = affine3f(
        linear3f(
          vec3f(numMacrocells.x, 0.f, 0.f),
          vec3f(0.f, numMacrocells.y, 0.f),
          vec3f(0.f, 0.f, numMacrocells.z)
        ).inverse(),
        vec3f(0.f, 0.f, 0.f)
      );
      auto unitToGrid = affine3f(
        linear3f(
          vec3f(numMacrocells.x, 0.f, 0.f),
          vec3f(0.f, numMacrocells.y, 0.f),
          vec3f(0.f, 0.f, numMacrocells.z)
        ),
        vec3f(0.f, 0.f, 0.f)
      );



      
      
      

#if 1

      // first do direct, then shadow
      bool rayKilled = false;
      for (int tmp = 0; tmp < 2; ++tmp) {
        vec3f gLower = xfmPoint(unitToGrid, xfmPoint(worldToUnit, myBox.lower));
        vec3f gorg = org /*- myBox.lower*/; 
        gorg = xfmPoint(unitToGrid, xfmPoint(worldToUnit, org)) - gLower;
        vec3f gdir = xfmVector(unitToGrid, xfmVector(worldToUnit, dir));

        dda::dda3(gorg,gdir,t1,
          vec3ui(numMacrocells),
          [&](const vec3i &cellIdx, float t00, float t11) -> bool
          {
            float majorant = dvr.mc.data[
              cellIdx.x + 
              cellIdx.y * dvr.mc.dims.x + 
              cellIdx.z * dvr.mc.dims.x * dvr.mc.dims.y 
            ].maxOpacity; // now pulling majorant from macrocell, rather than just assuming 1.

            if (majorant <= 0.f) return true; // this cell is empty, march to the next cell
            
            // maximum possible voxel density
            const float dt = 1.f; // relative to voxels
            const float DENSITY = ((dvr.xf.density == 0.f) ? 1.f : dvr.xf.density);//.03f;
            float t = t00;
            while (true) {
              // Sample a distance
              t = t - (log(1.0f - rnd()) / (majorant*DENSITY)) * dt; 

              // A boundary has been hit
              if (t >= t11) {
                break;
              }

              // Update current position
              vec3f P = gorg + t * gdir;
              vec3f worldP = xfmPoint(gridToUnit, xfmPoint(unitToWorld, P + gLower));

              // Sample heterogeneous media
              float f;
              if (!dvr.getVolume(f,worldP,ray.dbg)) { 
                // t += dt; // NM: not necessary, the sampled distance moves t forward.
                continue; 
              }
              vec4f xf = dvr.transferFunction(f,ray.dbg);
              f = xf.w;
              if (ray.dbg) printf("volume at %f is %f -> %f %f %f: %f\n",
                              t,f,xf.x,xf.y,xf.z,xf.w);
              // f = transferFunction(f);
            
              // Check if a collision occurred (real particles / real + fake particles)
              if (rnd() < f / (majorant*DENSITY)) {
                if (ray.isShadow) {
                  vec3f color = throughput * dvr.ambient();
                  if (ray.crosshair) color = vec3f(1.f)-color;
                  vopat.addPixelContribution(ray.pixelID,color);
                  vopat.killRay(tid);            
                  rayKilled = true;
                  return false; // terminate DDA
                } else {
                  org = worldP; 
                  ray.origin = org;
                  ray.setDirection(dvr.lightDirection());
                  dir = ray.getDirection();
                  
                  throughput *= vec3f(xf.x,xf.y,xf.z);
                  ray.throughput = to_half(throughput);
                  
                  t0 = 0.f;
                  t1 = CUDART_INF;
                  boxTest(myBox,ray,t0,t1);
                  t = 0.f; // reset t to the origin
                  ray.isShadow = true;

      #if ISO_SURFACE
                  // eventually need to do iso-marching here, too!!!!
                  isoDistance = -1;
      #endif
                  // continue;
                  // restart DDA
                  rayKilled = false;
                  return false; // terminate DDA
                }
              }
            }

            return true; // continue DDA
          },
          false
          /*ray.dbg*/);          

        if (rayKilled) 
        {
          vopat.killRay(tid); 
          return;        
        }
      }

#else
#ifdef ISO_SURFACE
      NOT WORKING YET
        float isoDistance = -1.f;
      {
        int numSegments = int(t1-t0+1);
        vec3f P1 = org + t0 * dir;
        float f1 = getClampVolume(f,globals,P);
        for (int i=1;i<=numSegments;i++) {
          float f0 = f1;

          float seg_t1 = t0 + float(i)/(t1-t0);
          P1 = org + seg_t1 * dir;
          f1 = getClampVolume(f,globals,P);

          if ((f0 != f1) && (f1 - ISO_VALUE)*(f0 - ISO_VALUE) <= 0.f) {
            isoDistance = (ISO_VALUE - f0) / (f1 - f0);
            break;
          }
        }
      }
      if (isoDistance >= 0.f)
        t1 = isoDistance;
#endif
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
#if ISO_SURFACE
          if (isoDistance >= 0.f) {
            // we DID have an iso-surface hit!
            org = org + isoDistance * dir;
            vec3f N = normalize(gradient(org,globals));
            if (dot(N,dir) > 0.f) N = -N;
            vec3f r = sampleCosineHemisphere();
          }
#endif
          break;
        }

        // Update current position
        vec3f P = org + t * dir;

        // Sample heterogeneous media
        float f;
        if (!dvr.getVolume(f,P)) { t += dt; continue; }
        vec4f xf = dvr.transferFunction(f);
        if (dbg) printf("volume at %f is %f -> %f %f %f: %f\n",
                        t,f,xf.x,xf.y,xf.z,xf.w);
        f = xf.w;
        // f = transferFunction(f);
      
        // Check if a collision occurred (real particles / real + fake particles)
        if (rnd() < f / majorant) {
          if (ray.isShadow) {
            vec3f color = throughput * dvr.ambient();
            if (ray.crosshair) color = vec3f(1.f)-color;
            vopat.addPixelContribution(ray.pixelID,color);
            vopat.killRay(tid);            
            return;
          } else {
            org = P; 
            ray.origin = org;
            ray.setDirection(dvr.lightDirection());
            dir = ray.getDirection();
            
            throughput *= vec3f(xf.x,xf.y,xf.z);
            ray.throughput = to_half(throughput);
            
            t0 = 0.f;
            t1 = CUDART_INF;
            boxTest(myBox,ray,t0,t1);
            t = 0.f; // reset t to the origin
            ray.isShadow = true;

#if ISO_SURFACE
            // eventually need to do iso-marching here, too!!!!
            isoDistance = -1;
#endif
            continue;
          }
        }
      }
      #endif

      int nextNode = computeNextNode(dvr,ray,t1,/*ray.dbg*/false);

      if (nextNode == -1) {
        vec3f color
          = (ray.isShadow)
          /* shadow ray that did reach the light (shadow rays that got
             blocked got terminated above) */
          ? dvr.lightColor() * throughput //albedo()
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

  Renderer *createRenderer_Woodcock(CommBackend *comm,
                                    Model::SP model,
                                    const std::string &fileNameBase,
                                    int rank)
  {
    VopatNodeRenderer<WoodcockKernels> *nodeRenderer
      = new VopatNodeRenderer<WoodcockKernels>
      (model,fileNameBase,rank);
    return new RayForwardingRenderer<WoodcockKernels::Ray>(comm,nodeRenderer);
  }

} // ::vopat
