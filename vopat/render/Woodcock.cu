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
      const vec3i numMacrocells = dvr.mc.dims;
      const vec3f stretch = vec3f(1.f);
      const vec3f mcScale = vec3f(numMacrocells);
#else
      vec3i numVoxels = dvr.volume.dims;
      vec3i numCells  = numVoxels - 1;

      vec3i numMacrocells = dvr.mc.dims;

      if (ray.dbg) printf("Woodcock (%f %f %f) mc (%i %i %i)!\n"
                          ,myBox.upper.x
                          ,myBox.upper.y
                          ,myBox.upper.z
                          ,numMacrocells.x
                          ,numMacrocells.y
                          ,numMacrocells.z
                          );
      
      vec3f stretch = vec3f(numMacrocells)*dvr.mc.width / vec3f(dvr.volume.dims-1);
      const vec3f mcScale = vec3f(numMacrocells) * rcp(stretch);
#endif
      
#if 1
      // first do direct, then shadow
      bool rayKilled = false;
      for (int rayType = 0; rayType < 2; ++rayType) {
        if (rayType == 0 && ray.isShadow) continue;
        if (rayType == 1 && (rayKilled || !ray.isShadow)) continue;

        vec3f mcOrg = (org - myBox.lower) * rcp(myBox.size()) * mcScale;
        vec3f mcDir = dir * rcp(myBox.size()) * mcScale;
        
        dda::dda3
          (mcOrg,mcDir,t1,vec3ui(numMacrocells),
           [&](const vec3i &cellIdx, float t00, float t11) -> bool
           {
             interval<float> inputRange
               = dvr.mc.data[cellIdx.x + 
                             cellIdx.y * dvr.mc.dims.x + 
                             cellIdx.z * dvr.mc.dims.x * dvr.mc.dims.y 
                             ].inputRange;

             float majorant
               = dvr.mc.data[cellIdx.x + 
                             cellIdx.y * dvr.mc.dims.x + 
                             cellIdx.z * dvr.mc.dims.x * dvr.mc.dims.y 
                             ].maxOpacity;

             if (ray.dbg)
               printf("macrocell t=[%f %f], v=[%f %f] maj=%f\n",
                      t00,t11,
                      inputRange.lower,inputRange.upper,
                      majorant);
             
             bool noISO = true;

             if (surf.iso.numActive > 0) {
               for (int i=0; i<SurfaceIntersector::Globals::MaxISOs; ++i) {
                 if (surf.iso.active[i] && surf.iso.values[i] >= inputRange.lo
                                        && surf.iso.values[i] <= inputRange.hi) {
                   noISO = false;
                   break;
                 }
               }
             }


             if (majorant <= 0.f && noISO)
               return true; // this cell is empty, march to the next cell
             
             // test if there's an intersection with a surface
             Surflet sample = surf.intersect(ray,t00,t11);
             if (sample.wasHit()) {
               t11 = min(t11,sample.t);
             }

             // maximum possible voxel density
             const float DENSITY = ((dvr.xf.density == 0.f) ? 1.f : dvr.xf.density);//.03f;
             float t = t00;
             while (true) {
               // Sample a distance
               t = t - (log(1.0f - rnd()) / (majorant*DENSITY)); 
               
               if (/*we left the cell: */t >= t11) {
                 /* leave this cell, but tell DDA to keep on going */
                 break;
               }

                      // Update current position
               vec3f P = org + t * dir;
               
               // Sample heterogeneous media
               float f;
               if (!dvr.getVolume(f,P,ray.dbg))
                 /* could not even sample his volume; assume
                    it's 0 and move on */
                 continue; 
               
               vec4f xf = dvr.transferFunction(f,ray.dbg);
               f = xf.w;
               vec3f albedo = vec3f(xf);
               if (ray.dbg) printf("volume at %f is %f -> %f %f %f: %f\n",
                                   t,f,xf.x,xf.y,xf.z,xf.w);
               
               // Check if a collision occurred (real particles / real + fake particles)
               if (rnd() >= (f / (majorant/**DENSITY*/))) {
                 if (ray.dbg) printf(" -> sample %f/%f -> rejected\n",
                                     f,majorant*DENSITY);
                 // sampled a virtual volume; keep on going
                 continue;
               }

               sample.type     = Surflet::Density;
               sample.t        = t;
               sample.isectPos = P;
               sample.kd       = albedo;

               if (f > 1e-4f) {
                 vec3f g;
                 dvr.getGradient(g,P,ray.dbg);
                 sample.gn = sample.sn = normalize(g);
               } else {
                 sample.gn = sample.sn = vec3f(0.f);
               }
               break;
             }
              
             if (!sample.wasHit())
               return true; // keep DDA on going

             if (ray.isShadow) {
               rayKilled = true;
               return false; // terminate DDA
             }

             org = sample.isectPos;//worldP; 
             ray.origin = org;
             
             {
               // add ambient illumination 
               vec3f color = throughput * dvr.ambient();
               if (ray.crosshair) color = vec3f(1.f)-color;
               vopat.addPixelContribution(ray.pixelID,color);
             }
             
             const int numLights = dvr.numDirLights();
             if (numLights == 0) {
               rayKilled = true;
               return false;
             }
             
             // add BRDF shading
             int which = dvr.uniformSampleOneLight(rnd);
             if (length(sample.sn) > .9f) {
               throughput *= sample.kd * fabsf(dot(sample.sn,dvr.lightDirection(which)))
                    * dvr.lightRadiance(which);
             } else {
               throughput *= sample.kd * ((float)numLights * dvr.lightRadiance(which));
             }

             ray.throughput = to_half(throughput);
          
#if 0
             vec3f lightDir = dvr.lightDirection(which);
#else
             float lpdf = 0.f;
             vec3f lightDir = dvr.sampleLightDirection(which,rnd,lpdf);
#endif
             ray.setDirection(lightDir);
             dir = ray.getDirection();
             
             t0 = 0.f;
             t1 = CUDART_INF;
             boxTest(myBox,ray,t0,t1);
             t = 0.f; // reset t to the origin
             ray.isShadow = true;
            
             if (ray.dbg) {
               printf("Shadow ray: origin: (%f,%f,%f), direction: (%f,%f,%f), t0: %f, t1: %f\n",
                      ray.origin.x,ray.origin.y,ray.origin.z,
                      ray.getDirection().x,
                      ray.getDirection().y,
                      ray.getDirection().z,
                      t0,t1);
             }

             // restart DDA
             rayKilled = false;
             return false; // terminate DDA
           },
           false
           /*ray.dbg*/);          
        
        if (rayKilled) {
          vopat.killRay(tid); 
          return;        
        }
      }
      
#else
      // maximum possible voxel density
      // const float DENSITY = .03f / ((vopat.xf.density == 0.f) ? 1.f : vopat.xf.density);//.03f;
      const float DENSITY = ((dvr.xf.density == 0.f) ? 1.f : dvr.xf.density);//.03f;
      float majorant = 1.f; // must be larger than the max voxel density
      float t = t0;
      if (isnan(t)) printf("t is NAN at start!\n");

# if 0
      if (!ray.dbg)  { vopat.killRay(tid); return; }
# endif
      
      while (true) {
        // Sample a distance
        const float xi = rnd();
        const float dt = - (log(1.0f - xi) / (majorant*DENSITY)); 
        t = t + dt;
        if (isnan(t)) printf("t is NAN xi %f dt %f dens %f org %f %f %f dir %f %f %f!\n",
                             xi,dt,DENSITY,org.x,org.y,org.z,dir.x,dir.y,dir.z);

        // A boundary has been hit
        if (t >= t1) {
          break;
        }

        // Update current position
        vec3f P = org + t * dir;
        if (isnan(P.x+P.y+P.z))
          printf("P is NAN xi %f dt %f dens %f org %f %f %f dir %f %f %f!\n",
                 xi,dt,DENSITY,org.x,org.y,org.z,dir.x,dir.y,dir.z);

        // Sample heterogeneous media
        float f;
        if (!dvr.getVolume(f,P)) { vopat.killRay(tid); return; }

        // if (!dvr.getVolume(f,P)) { t += dt; continue; }
        vec4f xf = dvr.transferFunction(f);
        if (ray.dbg) printf("volume (t=%f) %f dens %f,xf %f %f %f : %f\n",
                            t,f,DENSITY,xf.x,xf.y,xf.z,xf.w);
        f = xf.w;
      
        // Check if a collision occurred (real particles / real + fake particles)
        if (rnd() < f / majorant) {
          if (ray.isShadow) {
            // vec3f color = throughput * dvr.ambient();
            // if (ray.crosshair) color = vec3f(1.f)-color;
            // vopat.addPixelContribution(ray.pixelID,color);
            vopat.killRay(tid);            
            return;
          } else {
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
            if (numLights == 0.f) {
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
        }
      }
#endif

      int nextNode = computeNextNode(dvr,ray,t1,/*ray.dbg*/false);

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

  Renderer *createRenderer_Woodcock(CommBackend *comm,
                                    Model::SP model,
                                    const std::string &fileNameBase,
                                    int rank,
                                    int numSPP)
  {
    VopatNodeRenderer<WoodcockKernels> *nodeRenderer
      = new VopatNodeRenderer<WoodcockKernels>
      (model,fileNameBase,rank);
    return new RayForwardingRenderer<WoodcockKernels::Ray>(comm,nodeRenderer,numSPP);
  }

} // ::vopat
