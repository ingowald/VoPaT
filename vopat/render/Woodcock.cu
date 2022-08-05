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

#ifndef VOPAT_MAX_BOUNCES
  // 0 bounces == direct illum only
# define VOPAT_MAX_BOUNCES 0
#endif
  
  struct WoodcockKernels : public Vopat
  {
    static inline __device__
    void traceRay(int tid,
                  const typename Vopat::ForwardGlobals &vopat,
                  const typename Vopat::VolumeGlobals  &dvr,
                  const typename Vopat::SurfaceGlobals &surf);
  };
  

  inline __device__
  void WoodcockKernels::traceRay(int tid,
                                 const typename Vopat::ForwardGlobals &vopat,
                                 const typename Vopat::VolumeGlobals  &dvr,
                                 const typename Vopat::SurfaceGlobals &surf)
  {
    Ray ray = vopat.rayQueueIn[tid];
    if (!checkOrigin(ray))
      printf("bad ray %i/%i in enter traceray %f %f %f\n",
             tid,vopat.numRaysInQueue,
             ray.origin.x,ray.origin.y,ray.origin.z);

#if DEBUG_FORWARDS
    if (ray.numFwds > 8)
      printf("ray forwarded %i times\n",ray.numFwds);
#endif
    ray.dbg = vopat.fishy;

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

    // vec3i numMacrocells = dvr.mc.dims;

    const box3f myBox = dvr.rankBoxes[vopat.islandRank];
    if (ray.dbg)
      printf("======================\n(%i) Woodcock::localTrace box (%f %f %f) (%f %f %f) mc (%i %i %i)!\n"
             ,vopat.islandRank
             ,myBox.lower.x
             ,myBox.lower.y
             ,myBox.lower.z
             ,myBox.upper.x
             ,myBox.upper.y
             ,myBox.upper.z
             ,numMacrocells.x
             ,numMacrocells.y
             ,numMacrocells.z
             );
      
    // vec3f stretch = vec3f(numMacrocells)*dvr.mc.width / vec3f(dvr.volume.dims-1);
    // const vec3f mcScale = vec3f(numMacrocells) * rcp(stretch);

    vopat::Intersection closestHit;
      
    // ==================================================================
    // do however many bounces we can/want to do on the local
    // node. if we leave this loop the given input ray is done, one
    // wya or another (ie, it's been obsorbed, has terminated
    // traversal, or forwarded to another node.... but either way
    // it's done
    // ==================================================================
    int nextRankToSendTo = -1;
    // int numOuterLoop = 0;
    int age = -1;
    while (true) {
      ++age;
      // if (++numOuterLoop > 10) {
      //   printf("num outer loop %i\n",numOuterLoop);
      //   break;
      // }
      closestHit.type = vopat::Intersection::NONE;
      closestHit.t    = FLT_MAX;
        
      vec3f throughput = from_half(ray.throughput);
      vec3f dir = ray.getDirection();
    
      // translate ray to macro cell space
      const vec3f mcOrg = (ray.origin - myBox.lower) * rcp(myBox.size()) * mcScale;
      const vec3f mcDir = dir * rcp(myBox.size()) * mcScale;


      // ==================================================================
      // trace ONE ray - ie, the latest ray segment along the
      // path. we trace this through macrocells using DDA.
      // ==================================================================
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
      const float box_t0 = t0;
      const float box_t1 = t1;
      vec3f stretch = vec3f(numMacrocells)*dvr.mc.width / vec3f(dvr.volume.dims-1);
      const vec3f mcScale = vec3f(numMacrocells) * rcp(stretch);
#endif

      if (t0 == t1) {
        break;
      }
      
      if (t1 < t0) {
        // if (t1 != 0 || t0 != 0)
        {
          printf("(%i) - huh ... ray we got sent (age=%i) doesn't actually overlap this box!? ...\n",vopat.islandRank,age);
          printf("box : (%f %f %f)(%f %f %f)\n",
                 myBox.lower.x,
                 myBox.lower.y,
                 myBox.lower.z,
                 myBox.upper.x,
                 myBox.upper.y,
                 myBox.upper.z);
          printf("ray %f %f %f + t (%f %f %f) range (%f %f)\n",
                 ray.origin.x,
                 ray.origin.y,
                 ray.origin.z,
                 dir.x,
                 dir.y,
                 dir.z,t0,t1);
        }

        // we're just KILLING these rays here, but this is actually
        // not entirely correct - the ray COULD be out of box by
        // having been offset during shading, in which case it is
        // STILL a valid ray, just not on this rank...
        nextRankToSendTo = -1;
        break;
      }
      if (ray.dbg) {
        printf("----------------------\n");
        printf("local trace ray age=%i, isShadow %i, range %f %f\n",
               int(ray.numBounces),int(ray.isShadow),t0,t1);
        printf("ray %f %f %f + t (%f %f %f)\n",
               ray.origin.x,
               ray.origin.y,
               ray.origin.z,
               dir.x,
               dir.y,
               dir.z);
      }
      int numStepsDone = 0;
      dda::dda3
        (mcOrg,mcDir,t1,vec3ui(numMacrocells),
         [&](const vec3i &cellIdx, float t00, float t11) -> bool
         {
           if (cellIdx.x < 0 ||
               cellIdx.y < 0 ||
               cellIdx.z < 0 ||
               cellIdx.x >= numMacrocells.x ||
               cellIdx.y >= numMacrocells.y ||
               cellIdx.z >= numMacrocells.z) {
             printf("UGH - FISHY DDA CELL %i %i %i / %i %i %i!\n",
                    cellIdx.x,
                    cellIdx.y,
                    cellIdx.z,
                    numMacrocells.x,
                    numMacrocells.y,
                    numMacrocells.z
                    );
             return false;
           }
           if (++numStepsDone > 1024) {
             printf("LOTS of steps....\n");
             return false;
           }
           if (t00 >= closestHit.t)
             // we've traversed beyond a hit we alreayd have ...
             return false;

           // read this macrocell
           const auto mc = dvr.mc.data[cellIdx.x + 
                                       cellIdx.y * dvr.mc.dims.x + 
                                       cellIdx.z * dvr.mc.dims.x * dvr.mc.dims.y];

#if 0
           // ==================================================================
           // perform is-intersection within this cell...
           // ==================================================================
           const interval<float> &inputRange = mc.inputRange;
             
           bool macroCellOverlapsAnIsoValue = false;
           // if (ray.dbg) printf("ISO CONFIG %i : %i=%f %i=%f\n",
           //                     surf.iso.numActive,
           //                     surf.iso.active[0],
           //                     surf.iso.values[0],
           //                     surf.iso.active[1],
           //                     surf.iso.values[1]);
                               
           if (surf.iso.numActive > 0) {
             printf("iso!?\n");
             for (int i=0; i<SurfaceIntersector::Globals::MaxISOs; ++i) {
               if (!surf.iso.active[i] 
                   || (inputRange.lo > surf.iso.values[i])
                   || (inputRange.hi < surf.iso.values[i]))
                 continue;
               macroCellOverlapsAnIsoValue = true;
               break;
             }
           }
           if (macroCellOverlapsAnIsoValue) {
             // if (ray.dbg) printf("iso-surface overlaps!\n");
#if VOPAT_UMESH
             surf.intersect(closestHit,ray.origin,dir,t00,t11,ray.dbg);
#else
             surf.intersect(closestHit,ray.origin,dir,max(box_t0,t00),min(t1,t11),ray.dbg);
#endif
             // surf.intersect(closestHit,ray.origin,dir,t00,min(t1,t11),ray.dbg);
             // t11 = min(t11,closestHit.t);
           }
#endif
           
           // ==================================================================
           // perform woodcock-tracking within this cell, using this
           // cell's majorant
           // ==================================================================
           const float majorant = mc.maxOpacity;
           if (majorant <= 0.f)
             // cell is fully transparent, we can skip it right away
             return (t11 < closestHit.t);
             
           const float DENSITY = ((dvr.xf.density == 0.f) ? 1.f : dvr.xf.density);//.03f;
           float t = t00;
           while (true) {
             // Sample a distance
             t = t - (logf(1.f - rnd()) / (majorant*DENSITY)); 
               
             if (t >= t11)
               /* woodcock tracking stepped beyond end of
                  macrocell: */
               return (t11 < closestHit.t);
             
             // ------------------------------------------------------------------
             // query volume at given new sample pos
             // ------------------------------------------------------------------
             const vec3f samplePos = ray.origin + t * dir;
             float scalarFieldAtSamplePos;
             if (!dvr.getVolume(scalarFieldAtSamplePos,samplePos,ray.dbg))
               /* could not even sample his volume; assume
                  it's 0 and move on */
               continue; 
               
             // ------------------------------------------------------------------
             // compute sample density from transfer functoin
             // ------------------------------------------------------------------
             const vec4f xf = dvr.transferFunction(scalarFieldAtSamplePos,ray.dbg);
               
             // Check if a collision occurred (real particles / real + fake particles) 
             if (rnd() >= (xf.w / (majorant*DENSITY))) {
               // sampled a virtual volume; keep on going
               continue;
             } else {
               // yay! we had a surface interaction!
               closestHit.type = Intersection::VOLUME;
               closestHit.t    = t;
               closestHit.kd   = {xf.x,xf.y,xf.z};
               closestHit.Ng   = vec3f(0.f); // do NOT use a normal for volumes
               // aaaand .... done - won't ever find anything better than this!
               if (ray.dbg) printf(" => had a VOLUME hit at %f, f=%f xf=(%f %f %f;%f), maj %f dens %f\n",
                                   closestHit.t,
                                   scalarFieldAtSamplePos,
                                   xf.x,xf.y,xf.z,xf.w,
                                   majorant,
                                   DENSITY);
               return false;
             }
           }
           // should never reach here - above loop will always 'return'
         },ray.dbg);

      // ==================================================================
      // the latest ray segment in the path has now been traced;
      // let's check if it did fine a hit or not, and react
      // accordingly.
      // ==================================================================
        
      if (ray.dbg)
        printf("----------------------\ndone trace, hit type %i dist %f max dist this rank %f\n",
               int(closestHit.type),closestHit.t,t1);
      if (closestHit.type == Intersection::NONE) {
        // ------------------------------------------------------------------
        // if we reach here we did NOT have a hit .... but there
        // could still be another rank that might have one for this
        // ray - check for that.
        // ------------------------------------------------------------------
        nextRankToSendTo = computeNextNode(dvr,ray,t1 * 1.001f,/*ray.dbg*/false);
        if (ray.dbg)
          printf(" => NO_HIT case, next rank %i\n",nextRankToSendTo);
        if (nextRankToSendTo >= 0) {
          // there IS another rank - no matter what ray it is, just
          // forward and terminate all local work.
          break;
        } else {
          // there's NO other rank to send this ray to, so it's truly
          // done tracing now. React accordingly:
          vec3f imageContribution = 0.f;
          if (ray.isShadow) {
            imageContribution = throughput;
          } else if (ray.numBounces == 0) {
            imageContribution = Vopat::backgroundColor(ray,vopat);
            if (ray.crosshair) imageContribution = vec3f(1.f)-imageContribution;
          } else {
            imageContribution = throughput * dvr.ambientEnvLight();
            if (ray.crosshair) imageContribution = vec3f(1.f)-imageContribution;
          }
          if (imageContribution != vec3f(0.f)) {
            if (ray.dbg) printf(" -> image contrib %f %f %f\n",
                                imageContribution.x,
                                imageContribution.y,
                                imageContribution.z);
            // if (ray.crosshair) imageContribution = vec3f(1.f)-imageContribution;
            vopat.addPixelContribution(ray.pixelID,imageContribution);
          }
          break;
        }
        // should never reach here - that path has been terminated
        // and is one.
        
      } else {
        // ------------------------------------------------------------------
        // we DID have a hit in this node - react to it.
        // ------------------------------------------------------------------
        if (ray.isShadow) {
          // if the ray was a shadow ray; it's now been occluded,
          // has nothing to add to the image, and should terminate.
          break;
        }

        // ------------------------------------------------------------------
        // we DID have a hit, and it's NOT a shadow ray -> shade
        // ------------------------------------------------------------------
        ray.origin = ray.origin + closestHit.t * dir;
        if (!myBox.contains(ray.origin))
          printf("**PRE**-OFFSET HIT OUTSIDE BOX  %f %f %f, hit.t %f in %f %f!\n",
                 ray.origin.x,
                 ray.origin.y,
                 ray.origin.z,
                 closestHit.t,
#if VOPAT_UMESH
                 -1.f,-1.f
#else
                 box_t0,box_t1
#endif
                 );

        // if (ray.dbg) printf("SHADE origin to %f %f %f\n",
        //                 ray.origin.x,
        //                 ray.origin.y,
        //                 ray.origin.z);
        
        vec3f ambientContribution = 0.f;
        if (closestHit.type != Intersection::VOLUME) {
          closestHit.Ng = normalize(closestHit.Ng);
          if (dot(dir,closestHit.Ng)  > 0.f) 
            closestHit.Ng = - closestHit.Ng;

          ray.origin = ray.origin + .5f*closestHit.Ng;
          if (!myBox.contains(ray.origin)) {
            printf("POST-OFFSET HIT OUTSIDE BOX  %f %f %f, hit.t %f in %f %f; dg.Ng %f %f %f!\n",
                   ray.origin.x,
                   ray.origin.y,
                   ray.origin.z,
                   closestHit.t,
#if VOPAT_UMESH
                   -1.f,-1.f,
#else
                   box_t0,box_t1,
#endif
                   closestHit.Ng.x,
                   closestHit.Ng.y,
                   closestHit.Ng.z);
          // if (ray.dbg) printf("offset origin to %f %f %f\n",
          //                 ray.origin.x,
          //                 ray.origin.y,
          //                 ray.origin.z);
            nextRankToSendTo = -1;
            break;
          }
        }
        
        // ------------------------------------------------------------------
        // add ambient illumination
        // ------------------------------------------------------------------
#if 0
        if (closestHit.type == Intersection::VOLUME) { 
          vopat.addPixelContribution(ray.pixelID,throughput * dvr.ambient());
        } else {
          vopat.addPixelContribution(ray.pixelID,throughput * dvr.ambient()
                                     * dot(closestHit.Ng,normalize(dir)));
        }
#endif
          
        // ------------------------------------------------------------------
        // check if we can/should create a shadow ray we can only
        // trace ONE outgoing ray, so we'll have to choose if we
        // want to do a shadow ray or a bounce ray
        // ------------------------------------------------------------------
        float pdf = 1.f;
        bool pathBecomesShadowRay = true;
        // if (ray.dbg) printf("check bounces %i < %i?\n",
        //                     (int)ray.numBounces,(int)VOPAT_MAX_BOUNCES);
        if ((int)ray.numBounces < (int)VOPAT_MAX_BOUNCES) {
          // choose stochastically:
          float xi = rnd();
          pathBecomesShadowRay = (xi < .5f);
          // if (ray.dbg) printf("check xi %f result %i\n",xi,int(pathBecomesShadowRay));
          pdf *= 0.5f;
        } else {
          pathBecomesShadowRay = true;
        }
          
        // ------------------------------------------------------------------
        // if closestHit.type == HIT_SURFACE
        // ------------------------------------------------------------------
        if (pathBecomesShadowRay) {
          vec3f ldir;
          vec3f lpow;
          float lightPDF = dvr.sampleLight(rnd,
                                           // surface:
                                           ray.origin,closestHit.Ng,
                                           // result:
                                           ldir,lpow);
          if (lightPDF == 0.f)
            break;
          pdf          *= lightPDF;
          throughput   *= lpow;
          dir           = ldir;
          ray.isShadow  = true;
          if (ray.dbg) printf(" -> made SHADOW ray, new TP %f %f %f\n",
                              throughput.x,throughput.y,throughput.z);
        } else {
          float samplePDF
            = closestHit.sample(rnd,dir,ray.dbg);
          if (samplePDF == 0.f)
            break;
          pdf *= samplePDF;
          ray.numBounces = ray.numBounces+1;
          if (ray.dbg) printf(" -> made BOUNCE ray\n");
        }
          
        if (pdf == 0.f)
          break;

        // set and re-read from ray, to account for possible precision reduction
        ray.setDirection(dir);
        dir = ray.getDirection();
        throughput *= (closestHit.eval(dir,ray.dbg) / pdf);
        // if (ray.dbg) printf(" -> done EVAL, new TP %f %f %f\n",
        //                     throughput.x,throughput.y,throughput.z);
        if (reduce_max(throughput) <= 1e-2f)
          // probably sampled a back-facing light ...
          break;
          
        ray.throughput = to_half(throughput);
      }
    }
    if (nextRankToSendTo >= 0) {
      // if (ray.dbg)
      //   printf("FORWARDING RAY TO %i\n",nextRankToSendTo);
      vopat.forwardRay(tid,ray,nextRankToSendTo);
    } else 
      vopat.killRay(tid);
  }

  Renderer *createRenderer_Woodcock(CommBackend *comm,
                                    Model::SP model,
                                    const std::string &fileNameBase,
                                    int rank,
                                    int numSPP)
  {
    VopatNodeRenderer<WoodcockKernels> *nodeRenderer
      = new VopatNodeRenderer<WoodcockKernels>
      (model,fileNameBase,rank,comm->worker.gpuID);
    return new RayForwardingRenderer<WoodcockKernels::Ray>(comm,nodeRenderer,numSPP);
  }

} // ::vopat
