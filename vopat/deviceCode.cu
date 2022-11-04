// ======================================================================== //
// Copyright 2022-2022 Ingo Wald                                            //
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

// #include "deviceCode.h"
#include "LaunchParams.h"
// #include "vopat/NodeRenderer.h"
// #include "vopat/NextDomainKernel.h"
#include <cuda.h>
#include "vopat/DDA.h"

using namespace vopat;

// extern "C" __constant__ uint8_t optixLaunchParams[sizeof(LaunchParams)];
extern "C" __constant__ LaunchParams optixLaunchParams;

namespace vopat {

  inline __device__ const LaunchParams &LaunchParams::get()
  { return optixLaunchParams; }

  inline __device__
  void traceRayLocally(Random &rng, Ray &ray);

  
  // ##################################################################
  // NextDomainKernel accel struct programs
  // ##################################################################
  
  OPTIX_BOUNDS_PROGRAM(proxyBounds)(const void  *geomData,
                                    box3f       &primBounds,
                                    const int    primID)
  {
    NextDomainKernel::Geom &geom = *(NextDomainKernel::Geom*)geomData;
    auto &proxy = geom.proxies[primID];
    if (proxy.majorant == 0.f)
      primBounds = box3f();
    else
      primBounds = proxy.domain;
  }

  OPTIX_INTERSECT_PROGRAM(proxyIsec)()
  {
    auto &lp   = LaunchParams::get();
    auto &geom = owl::getProgramData<NextDomainKernel::Geom>();
    int primID = optixGetPrimitiveIndex();
    auto proxy = geom.proxies[primID];

    auto &prd = owl::getPRD<NextDomainKernel::PRD>();
    
    if (1 && prd.dbg)
      printf("(%i) isec proxy %i (%f %f %f)(%f %f %f):%i\n",
             lp.rank,
             primID,
             proxy.domain.lower.x,
             proxy.domain.lower.y,
             proxy.domain.lower.z,
             proxy.domain.upper.x,
             proxy.domain.upper.y,
             proxy.domain.upper.z,
             proxy.rankID);


    /* if this is a 'Find_Self_' phase we'll need to skip _all_other_
       ranks' proxies: */
    if (prd.phase == NextDomainKernel::Phase_FindSelf &&
        proxy.rankID != lp.rank)
      return;
    
    /* if this is a 'Find_Next_' phase we'll need to skip everybody
       that's already been tagged in the 'others' phase: */
    else if (prd.phase == NextDomainKernel::Phase_FindNext &&
             prd.alreadyTravedMask.hasBitSet(proxy.rankID))
      return;
    
    /*! if this is a 'Find_Others_' phase we can skip everybody that's
      already tagged */
    else if (prd.phase == NextDomainKernel::Phase_FindOthers &&
             prd.alreadyTravedMask.hasBitSet(proxy.rankID))
      return;

    vec3f org = optixGetWorldRayOrigin();
    vec3f dir = optixGetWorldRayDirection();
    float t0 = optixGetRayTmin();
    float t1 = optixGetRayTmax();
    if (!boxTest(proxy.domain,org,dir,t0,t1))
      // we didn't even hit this proxy, so everything else is moot;
      // exit.
      return;

    if (1 && prd.dbg)
      printf("(%i) proxy rank %i hit %f ray.tmax %f prd.dist %f prd.rank %i\n",
             lp.rank,proxy.rankID,
             t0,optixGetRayTmax(),prd.closestDist,(int)prd.closestRank);

    
    // ------------------------------------------------------------------
    // ok, we DID hit that proxy, at distance t0
    // ------------------------------------------------------------------

    /* if this is the find _others_ phase: check if the proxy _would_
       have been accepted before the currect-rank's hit, and if so,
       set the bit. Either way, exit right away (no accept) */
    if (prd.phase == NextDomainKernel::Phase_FindOthers) {
      if (t0 < prd.closestDist ||
          t0 == prd.closestDist && proxy.rankID < prd.closestDist)
        /* this hit WOULD have been accepted if we hadn't explcitly
           included it in the find-self phase */
        prd.alreadyTravedMask.setBit(proxy.rankID);
      return;
    }
    
    /* otherwise, we're either finding the first, ourselves, or the
       next other; in all of which cases we need to store the robustly
       closest hit */
    if ((t0 > prd.closestDist)
        ||
        ((t0 == prd.closestDist)
         &&
         (proxy.rankID >= prd.closestRank)
         ))
      return;

    /* seems we DO have a hit we wank to store - do so, and report one
       float-bit more distant to mmake sure we'll still get called for
       same-distance hits */
    prd.closestDist = t0;
    prd.closestRank = proxy.rankID;
    if (prd.dbg)
      printf("(%i) -> ACCEPTING hit %i dist %f\n",
             lp.rank,(int)prd.closestRank,prd.closestDist);
    float reported_t = nextafterf(t0,CUDART_INF);
    optixReportIntersection(reported_t,0);
  }

  inline __device__
  int NextDomainKernel::LPData::computeFirstRank(Ray &ray) const
  {
    auto &lp = LaunchParams::get();

    if (ray.dbg) printf("------------------ FIRST (%i)-------------\n",lp.rank);
    
    owl::Ray optix_ray(ray.origin,
                       ray.getDirection(),
                       0.f,ray.tMax);
    NextDomainKernel::PRD prd;
    prd.phase = NextDomainKernel::Phase_FindFirst;
    prd.closestDist = ray.tMax;
    prd.closestRank = -1;
    prd.dbg = ray.dbg;
    owl::traceRay(proxyBVH,optix_ray,prd);
    return prd.closestRank;
  }

  inline __device__
  int NextDomainKernel::LPData::computeNextRank(Ray &ray) const
  {
    auto &lp = LaunchParams::get();
    
    if (ray.dbg)
      printf("------------------ NEXT rank %i spawn %i-------------\n",
             lp.rank,ray.spawningRank);
    owl::Ray optix_ray(ray.origin,
                       ray.getDirection(),
                       0.f,nextafterf(ray.tMax,CUDART_INF));
    NextDomainKernel::PRD prd;
    prd.dbg = ray.dbg;
    prd.alreadyTravedMask.clearBits();
    prd.alreadyTravedMask.setBit(ray.spawningRank);
    if (ray.spawningRank != lp.rank) {
      // if we are NOT on the originating node

      // ------------------------------------------------------------------
      // phase 1: find ourselves - ie, find closest distance to any of
      // our proxies; that's the proxy that any _other_ rank would
      // have accepted to send this ray to us
      // ------------------------------------------------------------------

      if (ray.dbg) printf("------------------ NEXT, phase 1 -------------\n");
      optix_ray.tmax = CUDART_INF;
      prd.phase = NextDomainKernel::Phase_FindSelf;
      prd.closestRank = -1;
      prd.closestDist = CUDART_INF;
      if (ray.dbg)
        printf("tracing ray (%f %f %f)(%f %f %f) max_t %f\n",
               ray.origin.x,
               ray.origin.y,
               ray.origin.z,
               ray.getDirection().x,
               ray.getDirection().y,
               ray.getDirection().z,
               optix_ray.tmax);
      owl::traceRay(proxyBVH,optix_ray,prd);
      if (prd.closestRank != lp.rank) {
        if (ray.dbg)
          printf("baaad - ray couldn't find itself - found %i, should be %i\n",
                 prd.closestRank,lp.rank);
        return -1;
      }

      // ------------------------------------------------------------------
      // phase 2: trace ray again (up to ourselves), and mark all
      // those ranks whose proxies would have been accepted BEFORE
      // ours - those must be the ones that have already been
      // traversed by this ray
      // ------------------------------------------------------------------

      if (ray.dbg) printf("------------------ NEXT, phase 2 -------------\n");
      prd.phase = NextDomainKernel::Phase_FindOthers;
      prd.alreadyTravedMask.setBit(prd.closestRank);
      prd.closestRank = -1;
      //prd.closestDist = nextafterf(prd.closestDist,CUDART_INF);
      optix_ray.tmax = nextafterf(prd.closestDist,CUDART_INF);
      if (ray.dbg)
        printf("tracing ray max_t %f\n",optix_ray.tmax);
      owl::traceRay(proxyBVH,optix_ray,prd);
    }

    if (ray.dbg) printf("------------------ NEXT, phase 3 -------------\n");
    // ------------------------------------------------------------------
    // phase 3: find next closest proxy (up to ray.tmax) that belongs
    // on a rank that has _not_ yet been traversed
    // ------------------------------------------------------------------
    prd.phase = NextDomainKernel::Phase_FindNext;
    prd.closestRank = -1;
    prd.closestDist = ray.tMax;
    optix_ray.tmax  = ray.tMax;
      if (ray.dbg)
        printf("tracing ray max_t %f\n",optix_ray.tmax);
    owl::traceRay(proxyBVH,optix_ray,prd);
    
    return prd.closestRank;
  }



  // ##################################################################
  // UMeshVolume Shared-face Sampler Code
  // ##################################################################
  
  /*! closest-hit program for shared-faces geometry */
  OPTIX_CLOSEST_HIT_PROGRAM(UMeshGeomCH)()
  {
    const UMeshGeom &geom = owl::getProgramData<UMeshGeom>();
    int faceID = optixGetPrimitiveIndex();
    const vec2i &tetsOnFace = geom.tetsOnFace[faceID];
    int side   = optixIsFrontFaceHit();
    int tetID  = (&tetsOnFace.x)[side];
    if (tetID < 0)
      // outside face - no hit
      return;
  
    vec4i tet  = geom.tets[tetID];
    const vec3f P    = optixGetWorldRayOrigin();
    const vec3f A    = geom.vertices[tet.x] - P;
    const vec3f B    = geom.vertices[tet.y] - P;
    const vec3f C    = geom.vertices[tet.z] - P;
    const vec3f D    = geom.vertices[tet.w] - P;
    float fA = fabsf(dot(B,cross(C,D)));
    float fB = fabsf(dot(C,cross(D,A)));
    float fC = fabsf(dot(D,cross(A,B)));
    float fD = fabsf(dot(A,cross(B,C)));
    const float scale = 1.f/(fA+fB+fC+fD);
    fA *= scale;
    fB *= scale;
    fC *= scale;
    fD *= scale;
    auto &prd = owl::getPRD<UMeshVolume::SamplePRD>();
    prd.sampledValue
      = fA * geom.scalars[tet.x]
      + fB * geom.scalars[tet.y]
      + fC * geom.scalars[tet.z]
      + fD * geom.scalars[tet.w];
  }

  inline __device__
  vec3f backgroundColor(const Ray &ray)
  {
    auto &lp = LaunchParams::get();
    int iy = lp.fbLayer.indexToGlobal(ray.pixelID).y;
    float t = iy / float(lp.fbLayer.fullFbSize.y);
    const vec3f c = (1.0f - t)*vec3f(1.0f, 1.0f, 1.0f) + t * vec3f(0.5f, 0.7f, 1.0f);
    return c;
  }
  


  // ##################################################################
  // UNSORTED
  // ##################################################################

  
  inline __device__
  void traceAgainstLocalGeometry(Ray &path)
  {
    auto &lp = LaunchParams::get();
  }
    
  inline __device__
  void traceThroughLocalVolumeData(Ray &ray, Random &rng)
  {
    auto &lp = LaunchParams::get();
    bool dbg = ray.dbg;

    vec3f dda_org = xfmPoint(lp.mcGrid.worldToMcSpace,ray.getOrigin());
    vec3f dda_dir = xfmVector(lp.mcGrid.worldToMcSpace,ray.getDirection());

    vec3f ray_org = ray.getOrigin();
    vec3f ray_dir = ray.getDirection();
    float ray_t0=0,ray_t1=CUDART_INF;
    float dda_t0=0,dda_t1=CUDART_INF;
    boxTest(lp.volumeSampler.structured.dbg_domain,ray_org,ray_dir,ray_t0,ray_t1);
    boxTest(lp.volumeSampler.structured.dbg_domain,dda_org,dda_dir,dda_t0,dda_t1);
    if (0 && ray.dbg) {
      printf("(%i) dbg.domain %f %f %f  %f %f %f\n",
             lp.rank,
             lp.volumeSampler.structured.dbg_domain.lower.x,
             lp.volumeSampler.structured.dbg_domain.lower.y,
             lp.volumeSampler.structured.dbg_domain.lower.z,
             lp.volumeSampler.structured.dbg_domain.upper.x,
             lp.volumeSampler.structured.dbg_domain.upper.y,
             lp.volumeSampler.structured.dbg_domain.upper.z);
      printf("(%i) tracing dda ray (%f %f %f) + t *(%f %f %f) ray %f %f...dda %f %f\n",
             lp.rank,
             dda_org.x,
             dda_org.y,
             dda_org.z,
             dda_dir.x,
             dda_dir.y,
             dda_dir.z,
             ray_t0,ray_t1,
             dda_t0,dda_t1
             );
    }
    // if (ray.dbg) {
    // vec3f dda_p0 = dda_org + dda_t0 * dda_dir;
    // vec3f dda_p1 = dda_org + dda_t1 * dda_dir;
    // vec3f ray_p0 = ray_org + ray_t0 * ray_dir;
    // vec3f ray_p1 = ray_org + ray_t1 * ray_dir;
    // printf("dda - ray %f %f %f ...  %f %f %f\n",
    //        ray_p0.x,
    //        ray_p0.y,
    //        ray_p0.z,
    //        ray_p1.x,
    //        ray_p1.y,
    //        ray_p1.z);
    // printf("dda - dda %f %f %f ...  %f %f %f\n",
    //        dda_p0.x,
    //        dda_p0.y,
    //        dda_p0.z,
    //        dda_p1.x,
    //        dda_p1.y,
    //        dda_p1.z);
    // }

#if 1
    vec4f mapped_volume_at_t_hit = 0.f;
    const float DENSITY = ((lp.volumeSampler.xf.density == 0.f) ? 1.f : lp.volumeSampler.xf.density);//.03f;
    dda::dda3(dda_org,dda_dir,ray.tMax,
              vec3ui(lp.mcGrid.dims),
              [&](vec3i cellID,float t0, float t1)->bool {
                if (t0 >= ray.tMax)
                  return false;
                t1 = min(t1,ray.tMax);
                float majorant = lp.mcGrid.getMajorant(cellID);
                if (majorant == 0.f)
                  return t1 < ray.tMax;
                const float step_scale = 1.f/(majorant*DENSITY);

                float t = t0;
                while (true) {
                  // Sample a distance
                  t = t - (logf(1.f - rng()) * step_scale);
                  if (/*we left the cell: */t >= t1) 
                    /* leave this cell, but tell DDA to keep on going */
                    return t1 < ray.tMax;

                  vec3f P = ray.getOrigin()+t*ray.getDirection();
                  float f = CUDART_INF;
                  if (!lp.volumeSampler.structured.sample(f,P,dbg))
                    continue;
                  
                  mapped_volume_at_t_hit = lp.volumeSampler.xf.map(f);
                  if (rng()*majorant >= mapped_volume_at_t_hit.w)
                    // reject this sample
                    continue;
                  
                  // we DID sample the volume!
                  ray.tMax = t;
                  ray.hitType = Ray::HitType_Volume;
                  ray.hit.volume.color = to_half({ mapped_volume_at_t_hit.x,
                                                   mapped_volume_at_t_hit.y,
                                                   mapped_volume_at_t_hit.z });
                  return /* false == "we're done" !! */ false;
                }
              },
              dbg);
    
#else
    vec3f color = 0.f;
    float opacity = 0.f;
    
    dda::dda3(dda_org,dda_dir,ray.tMax,
              vec3ui(lp.mcGrid.dims),
              [&](vec3i cellID,float t0, float t1)->bool {
                if (0 && dbg)
                  printf("in mc cell %i %i %i range %f %f\n",cellID.x,cellID.y,cellID.z,t0,t1);

                float dt = 3.f;
                for (float t = t0+rng()*dt; true; t += dt) {
                  vec3f P = ray.getOrigin()+t*ray.getDirection();
                  float f = CUDART_INF;
                  if (t >= t1) {
                    if (0 && dbg) printf("exit cell at t %f\n",t);
                    break;
                  }
                  bool valid = lp.volumeSampler.structured.sample(f,P,dbg);
                  if (0 && dbg) printf("t %f P %f %f %f valid %i\n",t,
                                  P.x,P.y,P.z,int(valid));
                  
                  if (!valid)
                    continue;
                  
                  vec4f mapped
                    = lp.volumeSampler.xf.map(f);
                  color += (1.f-opacity) * mapped.w * vec3f(mapped.x,mapped.y,mapped.z);
                  opacity += (1.f-opacity)*mapped.w;
                  opacity = min(opacity,1.f);
                    if (0 && dbg)
                    printf("volume %f -> (%f %f %f ; %f)\n",f,mapped.x,mapped.y,mapped.z,mapped.w);
                }
                
                return true;
              },
              dbg);
    lp.fbLayer.addPixelContribution(ray.pixelID,opacity * color);
#endif
  }
    
  OPTIX_RAYGEN_PROGRAM(traceLocallyRG)()
  {
    auto &lp = LaunchParams::get();
    
    int rayID = owl::getLaunchIndex().x;

    Ray ray = lp.forwardGlobals.rayQueueIn[rayID];
    Random rng(ray.pixelID,0x2345678 + (lp.rank+lp.sampleID) * FNV_PRIME);
    traceRayLocally(rng,ray);
  } 

  inline __device__
  Ray generateRay(vec2i pixelID,
                  vec2f pixelPos)
  {
    auto &lp = LaunchParams::get();
    Ray ray;
    ray.pixelID  = lp.fbLayer.globalToIndex(pixelID);
    ray.isShadow = false;
    ray.hitType  = Ray::HitType_None;
    auto &camera = lp.camera;
    ray.origin = camera.lens_00;
    vec3f dir
      = camera.dir_00
      + camera.dir_du * (pixelID.x+pixelPos.x)
      + camera.dir_dv * (pixelID.y+pixelPos.y);
    ray.setDirection(dir);
    ray.throughput = to_half(vec3f(1.f));
    return ray;
  }

  OPTIX_RAYGEN_PROGRAM(generatePrimaryWaveRG)()
  {
    auto &lp = LaunchParams::get();
    const vec2i pixelID = lp.fbLayer.localToGlobal(owl::getLaunchIndex());//owl::getLaunchIndex();
    Random rng(pixelID.x+FNV_PRIME*pixelID.y,0x1234567 + FNV_PRIME * lp.sampleID);

    if (pixelID.x >= lp.fbLayer.fullFbSize.x ||
        pixelID.y >= lp.fbLayer.fullFbSize.y)
      return;
        
    vec2f pixelSample = { rng(), rng() };//.5f;

    Ray ray = generateRay(pixelID,pixelSample);
    auto &fullFbSize = lp.fbLayer.fullFbSize;

    //vec2i dbgPixel(540,1016-600);
    vec2i dbgPixel = fullFbSize/2;
    ray.dbg = 0;//(pixelID == dbgPixel);

    
    ray.crosshair
      = !ray.dbg && ((pixelID.x == dbgPixel.x) || (pixelID.y == dbgPixel.y));
    ray.spawningRank = lp.rank;
    int pixelOwner = lp.nextDomainKernel.computeFirstRank(ray);
#define VISUALIZE_PROXIES 0
#if VISUALIZE_PROXIES
    vec3f color
      = (pixelOwner == -1)
      ? abs(normalize(ray.getDirection()))
      : randomColor(pixelOwner);
    if (lp.rank == 0) 
      lp.fbLayer.addPixelContribution(ray.pixelID,color);
    return;
#endif
    if (pixelOwner == -1 && lp.rank == 0) {
      // pixel not owned by anybody; let's do background on rank 0
      vec3f frag = backgroundColor(ray);
      if (ray.crosshair) frag = 1.f - frag;
      lp.fbLayer.addPixelContribution(ray.pixelID,frag);
      return;
    }

    if (pixelOwner != lp.rank)
      // we don't own the closest proxy - let somebody else deal with it ...
      return;

#if 0
    // for ray generation we "forward" all rays to ourselves:
    lp.forwardGlobals.forwardRay(ray,lp.rank);
#else
    traceRayLocally(rng,ray);
#endif
    // lp.fbLayer.addPixelContribution(vec2i(pixelID.x,pixelID.y),abs(from_half(ray.direction)));
    // generatePrimaryWaveKernel
    //   (launchIdx,
    //    lp.forwardGlobals,
    //    lp.volumeGlobals);
  }
  



  inline __device__
  void traceRayLocally(Random &rng, Ray &ray)
  {
    auto &lp = LaunchParams::get();

    // if (!ray.dbg) return;

    traceAgainstLocalGeometry(ray);
    if (ray.isShadow && ray.hitType != Ray::HitType_None)
      // this shadow ray is occluded - let it drop dead.
      return;

    traceThroughLocalVolumeData(ray,rng);
    if (ray.isShadow && ray.hitType != Ray::HitType_None)
      // this shadow ray is occluded - let it drop dead.
      return;

    if (ray.dbg)
      printf("ray hit type %i at %f\n",
             int(ray.hitType),ray.tMax);
    
    // ==================================================================
    // check if ray needs futher processing on another node
    // ==================================================================
    int nextRankToSendTo = lp.nextDomainKernel.computeNextRank(ray);
    if (ray.dbg)
      printf("next rank: %i\n",nextRankToSendTo);
    if (nextRankToSendTo >= 0) {
      if (ray.dbg)
        printf("---> forwarding to %i\n",nextRankToSendTo);
      if (nextRankToSendTo == lp.rank) {
        printf("forwarding to ourselves!?\n");
        return;
      }
      lp.forwardGlobals.forwardRay(ray,nextRankToSendTo);
      return;
    }

    // ==================================================================
    // ray doesn't need forwarding; it's done and can be shaded here!
    // ==================================================================
    if (ray.isShadow) {
      // if we reach here we cannot have had any occlusion, else this
      // ray would ahve dies already...
      lp.fbLayer.addPixelContribution(ray.pixelID,from_half(ray.throughput));
      return;
    }

    if (ray.hitType == Ray::HitType_Volume) {
#if 1
      float ambient = .1f;
      vec3f frag = from_half(ray.throughput)*from_half(ray.hit.volume.color);
      if (ray.dbg)
        printf("(%i) adding frag %f %f %f\n",
               lp.rank,frag.x,frag.y,frag.z);
      lp.fbLayer.addPixelContribution(ray.pixelID,ambient * frag);

      ray.throughput = to_half(frag*(1.f-ambient));
      ray.isShadow = 1;
      ray.spawningRank = lp.rank;
      ray.hitType = Ray::HitType_None;
      vec3f lightDir = normalize(vec3f(1.f,1.f,1.f));
      ray.setOrigin(ray.getOrigin()
                    +ray.tMax*ray.getDirection()
                    +.1f*lightDir);
      ray.setDirection(lightDir);
      ray.tMax = CUDART_INF;
      lp.forwardGlobals.forwardRay(ray,lp.rank);
#else
      // this was a volume hit; let's just store that color
      vec3f frag = from_half(ray.throughput)*from_half(ray.hit.volume.color);
      if (ray.dbg)
        printf("(%i) adding frag %f %f %f\n",
               lp.rank,frag.x,frag.y,frag.z);
      lp.fbLayer.addPixelContribution(ray.pixelID,frag);
#endif
    }
    else if (ray.hitType == Ray::HitType_None) {
      // ray didn't hit anything; but we KNOW it's not a shadow ray -
      // see do abckground
      vec3f frag = backgroundColor(ray);
      if (ray.crosshair) frag = 1.f - frag;
      lp.fbLayer.addPixelContribution(ray.pixelID,frag);
    }
  }
  
}
