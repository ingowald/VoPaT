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
#include "vopat/surface/MeshGeom.h"
#include "vopat/volume/UMeshVolume.h"

using namespace vopat;

// extern "C" __constant__ uint8_t optixLaunchParams[sizeof(LaunchParams)];
extern "C" __constant__ LaunchParams optixLaunchParams;

namespace vopat {

  inline __device__ const LaunchParams &LaunchParams::get()
  { return optixLaunchParams; }

  inline __device__
  void traceRayLocally(Random &rng, Ray &ray);

  inline __device__ float aLittleBitBiggerThan(float f)
  {
    return CUDART_INF;
    return f * (1.f+1e-3f);
    // return f * (1.f+1e-4f);
    // return nextafterf(f,CUDART_INF);
  }
                                         
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

  OPTIX_ANY_HIT_PROGRAM(proxyAH)()
  {}

  OPTIX_CLOSEST_HIT_PROGRAM(proxyCH)()
  {}

  OPTIX_INTERSECT_PROGRAM(proxyIsec)()
  {
    auto &lp   = LaunchParams::get();
    auto &geom = owl::getProgramData<NextDomainKernel::Geom>();
    int primID = optixGetPrimitiveIndex();
    auto proxy = geom.proxies[primID];

    auto &prd = owl::getPRD<NextDomainKernel::PRD>();

    bool dbg_next = true;//false;
    if (dbg_next &&prd.dbg)
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
             prd.alreadyTravedMask.hasBitSet(proxy.rankID)) {
      if (dbg_next &&prd.dbg)
        printf("(%i) skipping proxy rank %i that already has a bit set...\n",
               lp.rank,proxy.rankID);
      return;
    }

    vec3f org = optixGetWorldRayOrigin();
    vec3f dir = optixGetWorldRayDirection();
    float t0 = optixGetRayTmin();
    float t1 = optixGetRayTmax();
    if (!boxTest(proxy.domain,org,dir,t0,t1,dbg_next && prd.dbg)) {
      // we didn't even hit this proxy, so everything else is moot;
      // exit.
      if (dbg_next &&prd.dbg)
        printf("(%i) missed... %f %f\n",lp.rank,t0,t1);
      return;
    }

    if (dbg_next && prd.dbg)
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
      if (dbg_next &&prd.dbg)
        printf(" others proxy.t %f prd.t %f proxy.rank %i prd.rank %i\n",
               t0,prd.closestDist,proxy.rankID,prd.closestRank);
      if (t0 < prd.closestDist ||
          t0 == prd.closestDist && proxy.rankID < prd.closestRank) {
        /* this hit WOULD have been accepted if we hadn't explicitly
           included it in the find-self phase */
        prd.alreadyTravedMask.setBit(proxy.rankID);
        if (dbg_next &&prd.dbg)
          printf(" -> DID set bit, mask now  0x%08x\n",int(prd.alreadyTravedMask.qwords[0]));
      } else {
        if (dbg_next &&prd.dbg)
          printf(" -> did NOT set bit!\n");
      }
        
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

    /* seems we DO have a hit we want to store - do so, and report one
       float-bit more distant to mmake sure we'll still get called for
       same-distance hits */
    prd.closestDist = t0;
    prd.closestRank = proxy.rankID;
    float reported_t = aLittleBitBiggerThan(t0);
    if (dbg_next &&prd.dbg)
      printf("(%i) -> ACCEPTING hit %i dist %f w/ reported dist %f\n",
             lp.rank,(int)prd.closestRank,prd.closestDist,reported_t);
    // float reported_t = nextafterf(t0,CUDART_INF);
    optixReportIntersection(reported_t,0);
  }

  inline __device__
  int NextDomainKernel::LPData::computeFirstRank(Ray &ray) const
  {
    auto &lp = LaunchParams::get();

    if (0 && ray.dbg) printf("------------------ FIRST (%i)-------------\n",lp.rank);
    
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

    bool dbg_next = true; //false;
    
    if (dbg_next && ray.dbg)
      printf("------------------ NEXT (on rank %i, spawn %i)-------------\n",
             lp.rank,ray.spawningRank);
    owl::Ray optix_ray(ray.origin,
                       ray.getDirection(),
                       0.f,aLittleBitBiggerThan(ray.tMax));
    NextDomainKernel::PRD prd;
    prd.dbg = ray.dbg;
    prd.alreadyTravedMask.clearBits();
    prd.alreadyTravedMask.setBit(ray.spawningRank);
    if (dbg_next && ray.dbg)
      printf("already trav mask INIT 0x%08x\n",int(prd.alreadyTravedMask.qwords[0]));
    if (ray.spawningRank != lp.rank) {
      // if we are NOT on the originating node

      // ------------------------------------------------------------------
      // phase 1: find ourselves - ie, find closest distance to any of
      // our proxies; that's the proxy that any _other_ rank would
      // have accepted to send this ray to us
      // ------------------------------------------------------------------

      if (dbg_next && ray.dbg)
        printf("------------------ NEXT, phase 1 (find self, self being %i) -------------\n",lp.rank);
      optix_ray.tmin = 0.f;
      optix_ray.tmax = CUDART_INF;
      prd.phase = NextDomainKernel::Phase_FindSelf;
      prd.closestRank = -1;
      prd.closestDist = CUDART_INF;
      if (dbg_next && ray.dbg)
        printf("tracing ray (%f %f %f)(%f %f %f) max_t %f\n",
               optix_ray.origin.x,
               optix_ray.origin.y,
               optix_ray.origin.z,
               optix_ray.direction.x,
               optix_ray.direction.y,
               optix_ray.direction.z,
               // ray.origin.x,
               // ray.origin.y,
               // ray.origin.z,
               // ray.getDirection().x,
               // ray.getDirection().y,
               // ray.getDirection().z,
               optix_ray.tmax);
      owl::traceRay(proxyBVH,optix_ray,prd);
      if (dbg_next && ray.dbg)
        printf("already trav mask FRST 0x%08x\n",int(prd.alreadyTravedMask.qwords[0]));
      if (prd.closestRank != lp.rank) {
        if (dbg_next && ray.dbg)
          printf("baaad - ray couldn't find itself - found %i, should be %i\n",
                 prd.closestRank,lp.rank);
        return -1;
      }
      const float distanceToSelf = prd.closestDist;

      // ------------------------------------------------------------------
      // phase 2: trace ray again (up to ourselves), and mark all
      // those ranks whose proxies would have been accepted BEFORE
      // ours - those must be the ones that have already been
      // traversed by this ray
      // ------------------------------------------------------------------

      if (dbg_next && ray.dbg) {
        printf("------------------ NEXT, phase 2, find OTHERS before us -------------\n");
        printf("now adding rank %i\n",prd.closestRank);
      }
#if 0
      NextDomainKernel::PRD prd2;
      prd2.dbg = ray.dbg;
      prd2.phase = NextDomainKernel::Phase_FindOthers;
      prd2.closestRank = lp.rank;
      prd2.closestDist = distanceToSelf;
      prd2.alreadyTravedMask.clearBits();
      prd2.alreadyTravedMask.setBit(lp.rank);
      prd2.alreadyTravedMask.setBit(ray.spawningRank);
      owl::Ray optix_ray2(ray.origin,
                          ray.getDirection(),
                          0.f,aLittleBitBiggerThan(distanceToSelf));
      if (dbg_next && ray.dbg)
        printf("already trav mask BEFR 0x%08x\n",int(prd2.alreadyTravedMask.qwords[0]));
      if (dbg_next && ray.dbg)
        printf("tracing ray (%f %f %f)(%f %f %f) max_t %f\n",
               optix_ray2.origin.x,
               optix_ray2.origin.y,
               optix_ray2.origin.z,
               optix_ray2.direction.x,
               optix_ray2.direction.y,
               optix_ray2.direction.z,
               // ray.origin.x,
               // ray.origin.y,
               // ray.origin.z,
               // ray.getDirection().x,
               // ray.getDirection().y,
               // ray.getDirection().z,
               optix_ray2.tmax);
      if (dbg_next && ray.dbg) {
        printf("tracing ray max_t %f stored hit %i @ %f\n",optix_ray2.tmax,
               prd2.closestRank,
               prd2.closestDist);
      }
      owl::traceRay(proxyBVH,optix_ray2,prd2);
      prd.alreadyTravedMask = prd2.alreadyTravedMask;
#else
      prd.phase = NextDomainKernel::Phase_FindOthers;
      prd.alreadyTravedMask.setBit(prd.closestRank);
      prd.closestRank = lp.rank;
      prd.closestDist = distanceToSelf;
      prd.dbg = ray.dbg;
      optix_ray.tmax = aLittleBitBiggerThan(distanceToSelf);
      if (dbg_next && ray.dbg)
        printf("already trav mask BEFR 0x%08x\n",int(prd.alreadyTravedMask.qwords[0]));
      if (dbg_next && ray.dbg)
        printf("tracing ray (%f %f %f)(%f %f %f) max_t %f\n",
               optix_ray.origin.x,
               optix_ray.origin.y,
               optix_ray.origin.z,
               optix_ray.direction.x,
               optix_ray.direction.y,
               optix_ray.direction.z,
               // ray.origin.x,
               // ray.origin.y,
               // ray.origin.z,
               // ray.getDirection().x,
               // ray.getDirection().y,
               // ray.getDirection().z,
               optix_ray.tmax);
      if (dbg_next && ray.dbg) {
        printf("tracing ray max_t %f stored hit %i @ %f\n",optix_ray.tmax,
               prd.closestRank,
               prd.closestDist);
      }
      owl::traceRay(proxyBVH,optix_ray,prd);
#endif
      if (dbg_next && ray.dbg)        
        printf("already trav mask AFTR 0x%08x\n",int(prd.alreadyTravedMask.qwords[0]));
    }

    if (dbg_next && ray.dbg)
      printf("------------------ NEXT, phase 3 -------------\n");
    // ------------------------------------------------------------------
    // phase 3: find next closest proxy (up to ray.tmax) that belongs
    // on a rank that has _not_ yet been traversed
    // ------------------------------------------------------------------
    
    NextDomainKernel::PRD prd3;
    prd3.alreadyTravedMask = prd.alreadyTravedMask;
    prd3.phase = NextDomainKernel::Phase_FindNext;
    prd3.dbg = prd.dbg;
    prd3.closestRank = -1;
    prd3.closestDist = ray.tMax;
    owl::Ray optix_ray3(ray.origin,
                        ray.getDirection(),
                        0.f,ray.tMax);
    if (dbg_next && ray.dbg)
      printf("tracing ray max_t %f\n",optix_ray.tmax);
    owl::traceRay(proxyBVH,optix_ray3,prd3
                  // ,
                  // OPTIX_RAY_FLAG_DISABLE_ANYHIT
                  //  // | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT
                  //  | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT
                  );
    
    return prd3.closestRank;
  }

  // ##################################################################
  // Surface Mesh Geometry Code
  // ##################################################################
  
  /*! closest-hit program for triangle mesh surface geometry */
  OPTIX_CLOSEST_HIT_PROGRAM(MeshGeomCH)()
  {
    const MeshGeom::DD &geom = owl::getProgramData<MeshGeom::DD>();
    MeshGeom::PRD &prd = owl::getPRD<MeshGeom::PRD>();
    int primID = optixGetPrimitiveIndex();
    vec3i indices = geom.indices[primID];
    
    vec3f v0 = geom.vertices[indices.x];
    vec3f v1 = geom.vertices[indices.y];
    vec3f v2 = geom.vertices[indices.z];
    
    v0 = optixTransformPointFromObjectToWorldSpace(v0);
    v1 = optixTransformPointFromObjectToWorldSpace(v1);
    v2 = optixTransformPointFromObjectToWorldSpace(v2);
      
    prd.t = optixGetRayTmax();
    prd.diffuseColor = geom.diffuseColor;
    prd.N = normalize(cross(v1-v0,v2-v0));
  }

  /*! closest-hit program for triangle mesh surface geometry */
  OPTIX_ANY_HIT_PROGRAM(MeshGeomAH)()
  {
  }

  // ##################################################################
  // UMeshVolume Shared-face Sampler Code
  // ##################################################################

#if UMESH_SHARED_FACES
  /*! closest-hit program for shared-faces geometry */
  OPTIX_CLOSEST_HIT_PROGRAM(UMeshGeomCH)()
  {
    const UMeshVolume::Geom &geom = owl::getProgramData<UMeshVolume::Geom>();
    int faceID = optixGetPrimitiveIndex();
    
    const vec2i &tetsOnFace = geom.tetsOnFace[faceID];
    int side   = optixIsFrontFaceHit();
    int tetID  = (&tetsOnFace.x)[side];
    if (tetID < 0)
      // outside face - no hit
      return;
    vec4i tet  = (vec4i&)geom.tets[tetID];
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
#else
  OPTIX_BOUNDS_PROGRAM(UMeshGeomBounds)(const void  *geomData,
                                        box3f       &primBounds,
                                        const int    primID)
  {
    const UMeshVolume::Geom &geom = *(const UMeshVolume::Geom*)geomData;
    vec4i tet = (const vec4i&)geom.tets[primID];
    primBounds = box3f()
      .including(geom.vertices[tet.x])
      .including(geom.vertices[tet.y])
      .including(geom.vertices[tet.z])
      .including(geom.vertices[tet.w]);
  }

  /*! closest-hit program for shared-faces geometry */
  OPTIX_INTERSECT_PROGRAM(UMeshGeomIS)()
  {
    const UMeshVolume::Geom &geom = owl::getProgramData<UMeshVolume::Geom>();
    int tetID = optixGetPrimitiveIndex();
    vec4i tet  = (vec4i&)geom.tets[tetID];
    const vec3f P    = optixGetWorldRayOrigin();
    const vec3f A    = geom.vertices[tet.x] - P;
    const vec3f B    = geom.vertices[tet.y] - P;
    const vec3f C    = geom.vertices[tet.z] - P;
    const vec3f D    = geom.vertices[tet.w] - P;
                                // vec3i{ A, C, B },
                                // vec3i{ A, D, C },
                                // vec3i{ A, B, D },
                                // vec3i{ B, C, D }
    float fD = (dot(A,cross(C,B)));
    float fB = (dot(A,cross(D,C)));
    float fC = (dot(A,cross(B,D)));
    float fA = (dot(B,cross(C,D)));

    float min_f = min(min(fA,fB),min(fC,fD));
    if (min_f * fA < 0.f) return;
    if (min_f * fB < 0.f) return;
    if (min_f * fC < 0.f) return;
    if (min_f * fD < 0.f) return;

    fA = fabsf(fA);
    fB = fabsf(fB);
    fC = fabsf(fC);
    fD = fabsf(fD);

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
    optixReportIntersection(0,0);
  }
  OPTIX_ANY_HIT_PROGRAM(UMeshGeomAH)()
  { optixTerminateRay(); }
  OPTIX_CLOSEST_HIT_PROGRAM(UMeshGeomCH)()
  {}
#endif
  
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
  void traceAgainstLocalGeometry(Ray &ray)
  {
  }
  
  inline __device__
  void traceAgainstReplicatedGeometry(Ray &ray)
  {
    auto &lp = LaunchParams::get();
    if (!lp.replicatedSurfaceBVH) return;
    
    owl::Ray optix_ray(ray.origin,
                       ray.getDirection(),
                       0.f,ray.tMax);
    
    MeshGeom::PRD prd;
    prd.t = CUDART_INF;
    
    owl::traceRay(lp.replicatedSurfaceBVH,optix_ray,prd);
    
    if (prd.t < CUDART_INF) {
      ray.hitType = Ray::HitType_Surf_Diffuse;
      ray.tMax    = prd.t;
      ray.hit.surf_diffuse.color = to_half(prd.diffuseColor);
      ray.hit.surf_diffuse.N     = to_half(prd.N);
    }
  }
    
  inline __device__
  void traceThroughLocalVolumeData(Ray &ray, Random &rng)
  {
    auto &lp = LaunchParams::get();
    bool dbg = 0; //ray.dbg;

    vec3f dda_org = xfmPoint(lp.mcGrid.worldToMcSpace,ray.getOrigin());
    vec3f dda_dir = xfmVector(lp.mcGrid.worldToMcSpace,ray.getDirection());

    vec3f ray_org = ray.getOrigin();
    vec3f ray_dir = ray.getDirection();

    vec4f mapped_volume_at_t_hit = 0.f;
    const float DENSITY = ((lp.volumeSampler.xf.density == 0.f) ? 1.f : lp.volumeSampler.xf.density);//.03f;
    dda::dda3(dda_org,dda_dir,ray.tMax,
              vec3ui(lp.mcGrid.dims),
              [&](vec3i cellID,float t0, float t1)->bool {
                if (dbg)
                  printf("in mc cell %i %i %i (of %i %i %i) range %f %f\n",
                         cellID.x,cellID.y,cellID.z,
                         lp.mcGrid.dims.x,
                         lp.mcGrid.dims.y,
                         lp.mcGrid.dims.z,
                         t0,t1);
                
                if (t0 >= ray.tMax)
                  return false;
                t1 = min(t1,ray.tMax);
                float majorant = lp.mcGrid.getMajorant(cellID);
                if (majorant == 0.f)
                  return t1 < ray.tMax;
                const float step_scale = 1.f/(majorant*DENSITY);
                
                if (dbg)
                  printf(" -> majorant %f\n",majorant);

                float t = t0;
                while (true) {
                  // Sample a distance
                  float r = rng();
                  t = t - (logf(1.f - r) * step_scale);
                  if (dbg) printf("  -> rng %f dt %f t %f\n",
                                  r,logf(1.f-r),t);
                  if (/*we left the cell: */t >= t1) 
                    /* leave this cell, but tell DDA to keep on going */
                    return t1 < ray.tMax;

                  vec3f P = ray.getOrigin()+t*ray.getDirection();
                  float f = CUDART_INF;

                  if (!lp.volumeSampler.sample(f,P,dbg)) {
                    if (1 && dbg) printf("-> volume miss!\n");
                    continue;
                  }
                  
                  mapped_volume_at_t_hit = lp.volumeSampler.xf.map(f);
                  if (1 && dbg) printf("-> volume at %f %f %f is xf %f -> %f!\n",
                                       P.x,P.y,P.z,f,mapped_volume_at_t_hit.w);
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
  }
    
  OPTIX_RAYGEN_PROGRAM(traceLocallyRG)()
  {
    auto &lp = LaunchParams::get();
    
    int rayID = owl::getLaunchIndex().x;

    Ray ray = lp.forwardGlobals.rayQueueIn[rayID];
    Random rng((ray.pixelID*2+ray.isShadow)*16+ray.numBounces,
               0x2345678 + (lp.rank*13+lp.sampleID));
    // for (int i=0;i<10;i++) rng();
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
    Random rng(pixelID.x+FNV_PRIME*pixelID.y,0x1234567 + 13 * 17 * lp.sampleID);
    for (int i=0;i<10;i++) rng();

    if (pixelID.x >= lp.fbLayer.fullFbSize.x ||
        pixelID.y >= lp.fbLayer.fullFbSize.y)
      return;
        
    vec2f pixelSample = { rng(), rng() };//.5f;

    Ray ray = generateRay(pixelID,pixelSample);
    auto &fullFbSize = lp.fbLayer.fullFbSize;

    //vec2i dbgPixel(540,1016-600);
    vec2i dbgPixel = fullFbSize/2;
    ray.dbg = (pixelID == dbgPixel);
    // ray.dbg = (ray.pixelID == 540106);
    ray.dbg = 0;

    ray.crosshair
      = !ray.dbg && ((pixelID.x == dbgPixel.x) || (pixelID.y == dbgPixel.y));
    ray.spawningRank = lp.rank;
    ray.numBounces = 0;
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

    // if we have replicated geometry we MAY have some intersection
    // with a ray even if there's no proxy - so even if the proxies
    // didn't 'claim' a pixel owner we still have to assign one so the
    // ray does get traced "somewhere". If we do NOT have replicated
    // geometry that means we'll go through all the pain of tracing a
    // ray locally that'll never hit any geometry, but that'll still
    // fall back to background color, so all's fine.
    if (pixelOwner == -1)
      pixelOwner = 0;

    if (pixelOwner != lp.rank)
      // we don't own the closest proxy - let somebody else deal with it ...
      return;

    traceRayLocally(rng,ray);
  }
  



  inline __device__
  void traceRayLocally(Random &rng, Ray &ray)
  {
    auto &lp = LaunchParams::get();

    // if (!ray.dbg) return;

    if (ray.spawningRank == lp.rank)
      traceAgainstReplicatedGeometry(ray);
    traceAgainstLocalGeometry(ray);
    
    if (ray.isShadow && ray.hitType != Ray::HitType_None)
      // this shadow ray is occluded - let it drop dead.
      return;

    traceThroughLocalVolumeData(ray,rng);
    if (ray.isShadow && ray.hitType != Ray::HitType_None) {
      if (ray.dbg) printf("shadow ray died OCCLUDED. done with it\n");
      // this shadow ray is occluded - let it drop dead.
      return;
    }

    if (ray.dbg)
      printf("(%i) ray hit type %i at %f\n",
             lp.rank,
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
      if (lp.emergency > 20) {
        printf("emergency ray: %i\n",ray.pixelID);
        ray.dbg = true;
      }
      lp.forwardGlobals.forwardRay(ray,nextRankToSendTo);
      return;
    }

    // ==================================================================
    // ray doesn't need forwarding; it's done and can be shaded here!
    // ==================================================================
    if (ray.isShadow) {
      vec3f frag = from_half(ray.throughput);
      if (ray.dbg) printf("shadow ray died UN-occluded -> adding frag %f %f %f\n",
                          frag.x,frag.y,frag.z);
      // if we reach here we cannot have had any occlusion, else this
      // ray would have died already...
      lp.fbLayer.addPixelContribution(ray.pixelID,frag);
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
      if (ray.dbg) {
        printf("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n");
        printf("SHADOW ray at %f %f %f\n",
               ray.getOrigin().x,
               ray.getOrigin().y,
               ray.getOrigin().z);
      }
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
    else if (ray.hitType == Ray::HitType_Surf_Diffuse) {
      // ray didn't hit anything; but we KNOW it's not a shadow ray -
      // see do abckground
      vec3f N = from_half(ray.hit.surf_diffuse.N);
      vec3f rd = from_half(ray.hit.surf_diffuse.color);
      float scale = .2f+.8f*fabsf(dot(ray.getDirection(),N));
      vec3f frag = from_half(ray.throughput) * scale * rd;
      
#if 1
      float ambient = .1f;
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
      if (ray.crosshair) frag = 1.f - frag;
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
