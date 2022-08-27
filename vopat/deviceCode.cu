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
#include "vopat/NodeRenderer.h"
#include "vopat/NextDomainKernel.h"
#include <cuda.h>

using namespace vopat;

extern "C" __constant__ uint8_t optixLaunchParams[sizeof(LaunchParams)];

namespace vopat {
  
  inline __device__ const LaunchParams &getLP()
  { return (const LaunchParams &)optixLaunchParams[0]; }


  inline __device__
  Ray generateRay(const ForwardGlobals &globals,
                  vec2i pixelID,
                  vec2f pixelPos)
  {
    Ray ray;
    ray.pixelID  = pixelID.x + globals.worldFbSize.x*pixelID.y;
    ray.isShadow = false;
    ray.origin = globals.camera.lens_00;
    vec3f dir
      = globals.camera.dir_00
      + globals.camera.dir_du * (pixelID.x+pixelPos.x)
      + globals.camera.dir_dv * (pixelID.y+pixelPos.y);
    ray.setDirection(dir);
    ray.throughput = to_half(vec3f(1.f));
    return ray;
  }

  inline __device__
  vec3f backgroundColor(const Ray &ray,
                        const ForwardGlobals &globals)
  {
    int iy = ray.pixelID / globals.worldFbSize.x;
    float t = iy / float(globals.worldFbSize.y);
    const vec3f c = (1.0f - t)*vec3f(1.0f, 1.0f, 1.0f) + t * vec3f(0.5f, 0.7f, 1.0f);
    return c;
  }
  

  inline __device__
  void generatePrimaryWaveKernel(const vec2i launchIdx,
                                 const ForwardGlobals &vopat,
                                 const VolumeGlobals &globals)
  {
    int ix = launchIdx.x;
    int iy = launchIdx.y;

    if (vec2i(ix,iy) == vec2i(0))
      printf("launch 0 : islandsize %i %i \n",
             vopat.islandFbSize.x,vopat.islandFbSize.y);
             
    if (ix >= vopat.islandFbSize.x) return;
    if (iy >= vopat.islandFbSize.y) return;

    bool dbg = vec2i(ix,iy) == vopat.islandFbSize/2;
    if (dbg) printf("=======================================================\ngeneratePrimaryWaveKernel %i %i\n",ix,iy);
    
    int myRank = vopat.islandRank;//myRank;
    int world_iy
      = vopat.islandIndex
      + iy * vopat.islandCount;
    Ray ray    = generateRay(vopat,vec2i(ix,world_iy),vec2f(.5f));
#if 0
    ray.dbg    = (vec2i(ix,world_iy) == vopat.worldFbSize/2);
    if (ray.dbg) printf("----------- NEW RAY -----------\n");
#else
    ray.dbg    = false;
#endif

    ray.numBounces = 0;
#if DEBUG_FORWARDS
    ray.numFwds = 0;
#endif
    ray.crosshair = (ix == vopat.worldFbSize.x/2) || (world_iy == vopat.worldFbSize.y/2);
    int dest   = computeInitialRank(globals,ray);

    if (dest < 0) {
      /* "nobody" owns this pixel, set to background on rank 0 */
      if (myRank == 0) {
        // vopat.accumBuffer[islandPixelID(vopat,ray.pixelID)] += DeviceKernels::backgroundColor(ray,vopat);
        vopat.addPixelContribution(ray.pixelID,backgroundColor(ray,vopat));
      }
      return;
    }
    if (dest != myRank) {
      /* somebody else owns this pixel; we don't do anything */
      return;
    }
    int queuePos = atomicAdd(&vopat.perRankSendCounts[myRank],1);
    
    if (queuePos >= vopat.islandFbSize.x*vopat.islandFbSize.y)
      printf("FISHY PRIMARY RAY POS!\n");
    
    vopat.rayQueueIn[queuePos] = ray;
    if (!checkOrigin(ray))
      printf("fishy primary ray!\n");
  }
  
  OPTIX_BOUNDS_PROGRAM(proxyBounds)(const void  *geomData,
                                    box3f       &primBounds,
                                    const int    primID)
  {
    NextDomainKernel::DD &geom = *(NextDomainKernel::DD*)geomData;
    auto &proxy = geom.proxies[primID];
    primBounds = proxy.domain;
  }

  OPTIX_INTERSECT_PROGRAM(proxyIsec)()
  {
    auto &geom = owl::getProgramData<NextDomainKernel::DD>();
    int primID = optixGetPrimitiveIndex();
    auto proxy = geom.proxies[primID];

    auto &prd = owl::getPRD<NextDomainKernel::PRD>();
    if (prd.skipCurrentRank && prd.currentRank == proxy.rankID) return;
  
    vec3f org = optixGetWorldRayOrigin();
    vec3f dir = optixGetWorldRayDirection();
    float t0 = optixGetRayTmin();
    float t1 = optixGetRayTmax();
    if (!boxTest(proxy.domain,org,dir,t0,t1))
      return;
    if ((t0 > prd.closestDist)
        ||
        ((t0 == prd.closestDist)
         &&
         (proxy.rankID >= prd.closestRank)
         ))
      return;

    prd.closestDist = t0;
    prd.closestRank = prd.currentRank;
    float reported_t = nextafterf(t0,CUDART_INF);
    optixReportIntersection(reported_t,0);
  }

#if VOPAT_UMESH
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
    auto &prd = owl::getPRD<UMeshSamplePRD>();
    prd.sampledValue
      = fA * geom.scalars[tet.x]
      + fB * geom.scalars[tet.y]
      + fC * geom.scalars[tet.z]
      + fD * geom.scalars[tet.w];
  }
#endif

  OPTIX_RAYGEN_PROGRAM(generatePrimaryWaveRG)()
  {
    auto &lp = getLP();
    const vec2i launchIdx = owl::getLaunchIndex();
    generatePrimaryWaveKernel
      (launchIdx,
       lp.forwardGlobals,
       lp.volumeGlobals);
  }

  // OPTIX_RAYGEN_PROGRAM(simpleRayGen)()
  // {
  //   const RayGenData &self = owl::getProgramData<RayGenData>();
  //   const vec2i pixelID = owl::getLaunchIndex();
  //   if (pixelID == owl::vec2i(0)) {
  //     printf("%sHello OptiX From your First RayGen Program%s\n",
  //            OWL_TERMINAL_CYAN,
  //            OWL_TERMINAL_DEFAULT);
  //   }

  //   const vec2f screen = (vec2f(pixelID)+vec2f(.5f)) / vec2f(self.fbSize);
  //   owl::Ray ray;
  //   ray.origin    
  //     = self.camera.pos;
  //   ray.direction 
  //     = normalize(self.camera.dir_00
  //                 + screen.u * self.camera.dir_du
  //                 + screen.v * self.camera.dir_dv);

  //   vec3f color;
  //   owl::traceRay(/*accel to trace against*/self.world,
  //                 /*the ray to trace*/ray,
  //                 /*prd*/color);
    
  //   const int fbOfs = pixelID.x+self.fbSize.x*pixelID.y;
  //   self.fbPtr[fbOfs]
  //     = owl::make_rgba(color);
  // }

  // OPTIX_CLOSEST_HIT_PROGRAM(TriangleMesh)()
  // {
  //   vec3f &prd = owl::getPRD<vec3f>();

  //   const TrianglesGeomData &self = owl::getProgramData<TrianglesGeomData>();
  
  //   // compute normal:
  //   const int   primID = optixGetPrimitiveIndex();
  //   const vec3i index  = self.index[primID];
  //   const vec3f &A     = self.vertex[index.x];
  //   const vec3f &B     = self.vertex[index.y];
  //   const vec3f &C     = self.vertex[index.z];
  //   const vec3f Ng     = normalize(cross(B-A,C-A));

  //   const vec3f rayDir = optixGetWorldRayDirection();

  //   // compute texture coordinates
  //   const vec2f uv     = optixGetTriangleBarycentrics();
  //   const vec2f tc
  //     = (1.f-uv.x-uv.y)*self.texCoord[index.x]
  //     +      uv.x      *self.texCoord[index.y]
  //     +           uv.y *self.texCoord[index.z];
  //   // retrieve color from texture
  //   vec4f texColor = tex2D<float4>(self.texture,tc.x,tc.y);

  //   prd = (.2f + .8f*fabs(dot(rayDir,Ng))) * vec3f(texColor);

  // }

  // OPTIX_MISS_PROGRAM(miss)()
  // {
  //   const vec2i pixelID = owl::getLaunchIndex();

  //   const MissProgData &self = owl::getProgramData<MissProgData>();
  
  //   vec3f &prd = owl::getPRD<vec3f>();
  //   int pattern = (pixelID.x / 8) ^ (pixelID.y/8);
  //   prd = (pattern&1) ? self.color1 : self.color0;
  // }



  OPTIX_RAYGEN_PROGRAM(traceLocallyRG)()
  {
    auto &lp = getLP();
    int rayID = owl::getLaunchIndex().x;
    Woodcock::traceRay(rayID,
                       lp.forwardGlobals,
                       lp.volumeGlobals,
                       lp.surfaceGlobals);
  }

  
}
