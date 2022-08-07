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
#include <owl/owl.h>
#include "vopat/render/VopatBase.h"

#include "vopat/render/Woodcock.cu"

using namespace vopat;

__constant__ uint8_t _optixLaunchParams[sizeof(LaunchParams)];

inline __device__ const LaunchParams &getLP()
{ return (const LaunchParams &)_optixLaunchParams; }

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

OPTIX_RAYGEN_PROGRAM(traceLocallyRG)()
{
  printf("blad\n");
}

OPTIX_RAYGEN_PROGRAM(generatePrimaryWaveRG)()
{
  auto &lp = getLP();
  const vec2i launchIdx = owl::getLaunchIndex();
  generatePrimaryWaveKernel<WoodcockKernels>
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

