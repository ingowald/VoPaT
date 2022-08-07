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

#include <string.h>
#include "VolumeRendererBase.h"
#include "../LaunchParams.h"


namespace std {
inline bool operator<(const owl::common::vec3i &a,
                      const owl::common::vec3i &b)
{
  if (a.x < b.x) return true;
  if (a.x > b.x) return false;
  if (a.y < b.y) return true;
  if (a.y > b.y) return false;
  if (a.z < b.z) return true;
  if (a.z > b.z) return false;
  return false;
}
}

namespace vopat {

  extern "C" char deviceCode_ptx[];

#if VOPAT_UMESH
  inline __device__
  float fatomicMin(float *addr, float value)
  {
    float old = *addr, assumed;
    if(old <= value) return old;
    do {
      assumed = old;
      old = atomicCAS((unsigned int*)addr, __float_as_int(assumed), __float_as_int(value));
        
    } while(old!=assumed);
    return old;
  }

  inline __device__
  float fatomicMax(float *addr, float value)
  {
    float old = *addr, assumed;
    if(old >= value) return old;
    do {
      assumed = old;
      old = atomicCAS((unsigned int*)addr, __float_as_int(assumed), __float_as_int(value));
        
    } while(old!=assumed);
    return old;
  }
  
  inline __device__
  int project(float f,
              const interval<float> range,
              int dim)
  {
    return max(0,min(dim-1,int(dim*(f-range.lower)/(range.upper-range.lower))));
  }

  inline __device__
  vec3i project(const vec3f &pos,
                const box3f &bounds,
                const vec3i &dims)
  {
    return vec3i(project(pos.x,{bounds.lower.x,bounds.upper.x},dims.x),
                 project(pos.y,{bounds.lower.y,bounds.upper.y},dims.y),
                 project(pos.z,{bounds.lower.z,bounds.upper.z},dims.z));
  }

  inline __device__
  void rasterBox(MacroCell *d_mcGrid,
                 const vec3i dims,
                 const box3f worldBounds,
                 const box4f primBounds4,
                 bool dbg=false)
  {
    box3f pb = box3f(vec3f(primBounds4.lower),
                     vec3f(primBounds4.upper));
    if (pb.lower.x >= pb.upper.x) return;
    if (pb.lower.y >= pb.upper.y) return;
    if (pb.lower.z >= pb.upper.z) return;

    vec3i lo = project(pb.lower,worldBounds,dims);
    vec3i hi = project(pb.upper,worldBounds,dims);

    for (int iz=lo.z;iz<=hi.z;iz++)
      for (int iy=lo.y;iy<=hi.y;iy++)
        for (int ix=lo.x;ix<=hi.x;ix++) {
          const int cellID
            = ix
            + iy * dims.x
            + iz * dims.x * dims.y;
          auto &cell = d_mcGrid[cellID].inputRange;
          fatomicMin(&cell.lower,primBounds4.lower.w);
          fatomicMax(&cell.upper,primBounds4.upper.w);
        }
  }
  
  constexpr int MAX_GRID_SIZE = 1024;
  
  __global__ void clearMCs(MacroCell *mcData,
                           vec3i dims)
  {
    int ix = threadIdx.x+blockIdx.x*blockDim.x; if (ix >= dims.x) return;
    int iy = threadIdx.y+blockIdx.y*blockDim.y; if (iy >= dims.y) return;
    int iz = threadIdx.z+blockIdx.z*blockDim.z; if (iz >= dims.z) return;

    int ii = ix + dims.x*(iy + dims.y*(iz));
    mcData[ii].inputRange.lower = +FLT_MAX;
    mcData[ii].inputRange.upper = -FLT_MAX;
  }
  
  __global__ void rasterTets(MacroCell *mcData,
                             vec3i mcDims,
                             box3f domain,
                             vec3f *vertices,
                             float *scalars,
                             umesh::UMesh::Tet *tets,
                             int numTets)
  {
    const int blockID
      = blockIdx.x
      + blockIdx.y * MAX_GRID_SIZE
      ;
    const int primIdx = blockID*blockDim.x + threadIdx.x;
    if (primIdx >= numTets) return;    

    umesh::UMesh::Tet tet = tets[primIdx];
    const box4f primBounds4 = box4f()
      .including(vec4f(vertices[tet.x],scalars[tet.x]))
      .including(vec4f(vertices[tet.y],scalars[tet.y]))
      .including(vec4f(vertices[tet.z],scalars[tet.z]))
      .including(vec4f(vertices[tet.w],scalars[tet.w]));

    rasterBox(mcData,mcDims,domain,primBounds4);
  }
#endif
  
  __global__ void checkSkipTree(BVHNode *nodes, int numNodes)
  {
    int tid = threadIdx.x+blockIdx.x*blockDim.x;
    if (tid >= numNodes) return;
    if (nodes[tid].skipTreeChild >=4) {
      printf("CHECK : %i -> %i (byte %lx ptr 0x%lx)\n",
             tid,nodes[tid].skipTreeChild,
             tid*sizeof(*nodes),
             &nodes[tid]);
    }
  }

  void sortIndices(int &A, int &B, int &orientation)
  {
    if (A > B) {
      std::swap(A,B);
      orientation = 1-orientation;
    }
  }
  
  void sortIndices(vec3i &face, int &orientation)
  {
    sortIndices(face.y,face.z,orientation);
    sortIndices(face.x,face.y,orientation);
    sortIndices(face.y,face.z,orientation);
  };
  
  template<typename Lambda>
  void iterateFaces(umesh::Tet tet, Lambda lambda)
  {
    int A = tet.x;
    int B = tet.y;
    int C = tet.z;
    int D = tet.w;
    std::vector<vec3i> faces = {
                                vec3i{ A, C, B },
                                vec3i{ A, D, C },
                                vec3i{ A, B, D },
                                vec3i{ B, C, D }
    };
    for (auto face : faces) {
      int orientation = 0;
      sortIndices(face,orientation);
      lambda(face,orientation);
    }
  }
  
  VolumeRenderer::VolumeRenderer(Model::SP model,
                                 const std::string &baseFileName,
                                 int islandRank,
                                 int gpuID)
    : model(model), islandRank(islandRank)
  {
    //    CUDA_CALL(SetDevice(comm->worker.gpuID));
    
    if (islandRank < 0)
      return;


#if VOPAT_UMESH_OPTIX
    owl = owlContextCreate(&gpuID,1);
    owlDevCode = owlModuleCreate(owl,deviceCode_ptx);
    OWLVarDecl args[] = {
                         { "vertices", OWL_BUFPTR, OWL_OFFSETOF(UMeshGeom,vertices) },
                         { "scalars", OWL_BUFPTR, OWL_OFFSETOF(UMeshGeom,scalars) },
                         { "tets", OWL_BUFPTR, OWL_OFFSETOF(UMeshGeom,tets) },
                         { "tetsOnFace", OWL_BUFPTR, OWL_OFFSETOF(UMeshGeom,tetsOnFace) },
                         { nullptr }
    };
    umeshGT = owlGeomTypeCreate(owl,OWL_TRIANGLES,
                                sizeof(UMeshGeom),
                                args,-1);
    owlGeomTypeSetClosestHit(umeshGT,0,owlDevCode,"UMeshGeomCH");
#endif
    
    // ------------------------------------------------------------------
    // upload per-rank boxes
    // ------------------------------------------------------------------
    std::vector<box3f> hostRankBoxes;
    for (auto brick : model->bricks)
      hostRankBoxes.push_back(
#if VOPAT_UMESH
                              brick->domain
#else
                              brick->spaceRange
#endif
                              );
    rankBoxes.upload(hostRankBoxes);
    globals.rankBoxes = rankBoxes.get();

    myBrick = model->bricks[islandRank];
    const std::string fileName = Model::canonicalRankFileName(baseFileName,islandRank);
#if VOPAT_UMESH
    myBrick->load(fileName);

    std::cout << "brick and umesh loaded" << std::endl;
# if VOPAT_UMESH_OPTIX
# else
    //    gdt::qbvh::BVH4 bvh;
    vopat::BVH bvh;
    std::cout << "building bvh ... " << prettyNumber(myBrick->umesh->tets.size()) << " tets" << std::endl;
    gdt::qbvh::build(bvh,myBrick->umesh->tets.size(),
                     [&](size_t tetID)->box3f {
                       auto tet = myBrick->umesh->tets[tetID];
                       return box3f((const vec3f&)myBrick->umesh->vertices[tet.x])
                         .including((const vec3f&)myBrick->umesh->vertices[tet.y])
                         .including((const vec3f&)myBrick->umesh->vertices[tet.z])
                         .including((const vec3f&)myBrick->umesh->vertices[tet.w]);
                     });
# endif
    globals.myRegion      = myBrick->domain;
    globals.umesh.domain = myBrick->domain;

#if VOPAT_UMESH_OPTIX
    umeshScalarsBuffer = owlDeviceBufferCreate(owl,OWL_FLOAT,
                                               myBrick->umesh->perVertex->values.size(),
                                               myBrick->umesh->perVertex->values.data());
    globals.umesh.scalars = (float*)owlBufferGetPointer(umeshScalarsBuffer,0);
    
    umeshVerticesBuffer = owlDeviceBufferCreate(owl,OWL_FLOAT3,
                                                myBrick->umesh->vertices.size(),
                                                myBrick->umesh->vertices.data());
    globals.umesh.vertices = (vec3f*)owlBufferGetPointer(umeshVerticesBuffer,0);

    umeshTetsBuffer = owlDeviceBufferCreate(owl,OWL_INT4,
                                            myBrick->umesh->tets.size(),
                                            myBrick->umesh->tets.data());
    globals.umesh.tets = (umesh::UMesh::Tet*)owlBufferGetPointer(umeshTetsBuffer,0);
#else
    myScalars.upload(myBrick->umesh->perVertex->values);
    globals.umesh.scalars   = myScalars.get();
    std::vector<vec3f> _vertices;
    for (auto &v : myBrick->umesh->vertices)
      _vertices.push_back(vec3f(v.x,v.y,v.z));
    myVertices.upload(_vertices);
// #else
//     myVertices.upload((const std::vector<vec3f> &)myBrick->umesh->vertices);
// #endif
    globals.umesh.vertices   = myVertices.get();

    myTets.upload(myBrick->umesh->tets);
    globals.umesh.tets   = myTets.get();
#endif
    
// #if 1
    

# if VOPAT_UMESH_OPTIX
    std::cout << "building shared faces accel" << std::endl;
    std::map<vec3i,int> faceID;
    std::vector<vec3i> sharedFaceIndices;
    std::vector<vec2i> sharedFaceNeighbors;
    for (auto tet : myBrick->umesh->tets)
      iterateFaces(tet,
                   [&faceID,&sharedFaceIndices,&sharedFaceNeighbors]
                   (const vec3i faceVertices, int side)
                   {
                     if (faceID.find(faceVertices) == faceID.end()) {
                       faceID[faceVertices] = sharedFaceIndices.size();
                       sharedFaceIndices.push_back(faceVertices);
                       sharedFaceNeighbors.push_back(vec2i(-1));
                     }
                   });
    
    for (int tetID=0;tetID<myBrick->umesh->tets.size();tetID++) {
      auto tet = myBrick->umesh->tets[tetID];
      iterateFaces(myBrick->umesh->tets[tetID],[&](vec3i faceVertices, int side){
          sharedFaceNeighbors[faceID[faceVertices]][side] = tetID;
        });
    }

    sharedFaceIndicesBuffer
      = owlManagedMemoryBufferCreate(owl,OWL_INT3,
                              sharedFaceIndices.size(),sharedFaceIndices.data());
    sharedFaceNeighborsBuffer
      = owlManagedMemoryBufferCreate(owl,OWL_INT2,
                              sharedFaceNeighbors.size(),sharedFaceNeighbors.data());
    umeshGeom = owlGeomCreate(owl,umeshGT);
    
    owlTrianglesSetVertices(umeshGeom,umeshVerticesBuffer,
                            myBrick->umesh->vertices.size(),sizeof(vec3f),0);
    owlTrianglesSetIndices(umeshGeom,sharedFaceIndicesBuffer,
                           sharedFaceIndices.size(),sizeof(vec3i),0);
    owlGeomSetBuffer(umeshGeom,"tets",umeshTetsBuffer);
    owlGeomSetBuffer(umeshGeom,"vertices",umeshVerticesBuffer);
    owlGeomSetBuffer(umeshGeom,"scalars",umeshScalarsBuffer);
    owlGeomSetBuffer(umeshGeom,"tetsOnFace",sharedFaceNeighborsBuffer);
    umeshAccel = owlTrianglesGeomGroupCreate(owl,1,&umeshGeom);

    // and put this into a single-instnace tlas
    owlGroupBuildAccel(umeshAccel);
    umeshAccel = owlInstanceGroupCreate(owl,1,&umeshAccel);
    
    owlGroupBuildAccel(umeshAccel);
    globals.umesh.sampleAccel = owlGroupGetTraversable(umeshAccel,0);
    
# else
    std::cout << "uploading nodes" << std::endl;
    PRINT(bvh.nodes[0].numChildren);
    PRINT(sizeof(bvh.nodes[0]));
    myBVHNodes.upload(bvh.nodes);
    for (auto &node : bvh.nodes)
      if (node.skipTreeChild >= 4) {
        PING; PRINT(node.skipTreeChild);
      };
    globals.umesh.bvhNodes = myBVHNodes.get();
    // PRINT(bvh.nodes.size());
    // PRINT(myBrick->umesh->perVertex->values.size());
    // {
    //   int bs = 128;
    //   int nb = divRoundUp((int)bvh.nodes.size(),bs);
    //   std::cout << "device-checking " << bvh.nodes.size() << " nodes' skip values (2)" << std::endl;
    //   checkSkipTree<<<nb,bs>>>(globals.umesh.bvhNodes,bvh.nodes.size());
    //   CUDA_SYNC_CHECK();
    //   std::cout << "done DEVICE checking of skip tree" << std::endl;
    // }
# endif
    
#else
# if VOPAT_VOXELS_AS_TEXTURE
    std::vector<float> hostVoxels;
    myBrick->load(hostVoxels,fileName);

    // Copy voxels to cuda array
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
    cudaExtent extent{(unsigned)myBrick->numVoxels.x,
                      (unsigned)myBrick->numVoxels.y,
                      (unsigned)myBrick->numVoxels.z};
    CUDA_CALL(Malloc3DArray(&voxelArray,&desc,extent,0));
    cudaMemcpy3DParms copyParms;
    memset(&copyParms,0,sizeof(copyParms));
    copyParms.srcPtr = make_cudaPitchedPtr(hostVoxels.data(),
                                           (size_t)myBrick->numVoxels.x*sizeof(float),
                                           (size_t)myBrick->numVoxels.x,
                                           (size_t)myBrick->numVoxels.y);
    copyParms.dstArray = voxelArray;
    copyParms.extent   = extent;
    copyParms.kind     = cudaMemcpyHostToDevice;
    CUDA_CALL(Memcpy3D(&copyParms));

    // Create a texture object
    cudaResourceDesc resourceDesc;
    memset(&resourceDesc,0,sizeof(resourceDesc));
    resourceDesc.resType         = cudaResourceTypeArray;
    resourceDesc.res.array.array = voxelArray;

    cudaTextureDesc textureDesc;
    memset(&textureDesc,0,sizeof(textureDesc));
    textureDesc.addressMode[0]   = cudaAddressModeClamp;
    textureDesc.addressMode[1]   = cudaAddressModeClamp;
    textureDesc.addressMode[2]   = cudaAddressModeClamp;
    textureDesc.filterMode       = cudaFilterModeLinear;
    textureDesc.readMode         = cudaReadModeElementType;
    textureDesc.normalizedCoords = false;

    CUDA_CALL(CreateTextureObject(&globals.volume.texObj,&resourceDesc,&textureDesc,0));

    // 2nd texture object for nearest filtering
    textureDesc.filterMode       = cudaFilterModePoint;
    CUDA_CALL(CreateTextureObject(&globals.volume.texObjNN,&resourceDesc,&textureDesc,0));
# else
#  if 1
    myBrick->load(voxels,fileName);
#  else
    std::vector<float> loadedVoxels = myBrick->load(fileName);
    voxels.upload(loadedVoxels);
#  endif
    globals.volume.voxels = voxels.get();
# endif
    globals.volume.dims   = myBrick->numVoxels;//voxelRange.size();
    globals.myRegion      = myBrick->spaceRange;
#endif
    /* initialize to model value range; xf editor may mess with that
       later on */
    globals.xf.domain = model->valueRange;
    globals.islandRank = islandRank;
    globals.islandSize = model->bricks.size();
  
    globals.gradientDelta = vec3f(1.f);

    initMacroCells();
  }

  void VolumeRenderer::initMacroCells()
  {
#if VOPAT_UMESH
    std::cout << "need to rebuild macro cells .." << std::endl;
    globals.mc.dims = 256;
    mcData.resize(volume(globals.mc.dims));
    CUDA_SYNC_CHECK();// PING;
    // CUDA_CALL(Memset(mcData.get(),0,mcData.numBytes()));
    clearMCs<<<(dim3)divRoundUp(globals.mc.dims,vec3i(4)),(dim3)vec3i(4)>>>
      (mcData.get(),globals.mc.dims);
    CUDA_SYNC_CHECK(); //PING;
    globals.mc.data  = mcData.get();

    if (myBrick->umesh->tets.empty())
      throw std::runtime_error("no tets!?");
    const unsigned int blockSize = 128;
    unsigned int numTets = (int)myBrick->umesh->tets.size();
    const unsigned int numBlocks = divRoundUp(numTets,blockSize*1024u);
    rasterTets
      <<<{1024u,numBlocks},{blockSize,1u}>>>
                             (globals.mc.data,
                              globals.mc.dims,
                              globals.umesh.domain,
                              globals.umesh.vertices,
                              globals.umesh.scalars,
                              globals.umesh.tets,
                              numTets);
                             CUDA_SYNC_CHECK();
#else
                             globals.mc.dims = divRoundUp(myBrick->numCells,vec3i(mcWidth));
                             mcData.resize(volume(globals.mc.dims));
                             globals.mc.data  = mcData.get();
                             globals.mc.width = mcWidth;
  
                             VoxelData voxelData = *(VoxelData*)&globals.volume;
                             initMacroCell<<<(dim3)globals.mc.dims,(dim3)vec3i(4)>>>
                               (globals.mc.data,globals.mc.dims,mcWidth,voxelData);
#endif
  }

  
  void VolumeRenderer::setTransferFunction(const std::vector<vec4f> &cm,
                                           const interval<float> &xfDomain,
                                           const float density)
  {
    if (islandRank < 0) {
    } else {
      colorMap.upload(cm);
      globals.xf.values = colorMap.get();
      globals.xf.numValues = cm.size();
      globals.xf.domain = xfDomain;
      globals.xf.density = density;

      mapMacroCell<<<(dim3)globals.mc.dims,(dim3)vec3i(4)>>>
        (mcData.get(),globals.mc.dims,
         globals.xf.values,
         globals.xf.numValues,
         globals.xf.domain);
    }
  }
    
}

