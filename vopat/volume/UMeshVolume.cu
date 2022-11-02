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

#include "vopat/volume/UMeshVolume.h"
#include "vopat/LaunchParams.h"

namespace vopat {

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

  void UMeshVolume::build(OWLContext owl,
                          OWLModule owlDevCode)
  {
    OWLVarDecl args[] = {
                         { "vertices", OWL_BUFPTR, OWL_OFFSETOF(Geom,vertices) },
                         { "scalars", OWL_BUFPTR, OWL_OFFSETOF(Geom,scalars) },
                         { "tets", OWL_BUFPTR, OWL_OFFSETOF(Geom,tets) },
                         { "tetsOnFace", OWL_BUFPTR, OWL_OFFSETOF(Geom,tetsOnFace) },
                         { nullptr }
    };
    gt = owlGeomTypeCreate(owl,OWL_TRIANGLES,
                           sizeof(Geom),
                           args,-1);
    owlGeomTypeSetClosestHit(gt,0,owlDevCode,"UMeshGeomCH");

    scalarsBuffer = owlDeviceBufferCreate(owl,OWL_FLOAT,
                                               myBrick->umesh->perVertex->values.size(),
                                               myBrick->umesh->perVertex->values.data());
    // globals.scalars = (float*)owlBufferGetPointer(scalarsBuffer,0);
    
    verticesBuffer = owlDeviceBufferCreate(owl,OWL_FLOAT3,
                                                myBrick->umesh->vertices.size(),
                                                myBrick->umesh->vertices.data());
    // globals.vertices = (vec3f*)owlBufferGetPointer(verticesBuffer,0);

    tetsBuffer = owlDeviceBufferCreate(owl,OWL_INT4,
                                            myBrick->umesh->tets.size(),
                                            myBrick->umesh->tets.data());
    // globals.tets = (umesh::UMesh::Tet*)owlBufferGetPointer(tetsBuffer,0);

    CUDA_SYNC_CHECK();
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

    std::cout << "shared faces stuff built on host, setting up device geom" << std::endl;
    CUDA_SYNC_CHECK();
    sharedFaceIndicesBuffer
      = owlManagedMemoryBufferCreate(owl,OWL_INT3,
                              sharedFaceIndices.size(),sharedFaceIndices.data());
    sharedFaceNeighborsBuffer
      = owlManagedMemoryBufferCreate(owl,OWL_INT2,
                              sharedFaceNeighbors.size(),sharedFaceNeighbors.data());
    geom = owlGeomCreate(owl,gt);
    
    owlTrianglesSetVertices(geom,verticesBuffer,
                            myBrick->umesh->vertices.size(),sizeof(vec3f),0);
    owlTrianglesSetIndices(geom,sharedFaceIndicesBuffer,
                           sharedFaceIndices.size(),sizeof(vec3i),0);
    owlGeomSetBuffer(geom,"tets",tetsBuffer);
    owlGeomSetBuffer(geom,"vertices",verticesBuffer);
    owlGeomSetBuffer(geom,"scalars",scalarsBuffer);
    owlGeomSetBuffer(geom,"tetsOnFace",sharedFaceNeighborsBuffer);
    blas = owlTrianglesGeomGroupCreate(owl,1,&geom);

    // and put this into a single-instnace tlas
    owlGroupBuildAccel(blas);
    tlas = owlInstanceGroupCreate(owl,1,&blas);
    
    owlGroupBuildAccel(tlas);
    globals.sampleAccel = owlGroupGetTraversable(tlas,0);
    
    CUDA_SYNC_CHECK();
    std::cout << "shared faces stuff built. done" << std::endl;
  }
  
  void UMeshVolume::setDD(OWLLaunchParams lp) 
  {
    // globals.xf.values    = this->xf.colorMap.get();
    // globals.xf.numValues = this->xf.colorMap.N;
    // globals.xf.domain    = this->xf.domain;
    // globals.xf.density   = this->xf.density;
        
    owlParamsSetRaw(lp,"volumeSampler.umesh",&globals);
  }
  
  void UMeshVolume::addLPVars(std::vector<OWLVarDecl> &lpVars) 
  {
    lpVars.push_back({"volumeSampler.umesh",OWL_USER_TYPE(DD),
                      OWL_OFFSETOF(LaunchParams,volumeSampler.umesh)});
  }
  






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
  
  void UMeshVolume::buildMCs(MCGrid &mcGrid) 
  {
    std::cout << OWL_TERMINAL_BLUE
              << "#vopat.umesh: building macro cells .."
              << OWL_TERMINAL_DEFAULT
              << std::endl;
    mcGrid.dd.dims = 256;
    mcGrid.cells.resize(volume(mcGrid.dd.dims));
    mcGrid.dd.cells = mcGrid.cells.get();
    CUDA_SYNC_CHECK();// PING;
    // CUDA_CALL(Memset(mcGrid.cells.get(),0,mcGrid.cells.numBytes()));
    clearMCs<<<(dim3)divRoundUp(mcGrid.dd.dims,vec3i(4)),(dim3)vec3i(4)>>>
      (mcGrid.cells.get(),mcGrid.dd.dims);
    CUDA_SYNC_CHECK(); //PING;
    mcGrid.dd.cells  = mcGrid.cells.get();

    if (myBrick->umesh->tets.empty())
      throw std::runtime_error("no tets!?");
    const unsigned int blockSize = 128;
    unsigned int numTets = (int)myBrick->umesh->tets.size();
    const unsigned int numBlocks = divRoundUp(numTets,blockSize*1024u);
    dim3 _nb{1024u,numBlocks,1u};
    dim3 _bs{blockSize,1u,1u};
    rasterTets<<<_nb,_bs>>>
      (mcGrid.dd.cells,
       mcGrid.dd.dims,
       myBrick->domain,
       (vec3f*)owlBufferGetPointer(verticesBuffer,0),
       (float*)owlBufferGetPointer(scalarsBuffer,0),
       (umesh::UMesh::Tet*)owlBufferGetPointer(tetsBuffer,0),
       myBrick->umesh->tets.size());
    CUDA_SYNC_CHECK();
    std::cout << OWL_TERMINAL_GREEN
              << "#vopat.umesh: done building macro cells .."
              << OWL_TERMINAL_DEFAULT
              << std::endl;
  }
}
