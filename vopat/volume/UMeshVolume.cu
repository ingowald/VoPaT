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
    owlParamsSetRaw(lp,"volumeSampler.umesh",&globals);
  }
  
  void UMeshVolume::addLPVars(std::vector<OWLVarDecl> &lpVars) 
  {
    lpVars.push_back({"volumeSampler.umesh",OWL_USER_TYPE(DD),
                      OWL_OFFSETOF(LaunchParams,volumeSampler.umesh)});
  }
}
