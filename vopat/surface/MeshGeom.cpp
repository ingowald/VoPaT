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

#include "vopat/surface/MeshGeom.h"

namespace vopat {

  OWLGeomType MeshGeom::meshGT = 0;

  /*! defines 'our' geometry type in the given owl context */
  void MeshGeom::defineGeometryType(OWLContext owl, OWLModule devCode)
  {
    if (meshGT) return;

    std::vector<OWLVarDecl> vars;
    vars.push_back({"vertices",    OWL_BUFPTR,OWL_OFFSETOF(DD,vertices)});
    vars.push_back({"indices",     OWL_BUFPTR,OWL_OFFSETOF(DD,indices)});
    vars.push_back({"diffuseColor",OWL_FLOAT3,OWL_OFFSETOF(DD,diffuseColor)});

    meshGT = owlGeomTypeCreate(owl,OWL_TRIANGLES,sizeof(DD),vars.data(),vars.size());
    owlGeomTypeSetClosestHit(meshGT,0,devCode,"MeshGeomCH");
    owlGeomTypeSetAnyHit(meshGT,0,devCode,"MeshGeomAH");
  }
    
  OWLGeom MeshGeom::createGeom(OWLContext owl, mini::Mesh::SP mesh)
  {
    OWLBuffer vertexBuffer
      = owlManagedMemoryBufferCreate(owl,OWL_FLOAT3,
                                     mesh->vertices.size(),
                                     mesh->vertices.data());
    OWLBuffer indexBuffer
      = owlManagedMemoryBufferCreate(owl,OWL_INT3,
                                     mesh->indices.size(),
                                     mesh->indices.data());
    
    OWLGeom geom = owlGeomCreate(owl,meshGT);
    owlTrianglesSetVertices(geom,vertexBuffer,mesh->vertices.size(),
                            sizeof(vec3f),0);
    owlTrianglesSetIndices(geom,indexBuffer,mesh->indices.size(),
                           sizeof(vec3i),0);
    owlGeomSetBuffer(geom,"vertices",vertexBuffer);
    owlGeomSetBuffer(geom,"indices",indexBuffer);
    return geom;
  }
  
} // :: vopat
