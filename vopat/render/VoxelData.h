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

#pragma once

#include "vopat/common.h"
#if VOPAT_UMESH
# include "vopat/umesh/qbvh.h"
# include <umesh/UMesh.h>
#endif

namespace vopat {

#if VOPAT_UMESH
  typedef gdt::qbvh::BVH<4> BVH;
  typedef gdt::qbvh::Node<4> BVHNode;
  
  struct UMeshData {
    inline __device__ bool sample(float &f, vec3f P, bool dbg) const;
    inline __device__ bool sampleElement(const int idx, float &f, vec3f P, bool dbg) const;
    /*! look up the given 3D (*local* world-space) point in the volume, and return the gradient */
    inline __device__ bool gradient(vec3f &g, vec3f P, vec3f delta, bool dbg) const;

    box3f  domain;
    vec3f *vertices;
    float *scalars;
    umesh::UMesh::Tet *tets;
    // int    numVertices;
    // int    numTets;
    BVHNode *bvhNodes;
  };
  
/*! computes the (oriented) volume of the tet given by the four
    vertices */
inline __device__
float volume(const vec3f &P,
             const vec3f &A,
             const vec3f &B,
             const vec3f &C)
{
  return dot(P-A,cross(B-A,C-A));
}

/*! compute point-in-tet test for given point P. If point is inside
    the tet, this linearly interpolates among the tet's vertex values,
    and returns true (the return value is passed through &value); if
    outside, it returns false, and value is undefined */
  inline __device__ bool UMeshData::sampleElement(const int tetID, float &value, vec3f P, bool dbg) const
// inline __device__ bool
// interpolateTet(const int tetID,
//                const vec3f &P,
//                float &value)
  {
    const umesh::UMesh::Tet index = tets[tetID];
    const vec3f V0 = vertices[index.x];
    const vec3f V1 = vertices[index.y];
    const vec3f V2 = vertices[index.z];
    const vec3f V3 = vertices[index.w];
    
    const vec3f P0 = (const vec3f &)V0;
    const vec3f P1 = (const vec3f &)V1;
    const vec3f P2 = (const vec3f &)V2;
    const vec3f P3 = (const vec3f &)V3;

    const float vol_all = volume(/*point*/P0,/*base-tri*/P1,P3,P2);
    if (vol_all == 0.f) return false;
    
    const float bary0 = volume(/*point*/P,/*base-tri*/P1,P3,P2) / vol_all;
    if (bary0 < 0.f) return false;
    
    const float bary1 = volume(/*point*/P,/*base-tri*/P0,P2,P3) / vol_all;
    if (bary1 < 0.f) return false;
    
    const float bary2 = volume(/*point*/P,/*base-tri*/P0,P3,P1) / vol_all;
    if (bary2 < 0.f) return false;
    
    const float bary3 = volume(/*point*/P,/*base-tri*/P0,P1,P2) / vol_all;
    if (bary3 < 0.f) return false;

    value
      = //attrPerVertex
      // ?
      (bary0 * scalars[index.x] +
       bary1 * scalars[index.y] +
       bary2 * scalars[index.z] +
       bary3 * scalars[index.w])
      // : scalars[tetID]
      ;
    return true;
}

  inline __device__ bool UMeshData::sample(float &f, vec3f P, bool dbg) const
  {
#if SKIP_TREE
    int nodeID = 0;
    int childID = 0;
    BVHNode  node;
    gdt::qbvh::ChildRef cr;
    while (true) {
      while (true) {
        if (nodeID < 0)
          return false;
        
        node = bvhNodes[nodeID];
        cr   = node.childRef[childID];
        
        if (cr.valid() && node.getBounds(childID).contains(P)) {
          if (cr.isLeaf())
            break;
          nodeID = cr.getChildIndex();
          childID = 0;
          continue;
        }
        childID++;
        if (childID >= node.numChildren || !node.childRef[childID].valid()) {
          nodeID = node.skipTreeNode;
          childID = node.skipTreeChild;
        }
      }
      // now all are a at a leaf ...
      if (sampleElement(cr.getPrimIndex(),f,P,dbg))
        return true;
      
      childID++;
      if (childID >= node.numChildren || !node.childRef[childID].valid()) {
        nodeID = node.skipTreeNode;
        childID = node.skipTreeChild;
      }
    }
#else
    enum { stackDepth = 30 };
    int stackPtr = 0;
    int nodeStack[stackDepth];
    int nodeID = 0;
    while (1) {
      const BVHNode node = bvhNodes[nodeID];
      for (int i=0;i<node.numChildren;i++) {
        auto cr = node.childRef[i];
        if (!cr.valid() || !node.getBounds(i).contains(P)) continue;
        if (cr.isLeaf()) {
          if (sampleElement(cr.getPrimIndex(),f,P,dbg))
            return true;
        } else {
          if (stackPtr >= stackDepth-1)
            printf("CAREFUL - STACK OVERFLOW!?\n");
          else
            nodeStack[stackPtr++] = cr.getChildIndex();
        }
      }
      if (stackPtr == 0) return false;
      nodeID = nodeStack[--stackPtr];
    }
#endif
  }
  
  inline __device__ bool UMeshData::gradient(vec3f &g, vec3f P, vec3f delta, bool dbg) const
  {
    float right,left,top,bottom,front,back;
    sample(right, P+vec3f(delta.x,0.f,0.f),dbg);
    sample(left,  P-vec3f(delta.x,0.f,0.f),dbg);
    sample(top,   P+vec3f(0.f,delta.y,0.f),dbg);
    sample(bottom,P-vec3f(0.f,delta.y,0.f),dbg);
    sample(front, P+vec3f(0.f,0.f,delta.z),dbg);
    sample(back,  P-vec3f(0.f,0.f,delta.z),dbg);
    g = vec3f(right-left,top-bottom,front-back);
    return true;
  }

#endif
  
  struct VoxelData {
#if VOPAT_VOXELS_AS_TEXTURE
    cudaTextureObject_t texObj;
    cudaTextureObject_t texObjNN;
#else
    float *voxels;
#endif
    vec3i dims;

    /*! look up the given 3D (*local* world-space) point in the volume, and return interpolated scalar value */
    inline __device__ bool sample(float &f, vec3f P, bool dbg=false) const;

    /*! look up the given 3D (*local* world-space) point in the volume, and return the gradient */
    inline __device__ bool gradient(vec3f &g, vec3f P, vec3f delta, bool dbg=false) const;
  };

  /*! look up the given 3D (*local-space*) point in the volume, and return interpolated scalar value */
  inline __device__ bool VoxelData::sample(float &f, vec3f P, bool dbg) const
  {
#if VOPAT_VOXELS_AS_TEXTURE
    P += vec3f(.5f); // Transform to CUDA texture cell-centric
    tex3D(&f,this->texObj,P.x,P.y,P.z);
    return true;
#else
#if 1
    // tri-lerp:
    vec3ui cellID = clamp(vec3ui(P),vec3ui(0),vec3ui(dims)-vec3ui(2));
    vec3f  frac   = P - floor(P);

    size_t cx0 = cellID.x;
    size_t cy0 = cellID.y * size_t(this->dims.x);
    size_t cz0 = cellID.z * (size_t(this->dims.x) * size_t(this->dims.y));
    size_t cx1 = cx0 + 1;
    size_t cy1 = cy0 + size_t(this->dims.x);
    size_t cz1 = cz0 + (size_t(this->dims.x) * size_t(this->dims.y));
      
    float f000 = this->voxels[cx0+cy0+cz0];
    float f001 = this->voxels[cx1+cy0+cz0];
    float f010 = this->voxels[cx0+cy1+cz0];
    float f011 = this->voxels[cx1+cy1+cz0];
    float f100 = this->voxels[cx0+cy0+cz1];
    float f101 = this->voxels[cx1+cy0+cz1];
    float f110 = this->voxels[cx0+cy1+cz1];
    float f111 = this->voxels[cx1+cy1+cz1];

    float f00x = (1.f-frac.x)*f000 + frac.x*f001;
    float f01x = (1.f-frac.x)*f010 + frac.x*f011;
    float f10x = (1.f-frac.x)*f100 + frac.x*f101;
    float f11x = (1.f-frac.x)*f110 + frac.x*f111;

    float f0y = (1.f-frac.y)*f00x + frac.y*f01x;
    float f1y = (1.f-frac.y)*f10x + frac.y*f11x;

    float fz = (1.f-frac.z)*f0y + frac.z*f1y;
    f = fz;

    if (isnan(f)) {
      printf("f is NAN! P %f %f %f lerp %f %f %f\n",P.x,P.y,P.z,frac.x,frac.y,frac.z);
    }
    return true;
#else
    // nearest 
    vec3ui cellID = clamp(vec3ui(P),vec3ui(0),vec3ui(dims)-vec3ui(2));
      
    f = this->voxels[cellID.x
                            +this->dims.x*(cellID.y
                                                  +this->dims.y*size_t(cellID.z))];
    return true;
#endif
#endif
  }

  inline __device__ bool VoxelData::gradient(vec3f &g, vec3f P, vec3f delta, bool dbg) const
  {
    float right,left,top,bottom,front,back;
    sample(right, P+vec3f(delta.x,0.f,0.f),dbg);
    sample(left,  P-vec3f(delta.x,0.f,0.f),dbg);
    sample(top,   P+vec3f(0.f,delta.y,0.f),dbg);
    sample(bottom,P-vec3f(0.f,delta.y,0.f),dbg);
    sample(front, P+vec3f(0.f,0.f,delta.z),dbg);
    sample(back,  P-vec3f(0.f,0.f,delta.z),dbg);
    g = vec3f(right-left,top-bottom,front-back);
    return true;
  }

} // ::vopat

