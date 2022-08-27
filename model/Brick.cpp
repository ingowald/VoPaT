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

#include "model/Brick.h"
#include <fstream>

namespace vopat {

#if VOPAT_UMESH
  std::vector<box4f> Brick::makeShards(int numShards)
  {
    struct ShardNode {
      box3f domain;
      umesh::range1f valueRange;
      int childID = -1;
      int numShards;
    };

    // ------------------------------------------------------------------
    // BUILD kd-tree that defines the shards
    // ------------------------------------------------------------------
    std::vector<ShardNode> nodes;
    nodes.resize(1);
    nodes[0].domain = getDomain();
    nodes[0].childID = -1;
    nodes[0].numShards = numShards;
    std::stack<int> todo; todo.push(0);
    while (!todo.empty()) {
      int nodeID = todo.top();todo.pop();
      if (nodes[nodeID].numShards == 1)
        continue;
      int childID = nodes.size();
      nodes[nodeID].childID = childID;
      nodes.push_back({});
      nodes.push_back({});

      int Nl = numShards/2;
      int Nr = numShards - Nl;
      float ratio = float(Nl)/float(numShards);
      auto &pDomain = nodes[nodeID].domain;
      auto &lDomain = nodes[childID+0].domain;
      auto &rDomain = nodes[childID+1].domain;

      int dim = arg_max(pDomain.size());
      float mid = pDomain.lower[dim] + pDomain.size()[dim]*ratio;
      
      lDomain = rDomain = domain;
      lDomain.upper[dim] = rDomain.lower[dim] = mid;

      nodes[childID+0].numShards = Nl;
      nodes[childID+1].numShards = Nr;

      if (Nl > 1) todo.push(childID+0);
      if (Nr > 1) todo.push(childID+1);      
    };
    
    // ------------------------------------------------------------------
    // RASTER the umesh
    // ------------------------------------------------------------------
    std::vector<umesh::UMesh::PrimRef> prims = umesh->createVolumePrimRefs();
    for (auto prim : prims) {
      umesh::box3f primBounds = umesh->getBounds(prim);
      umesh::range1f primRange = umesh->getValueRange(prim);

      todo.push(0);
      while(!todo.empty()) {
        int nodeID = todo.top(); todo.pop();
        const int childID = nodes[nodeID].childID;
        if (childID >= 0) {
          for (int c=0;c<2;c++)
            if (nodes[childID+c].domain.overlaps((const box3f&)primBounds))
              todo.push(nodes[nodeID].childID+c);
          
        }
        else
          nodes[nodeID].valueRange.extend(primRange);
      }
    }
    
    // ------------------------------------------------------------------
    // gather the leaves
    // ------------------------------------------------------------------
    std::vector<box4f> result;
    for (auto &node : nodes) {
      if (node.childID >= 0) continue;
      result.push_back({ { node.domain.lower.x,
                           node.domain.lower.y,
                           node.domain.lower.z,
                           node.valueRange.lower },
                         { node.domain.upper.x,
                           node.domain.upper.y,
                           node.domain.upper.z,
                           node.valueRange.upper }});
    }
    return result;
  }
#else
  void Brick::makeShards(std::vector<box4f> &result,
                         const Brick *brick,
                         const float *scalars,
                         const box3i &cellRange,
                         int numShards)
  {
    if (numShards == 1 || cellRange.size() == vec3i(1)) {
      box4f shard;
      shard.lower.x = float(cellRange.lower.x);
      shard.lower.y = float(cellRange.lower.y);
      shard.lower.z = float(cellRange.lower.z);

      shard.upper.x = float(cellRange.upper.x);
      shard.upper.y = float(cellRange.upper.y);
      shard.upper.z = float(cellRange.upper.z);

      range1f valueRange;
      const vec3i localSize = brick->numVoxels;
      for (int iz=cellRange.lower.z;iz<=cellRange.upper.z;iz++)
        for (int iy=cellRange.lower.y;iy<=cellRange.upper.y;iy++)
          for (int ix=cellRange.lower.x;ix<=cellRange.upper.x;ix++) {
            int scalarIdx
              = localIdx.x
              + localSize.x * localIdx.y
              + localSize.x * localSize.y * localIdx.z;
            vec3 localIdx = vec3i(ix,iy,iz)-brick->voxelRange.lower;
            valueRange.extend(scalars[scalarIdx]);
          }
      shard.lower.w = valueRange.lower;
      shard.upper.w = valueRange.upper;
      results.push_back(shard);
    } else {
      int dim = arg_max(cellRange.size());
      int Nl = numShards/2;
      int Nr = numShards - Nl;
      int mid = cellRange.lower[dim] + size_t(Nl * cellRange.size()[dim]) / numShards;
      box3i lRange = cellRange; lRange.upper[dim] = mid;
      box3i rRange = cellRange; rRange.lower[dim] = mid;
      
      makeShards(result,brick,scalars,lRange,Nl);
      makeShards(result,brick,scalars,rRange,Nr);
    }
  }
  
  std::vector<box4f> Brick::makeShards(int numShards, const float *scalars)
  {
    std::vector<box4f> result;
    makeShards(results,this,scalars,this->cellRange,numShards);
  }
#endif

  // #if VOPAT_UMESH
  //   std::vector<box4f> Brick::computeShards(int numShards)
  //   {
  //     return {}
  //   }
  // #else
  
  //   std::vector<box4f> Brick::computeShards(int numShards)
  //   {
//     return computeShards(this,cellRange,numShards);  
//   }
// #endif    

  // float clamp01(float f)
  // { return min(1.f,max(0.f,f)); }

#if VOPAT_UMESH
  void Brick::load(const std::string &fileName)
  {
    umesh = umesh::UMesh::loadFrom(fileName);
    valueRange = {};
    for (auto v : umesh->perVertex->values)
      valueRange.extend(v);

    PRINT(umesh->toString());
#if 0
    const int maxTets = (128
                         + 0*64
                         + 1*32
                         // + 1*16
                         // + 1*8
                         // + 1*4
                         + 1*1
                         )*1024*1024;
    if (umesh->tets.size() > maxTets) {
      for (int i=0;i<maxTets;i++) {
        umesh->tets[i] = umesh->tets[umesh->tets.size()-maxTets+i];
      }
      umesh->tets.resize(maxTets);
    }
#endif
  }
#else
  Brick::Brick(int ID,
               /*! total num voxels in the *entire* model */
               const vec3i &numVoxelsTotal,
               /*! desired range of *cells* (not voxels) to load from this
                 volume, *including* the lower coordinates but *excluding* the
                 upper. 
                 
                 Eg, for a volume of 10x10 voxels (ie, 9x9 cells) the
                 range {(2,2),(4,4)} would span cells (2,2),(3,2),(3,2) and
                 (3,3); and to do that wouldread the voxels from (2,2) to
                 including (4,4) (ie, the brick would have 2x2 cells and 3x3
                 voxels. */
               const box3i &desiredCellRange)
    : ID(ID)
  {
    this->cellRange = desiredCellRange;
    this->voxelRange = {desiredCellRange.lower,desiredCellRange.upper+vec3i(1)};
    this->spaceRange.lower = vec3f(this->voxelRange.lower);
    this->spaceRange.upper = vec3f(this->voxelRange.upper-1);
    
    this->numVoxels = this->voxelRange.size();
    this->numCells  = this->voxelRange.size() - vec3i(1);
    this->numVoxelsParent = numVoxelsTotal;
  }
#endif

  template<typename T> float voxelToFloat(T t);

  template<> float voxelToFloat(float f) { return f; }
  template<> float voxelToFloat(uint8_t ui) { return ui / float(255.f); }
  template<> float voxelToFloat(uint16_t ui) { return ui / float((1<<16)-1); }
  
  std::string Brick::toString() const
  {
    std::stringstream ss;
#if VOPAT_UMESH
    ss << "Brick(domain=" << domain << ",umesh=";
    if (umesh)
      ss << umesh->toString();
    else
      ss << "null";
    ss << ")";
#else
    ss << "Brick{voxels begin/end=" << voxelRange << ", space="<<spaceRange << ", numVox=" << numVoxels << "}";
#endif
    return ss.str();
  }

#if VOPAT_UMESH
#else
  /*! load a given time step and variable's worth of voxels from given file name */
  void Brick::load(CUDAArray<float> &devMem,
                   const std::string &fileName)
  {
    std::ifstream in(fileName,std::ios::binary);
    if (!in) throw std::runtime_error("could not open '"+fileName+"'");
    std::vector<float> slice;
    slice.resize(numVoxels.x*numVoxels.y);
    devMem.resize(numVoxels.x*size_t(numVoxels.y)*numVoxels.z);
    int lastPing = -1;
    for (int z=0;z<numVoxels.z;z++) {
      in.read((char*)slice.data(),slice.size()*sizeof(float));
      const int increments = 1;// print in 5% intervals
      int percentDone = increments * int(z * 100.f / (increments*numVoxels.z));
      if (percentDone != lastPing) {
        printf("\r(%i) loaded %i%%   ",ID,percentDone);fflush(0);
        lastPing = percentDone;
      }
      devMem.upload(slice,z*slice.size());
    }
    printf("\r(%i) loaded 100%%...\n",ID);fflush(0);
  }

  /*! load a given time step and variable's worth of voxels from given file name */
  void Brick::load(std::vector<float> &hostMem,
                   const std::string &fileName)
  {
    std::ifstream in(fileName,std::ios::binary);
    if (!in) throw std::runtime_error("could not open '"+fileName+"'");
    std::vector<float> slice;
    slice.resize(numVoxels.x*numVoxels.y);
    hostMem.resize(numVoxels.x*size_t(numVoxels.y)*numVoxels.z);
    int lastPing = -1;
    for (int z=0;z<numVoxels.z;z++) {
      in.read((char*)slice.data(),slice.size()*sizeof(float));
      const int increments = 1;// print in 5% intervals
      int percentDone = increments * int(z * 100.f / (increments*numVoxels.z));
      if (percentDone != lastPing) {
        printf("\r(%i) loaded %i%%   ",ID,percentDone);fflush(0);
        lastPing = percentDone;
      }
      memcpy(hostMem.data()+z*slice.size(),slice.data(),slice.size()*sizeof(float));
    }
    printf("\r(%i) loaded 100%%...\n",ID);fflush(0);
  }
  
  template<typename T>
  std::vector<float> Brick::loadRegionRAW(const std::string &rawFileName)
  {
    std::ifstream in(rawFileName,std::ios::binary);
    if (!in) throw std::runtime_error("could not open '"+rawFileName+"'");
    std::vector<float> voxels(volume(numVoxels));
    std::vector<T> line(numVoxels.x);
    for (int iz=0;iz<numVoxels.z;iz++)
      for (int iy=0;iy<numVoxels.y;iy++) {
        size_t idxOfs
          = (voxelRange.lower.z+iz) * size_t(numVoxelsParent.x) * size_t(numVoxelsParent.y)
          + (voxelRange.lower.y+iy) * size_t(numVoxelsParent.x)
          + (voxelRange.lower.x+ 0) * size_t(1);
        in.seekg(idxOfs*sizeof(T),std::ios::beg);
        in.read((char *)line.data(),
                numVoxels.x * sizeof(T));
        if (!in || in.bad()) throw std::runtime_error("error reading from '"+rawFileName+"'");
        // PING; PRINT(idxOfs); PRINT(line[0]);
        for (int ix=0;ix<numVoxels.x;ix++)
          voxels[ix+numVoxels.x*(iy+numVoxels.y*size_t(iz))] = voxelToFloat(line[ix]);
      }
    return voxels;
  }

  template std::vector<float> Brick::loadRegionRAW<float>(const std::string &rawFileName);
  template std::vector<float> Brick::loadRegionRAW<uint8_t>(const std::string &rawFileName);   
  template std::vector<float> Brick::loadRegionRAW<uint16_t>(const std::string &rawFileName);   
#endif
}
