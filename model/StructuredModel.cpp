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

#include "model/StructuredModel.h"
#include "model/IO.h"
#include <fstream>

namespace vopat {

  template<typename T> float voxelToFloat(T t);

  template<> float voxelToFloat(float f) { return f; }
  template<> float voxelToFloat(uint8_t ui) { return ui / float(255.f); }
  template<> float voxelToFloat(uint16_t ui) { return ui / float((1<<16)-1); }

  
  template<typename T>
  StructuredBrick::SP makeBrickRaw(const box3i &cellRange,
                                   const vec3i &numVoxelsParent,
                                   const std::string &rawFileName)
  {
    vec3i numVoxels = cellRange.size()+1;
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

  StructuredBrick::StructuredBrick(int ID,
                                   const std::string &constDataFileName)
    : Brick(ID)
  {
    std::ifstream in(constDataFileName,std::ios::binary);
    
    read(in,cellRange);
    read(in,voxelRange);
    read(in,spaceRange);
    read(in,numVoxels);
    read(in,numCells);
    read(in,numVoxelsParent);
    read(in,scalars);
    // this->cellRange = desiredCellRange;
    // this->voxelRange = {desiredCellRange.lower,desiredCellRange.upper+vec3i(1)};
    // this->spaceRange.lower = vec3f(this->voxelRange.lower);
    // this->spaceRange.upper = vec3f(this->voxelRange.upper-1);
    
    // this->numVoxels = this->voxelRange.size();
    // this->numCells  = this->voxelRange.size() - vec3i(1);
    // this->numVoxelsParent = numVoxelsTotal;
  }

  void StructuredBrick::write(std::ostream &out) const
  {
    write(out,cellRange);
    write(out,voxelRange);
    write(out,spaceRange);
    write(out,numVoxels);
    write(out,numCells);
    write(out,numVoxelsParent);
    write(out,scalars);
  }
  
  template StructuredBrick::SP Brick::makeBrickRaw<float>(const box3i &cellRange,
                                                          const vec3i &numVoxels,
                                                          const std::string &rawFileName);
  template StructuredBrick::SP Brick::makeBrickRaw<uint8_t>(const box3i &cellRange,
                                                            const vec3i &numVoxels,
                                                         const std::string &rawFileName);
  template StructuredBrick::SP Brick::makeBrickRaw<uint16_t>(const box3i &cellRange,
                                                            const vec3i &numVoxels,
                                                            const std::string &rawFileName);
}
