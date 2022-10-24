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

  std::string StructuredBrick::toString() const
  {
    std::stringstream ss;
    ss << "StructuredBrick #"<<ID << " voxels " << voxelRange << " / " << numVoxelsParent;
    return ss.str();
  }
  
  template<typename T>
  StructuredBrick::SP makeBrickRaw(int ID,
                                   const box3i &cellRange,
                                   const vec3i &numVoxelsParent,
                                   const std::string &rawFileName)
  {
    vec3i numVoxels = cellRange.size()+1;
    std::ifstream in(rawFileName,std::ios::binary);
    if (!in) throw std::runtime_error("could not open '"+rawFileName+"'");
    std::vector<float> voxels(volume(numVoxels));
    std::vector<T> line(numVoxels.x);
    const box3i voxelRange = { cellRange.lower, cellRange.upper+vec3i(1) };
    // range1f valueRange;
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
        for (int ix=0;ix<numVoxels.x;ix++) {
          float v = voxelToFloat(line[ix]);
          voxels[ix+numVoxels.x*(iy+numVoxels.y*size_t(iz))] = v;
          // valueRange.extend(v);
        }
      }

    StructuredBrick::SP brick = StructuredBrick::create(ID);
    brick->scalars = voxels;
    brick->cellRange = cellRange;
    brick->voxelRange = voxelRange;
    brick->spaceRange = { vec3f(voxelRange.lower), vec3f(voxelRange.upper) };
    brick->numVoxels = voxelRange.size();
    brick->numCells = cellRange.size();
    brick->numVoxelsParent = numVoxelsParent;
    return brick;
  }
  
  void StructuredBrick::recMakeShards(std::vector<Shard> &result,
                                   const box3i &cellRange,
                                   int numShards)
  {
    if (numShards == 1 || cellRange.size() == vec3i(1)) {
      Shard shard;
      shard.domain.lower.x = float(cellRange.lower.x);
      shard.domain.lower.y = float(cellRange.lower.y);
      shard.domain.lower.z = float(cellRange.lower.z);

      shard.domain.upper.x = float(cellRange.upper.x);
      shard.domain.upper.y = float(cellRange.upper.y);
      shard.domain.upper.z = float(cellRange.upper.z);

      range1f valueRange;
      const vec3i localSize = this->numVoxels;
      for (int iz=cellRange.lower.z;iz<=cellRange.upper.z;iz++)
        for (int iy=cellRange.lower.y;iy<=cellRange.upper.y;iy++)
          for (int ix=cellRange.lower.x;ix<=cellRange.upper.x;ix++) {
            vec3i localIdx = vec3i(ix,iy,iz)-this->voxelRange.lower;
            int scalarIdx
              = localIdx.x
              + localSize.x * localIdx.y
              + localSize.x * localSize.y * localIdx.z;
            valueRange.extend(scalars[scalarIdx]);
          }
      shard.valueRange = valueRange;
      result.push_back(shard);
    } else {
      int dim = arg_max(cellRange.size());
      int Nl = numShards/2;
      int Nr = numShards - Nl;
      int mid = cellRange.lower[dim] + size_t(Nl * cellRange.size()[dim]) / numShards;
      box3i lRange = cellRange; lRange.upper[dim] = mid;
      box3i rRange = cellRange; rRange.lower[dim] = mid;
      
      recMakeShards(result,lRange,Nl);
      recMakeShards(result,rRange,Nr);
    }
  }
  
  std::vector<Shard> StructuredBrick::makeShards(int numShards)
  {
    std::vector<Shard> result;
    recMakeShards(result,this->cellRange,numShards);
    return result;
  }

  // StructuredBrick::StructuredBrick(int ID,
  //                                  const std::string &constDataFileName)
  //   : Brick(ID)
  // {
  //   std::ifstream in(constDataFileName,std::ios::binary);
    
  //   read(in,cellRange);
  //   read(in,voxelRange);
  //   read(in,spaceRange);
  //   read(in,numVoxels);
  //   read(in,numCells);
  //   read(in,numVoxelsParent);
  //   read(in,scalars);
  //   // this->cellRange = desiredCellRange;
  //   // this->voxelRange = {desiredCellRange.lower,desiredCellRange.upper+vec3i(1)};
  //   // this->spaceRange.lower = vec3f(this->voxelRange.lower);
  //   // this->spaceRange.upper = vec3f(this->voxelRange.upper-1);
    
  //   // this->numVoxels = this->voxelRange.size();
  //   // this->numCells  = this->voxelRange.size() - vec3i(1);
  //   // this->numVoxelsParent = numVoxelsTotal;
  // }

  void StructuredBrick::writeConstantData(const std::string &fileName) const
  {
    std::ofstream out(fileName,std::ios::binary);
    write(out,cellRange);
    write(out,voxelRange);
    write(out,spaceRange);
    write(out,numVoxels);
    write(out,numCells);
    write(out,numVoxelsParent);
  }
  
  void StructuredBrick::writeTimeStep(const std::string &fileName) const
  {
    std::ofstream out(fileName,std::ios::binary);
    write(out,scalars);
  }

  void StructuredBrick::loadConstantData(const std::string &fileName)
  {
    std::ifstream in(fileName,std::ios::binary);
    read(in,cellRange);
    read(in,voxelRange);
    read(in,spaceRange);
    read(in,numVoxels);
    read(in,numCells);
    read(in,numVoxelsParent);
  }
  
  void StructuredBrick::loadTimeStep(const std::string &fileName)
  {
    std::ifstream in(fileName,std::ios::binary);
    read(in,scalars);
  }

  // ------------------------------------------------------------------
  // explicit template instantiations
  // ------------------------------------------------------------------
  
  template StructuredBrick::SP
  makeBrickRaw<float>(int ID,
                      const box3i &cellRange,
                      const vec3i &numVoxels,
                      const std::string &rawFileName);
  template StructuredBrick::SP
  makeBrickRaw<uint8_t>(int ID,
                        const box3i &cellRange,
                        const vec3i &numVoxels,
                        const std::string &rawFileName);
  template StructuredBrick::SP
  makeBrickRaw<uint16_t>(int ID,
                         const box3i &cellRange,
                         const vec3i &numVoxels,
                         const std::string &rawFileName);
}
