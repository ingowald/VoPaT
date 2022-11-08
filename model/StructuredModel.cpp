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
  
  void StructuredBrick::recMakeVolumeProxies(std::vector<VolumeProxy> &result,
                                             const box3i &cellRange,
                                             int numVolumeProxies)
  {
    if (numVolumeProxies == 1 || cellRange.size() == vec3i(1)) {
      VolumeProxy volumeProxy;
      volumeProxy.domain.lower.x = float(cellRange.lower.x);
      volumeProxy.domain.lower.y = float(cellRange.lower.y);
      volumeProxy.domain.lower.z = float(cellRange.lower.z);

      volumeProxy.domain.upper.x = float(cellRange.upper.x);
      volumeProxy.domain.upper.y = float(cellRange.upper.y);
      volumeProxy.domain.upper.z = float(cellRange.upper.z);

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
      volumeProxy.valueRange = valueRange;
      result.push_back(volumeProxy);
    } else {
      int dim = arg_max(cellRange.size());
      int Nl = numVolumeProxies/2;
      int Nr = numVolumeProxies - Nl;
      int mid = cellRange.lower[dim] + size_t(Nl * cellRange.size()[dim]) / numVolumeProxies;
      box3i lRange = cellRange; lRange.upper[dim] = mid;
      box3i rRange = cellRange; rRange.lower[dim] = mid;
      
      recMakeVolumeProxies(result,lRange,Nl);
      recMakeVolumeProxies(result,rRange,Nr);
    }
  }
  
  std::vector<VolumeProxy> StructuredBrick::makeVolumeProxies(int numVolumeProxies)
  {
    std::vector<VolumeProxy> result;
    recMakeVolumeProxies(result,this->cellRange,numVolumeProxies);
    return result;
  }

  void StructuredBrick::writeUnvaryingData(const std::string &fileName) const
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

  void StructuredBrick::loadUnvaryingData(const std::string &fileName)
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
