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

#include "vopat/model/Brick.h"
#include <fstream>

namespace vopat {

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
  

  template<typename T> float voxelToFloat(T t);

  template<> float voxelToFloat(float f) { return f; }
  template<> float voxelToFloat(uint8_t ui) { return ui / float(255.f); }
  template<> float voxelToFloat(uint16_t ui) { return ui / float((1<<16)-1); }
  
  template<typename T>
  std::vector<float> Brick::loadRegionRAW(const std::string &rawFileName)
  {
    std::ifstream in(rawFileName,std::ios::binary);
    std::vector<float> voxels(volume(numVoxels));
    std::vector<T> line(numVoxels.x);
    PRINT(numVoxelsParent);
    PRINT(rawFileName);
    PRINT(voxelRange);
    PRINT(numVoxels);
    for (int iz=0;iz<numVoxels.z;iz++)
      for (int iy=0;iy<numVoxels.y;iy++) {
        size_t idxOfs
          = (voxelRange.lower.z+iz) * size_t(numVoxelsParent.x) * size_t(numVoxelsParent.y)
          + (voxelRange.lower.y+iy) * size_t(numVoxelsParent.x)
          + (voxelRange.lower.x+ 0) * size_t(1);
        in.seekg(idxOfs*sizeof(T),std::ios::beg);
        in.read((char *)line.data(),
                numVoxels.x * sizeof(T));
        // PING; PRINT(idxOfs); PRINT(line[0]);
        for (int ix=0;ix<numVoxels.x;ix++)
          voxels[ix+numVoxels.x*(iy+numVoxels.y*size_t(iz))] = voxelToFloat(line[ix]);
      }
    return voxels;
  }

  std::string Brick::toString() const
  {
    std::stringstream ss;
    ss << "Brick{voxels begin/end=" << voxelRange << ", space="<<spaceRange << ", numVox=" << numVoxels << "}";
    return ss.str();
  }

  float clamp01(float f)
  { return min(1.f,max(0.f,f)); }
  
  /*! load a given time step and variable's worth of voxels from given file name */
  std::vector<float> Brick::load(const std::string &fileName)
  {
    std::ifstream in(fileName,std::ios::binary);
    if (!in) throw std::runtime_error("could not open '"+fileName+"'");
    std::vector<float> loadedVoxels;
    loadedVoxels.resize(volume(numVoxels));
    in.read((char*)loadedVoxels.data(),volume(numVoxels)*sizeof(float));

    float lo = 0.f;
    float hi = 0.f;
    for (auto &v : loadedVoxels) {
      lo = min(lo,v);
      hi = max(hi,v);
      v = clamp01(v);
    }
    PRINT(lo);
    PRINT(hi);
    
    return loadedVoxels;
  }

  template std::vector<float> Brick::loadRegionRAW<float>(const std::string &rawFileName);
  template std::vector<float> Brick::loadRegionRAW<uint8_t>(const std::string &rawFileName);   
  template std::vector<float> Brick::loadRegionRAW<uint16_t>(const std::string &rawFileName);   
}
