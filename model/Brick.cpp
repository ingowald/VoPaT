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

// #if 0
//   std::string Brick::toString() const
//   {
//     std::stringstream ss;
// #if VOPAT_UMESH
//     ss << "Brick(domain=" << domain << ",umesh=";
//     if (umesh)
//       ss << umesh->toString();
//     else
//       ss << "null";
//     ss << ")";
// #else
//     ss << "Brick{voxels begin/end=" << voxelRange << ", space="<<spaceRange << ", numVox=" << numVoxels << "}";
// #endif
//     return ss.str();
//   }

  // /*! load a given time step and variable's worth of voxels from given file name */
  // void Brick::load(CUDAArray<float> &devMem,
  //                  const std::string &fileName)
  // {
  //   std::ifstream in(fileName,std::ios::binary);
  //   if (!in) throw std::runtime_error("could not open '"+fileName+"'");
  //   std::vector<float> slice;
  //   slice.resize(numVoxels.x*numVoxels.y);
  //   devMem.resize(numVoxels.x*size_t(numVoxels.y)*numVoxels.z);
  //   int lastPing = -1;
  //   for (int z=0;z<numVoxels.z;z++) {
  //     in.read((char*)slice.data(),slice.size()*sizeof(float));
  //     const int increments = 1;// print in 5% intervals
  //     int percentDone = increments * int(z * 100.f / (increments*numVoxels.z));
  //     if (percentDone != lastPing) {
  //       printf("\r(%i) loaded %i%%   ",ID,percentDone);fflush(0);
  //       lastPing = percentDone;
  //     }
  //     devMem.upload(slice,z*slice.size());
  //   }
  //   printf("\r(%i) loaded 100%%...\n",ID);fflush(0);
  // }

  // /*! load a given time step and variable's worth of voxels from given file name */
  // void Brick::load(std::vector<float> &hostMem,
  //                  const std::string &fileName)
  // {
  //   std::ifstream in(fileName,std::ios::binary);
  //   if (!in) throw std::runtime_error("could not open '"+fileName+"'");
  //   std::vector<float> slice;
  //   slice.resize(numVoxels.x*numVoxels.y);
  //   hostMem.resize(numVoxels.x*size_t(numVoxels.y)*numVoxels.z);
  //   int lastPing = -1;
  //   for (int z=0;z<numVoxels.z;z++) {
  //     in.read((char*)slice.data(),slice.size()*sizeof(float));
  //     const int increments = 1;// print in 5% intervals
  //     int percentDone = increments * int(z * 100.f / (increments*numVoxels.z));
  //     if (percentDone != lastPing) {
  //       printf("\r(%i) loaded %i%%   ",ID,percentDone);fflush(0);
  //       lastPing = percentDone;
  //     }
  //     memcpy(hostMem.data()+z*slice.size(),slice.data(),slice.size()*sizeof(float));
  //   }
  //   printf("\r(%i) loaded 100%%...\n",ID);fflush(0);
  // }
  
  // template<typename T>
  // std::vector<float> Brick::loadRegionRAW(const std::string &rawFileName)
  // {
  //   std::ifstream in(rawFileName,std::ios::binary);
  //   if (!in) throw std::runtime_error("could not open '"+rawFileName+"'");
  //   std::vector<float> voxels(volume(numVoxels));
  //   std::vector<T> line(numVoxels.x);
  //   for (int iz=0;iz<numVoxels.z;iz++)
  //     for (int iy=0;iy<numVoxels.y;iy++) {
  //       size_t idxOfs
  //         = (voxelRange.lower.z+iz) * size_t(numVoxelsParent.x) * size_t(numVoxelsParent.y)
  //         + (voxelRange.lower.y+iy) * size_t(numVoxelsParent.x)
  //         + (voxelRange.lower.x+ 0) * size_t(1);
  //       in.seekg(idxOfs*sizeof(T),std::ios::beg);
  //       in.read((char *)line.data(),
  //               numVoxels.x * sizeof(T));
  //       if (!in || in.bad()) throw std::runtime_error("error reading from '"+rawFileName+"'");
  //       // PING; PRINT(idxOfs); PRINT(line[0]);
  //       for (int ix=0;ix<numVoxels.x;ix++)
  //         voxels[ix+numVoxels.x*(iy+numVoxels.y*size_t(iz))] = voxelToFloat(line[ix]);
  //     }
  //   return voxels;
  // }

}
