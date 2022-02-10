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

#include "vopat/model/Model.h"
#include <fstream>

namespace vopat {

  ModelMeta::ModelMeta(const std::string &fileName)
    : fileName(fileName)
  {
    std::ifstream in(fileName);
    in >> numCells.x >> numCells.y >> numCells.z;
    //    in >> numBlocks.x >> numBlocks.y >> numBlocks.z;
  }

  Brick::SP Brick::load(ModelMeta::SP meta,
                        const vec3i &blockIdx)
  {
    std::ifstream in(meta->fileName);

    Brick::SP brick = std::make_shared<Brick>();
    brick->voxelRange.lower = blockIdx * meta->numVoxels / meta->numBricks;
    brick->voxelRange.upper
      = min((blockIdx + vec3i(1)) * meta->numVoxels / meta->numBricks + vec3i(1),
            meta->numVoxels);
    brick->spaceRange.lower = vec3f(brick->voxelRange.lower) / vec3f(meta->numVoxels - 1);
    brick->spaceRange.upper = vec3f(brick->voxelRange.upper) / vec3f(meta->numVoxels - 1);
    
    brick->numVoxels = brick->voxelRange.size();
    brick->numCells  = brick->voxelRange.size() - vec3i(1);

    brick->voxels.resize(volume(brick->numVoxels));
    for (int iz=0;iz<brick->numVoxels.z;iz++)
      for (int iy=0;iy<brick->numVoxels.y;iy++) {
        size_t idxOfs
          = (brick->voxelRange.lower.z+iz) * size_t(meta->numVoxels.x) * size_t(meta->numVoxels.y)
          + (brick->voxelRange.lower.y+iy) * size_t(meta->numVoxels.x);
        in.seekg(idxOfs*sizeof(float),std::ios::beg);
        in.read((char *)(brick->voxels.data()
                         + iz * size_t(meta->numVoxels.x) * size_t(meta->numVoxels.y)
                         + iy * size_t(meta->numVoxels.x)),
                brick->numVoxels.x * sizeof(float));
      }
    return brick;
  }

}
