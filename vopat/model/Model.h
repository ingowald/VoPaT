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

namespace vopat {

  struct ModelMeta {
    typedef std::shared_ptr<ModelMeta> SP;

    ModelMeta(const std::string &fileName);
    
    static SP load(const std::string &fileName)
    { return std::make_shared<ModelMeta>(fileName); }

    std::string      fileName;
    vec3i            numCells;
    vec3i            numVoxels;
    vec3i            numBricks;
    std::vector<int> brickOwner;
  };

  struct Brick {
    typedef std::shared_ptr<Brick> SP;

    SP load(ModelMeta::SP meta, const vec3i &brickIdx);

    box3i voxelRange;
    box3f spaceRange;
    vec3i numVoxels;
    vec3i numCells;
    std::vector<float> voxels;
  };
  
  struct RankData {
    typedef std::shared_ptr<RankData> SP;
  };

}
