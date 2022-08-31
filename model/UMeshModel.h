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

#include "model/Model.h"

namespace vopat {

  struct UMeshBrick : public Brick {
    std::string toString() const overrride;

    /*! load a given time step and variable's worth of voxels from given file name */
    void load(const std::string &fileName);
    box3f getDomain() const { return domain; }
    std::vector<box4f> makeShards(int numShards) override;
    void write(std::ostream &out) const override;

    umesh::UMesh::SP umesh;
    box3f domain;
  };

  struct UMeshModel : public Model {
    typedef std::shared_ptr<UMeshModel> SP;

    box3f getBounds() const {
      box3f bounds;
      for (auto brick : bricks)
        bounds.extend(brick->domain);
      return bounds;
    }
  };
  
}

