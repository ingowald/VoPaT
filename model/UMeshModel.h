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
#include <umesh/UMesh.h>

namespace vopat {

  using umesh::UMesh;
  
  /*! this is for a spatially patitioned umesh, so each brick has a
    domain that doesn't overlap other bricks */
  struct UMeshBrick : public Brick {
    typedef std::shared_ptr<UMeshBrick> SP;
    
    static SP create(int ID)
    { return std::make_shared<UMeshBrick>(ID); }

    UMeshBrick(int ID) : Brick(ID) {}
    
    std::string toString() const override;

    // ------------------------------------------------------------------
    // interface for the BUILDER/SPLITTER 
    // ------------------------------------------------------------------
    void writeConstantData(const std::string &fileName) const override;
    void writeTimeStep(const std::string &fileName) const override;

    // ------------------------------------------------------------------
    // interface for the RENDERER
    // ------------------------------------------------------------------
    
    /*! load all of the time-step and variabel independent data */
    void loadConstantData(const std::string &fileName) override;
    
    /*! load a given time step and variable's worth of voxels from given file name */
    void loadTimeStep(const std::string &fileName) override;
    
    /*! create given number of shards to describe the currently loaded data */
    std::vector<Shard> makeShards(int numShards) override;

    umesh::UMesh::SP umesh;
    box3f domain;
  };

  struct UMeshModel : public Model {
    typedef std::shared_ptr<UMeshModel> SP;

    UMeshModel() : Model("UMeshModel/Spatial") {}
    
    Brick::SP createBrick(int ID) override { return UMeshBrick::create(ID); }
    
    static UMeshModel::SP create() { return std::make_shared<UMeshModel>(); }
  };
  
}

