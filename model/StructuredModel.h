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

  struct StructuredBrick : public Brick {
    typedef std::shared_ptr<StructuredBrick> SP;
    
    static SP create(int ID)
    { return std::make_shared<StructuredBrick>(ID); }

    StructuredBrick(int ID) : Brick(ID) {}
    
    std::string toString() const override;

    // ------------------------------------------------------------------
    // interface for the BUILDER/SPLITTER 
    // ------------------------------------------------------------------
    void writeUnvaryingData(const std::string &fileName) const override;
    void writeTimeStep(const std::string &fileName) const override;
    
    // ------------------------------------------------------------------
    // interface for the RENDERER
    // ------------------------------------------------------------------
    
    void loadUnvaryingData(const std::string &fileName) override;
    void loadTimeStep(const std::string &fileName) override;
    
    std::vector<Shard> makeShards(int numShards) override;

  private:
    /*! internal helper function for recursive subdividion when making shards */
    void recMakeShards(std::vector<Shard> &result,
                       const box3i &cellRange,
                       int numShardsForThisRange);
  public:
    box3i cellRange;
    box3i voxelRange;
    box3f spaceRange;
    vec3i numVoxels;
    vec3i numCells;
    vec3i numVoxelsParent;

    std::vector<float> scalars;
  };

  template<typename T>
  StructuredBrick::SP makeBrickRaw(int ID,
                                   const box3i &cellRange,
                                   const vec3i &numVoxels,
                                   const std::string &rawFileName);
  
  struct StructuredModel : public Model {
    typedef std::shared_ptr<StructuredModel> SP;
      
    StructuredModel() : Model("StructuredModel<float>") {}

    Brick::SP createBrick(int ID) override { return StructuredBrick::create(ID); }
    
    static StructuredModel::SP create() { return std::make_shared<StructuredModel>(); }
  };
  
}

