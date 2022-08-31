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
    static SP create(int ID,
                     const vec3i &numVoxelsTotal,
                     const box3i &desiredCellRange)
    { return std::make_shared<StructuredBrick>(ID,numVoxelsTotal,desiredCellRange); }

    StructuredBrick(int ID, const std::string &constDataFileName);
    // StructuredBrick(int ID,
    //                 /*! total num voxels in the *entire* model */
    //                 const vec3i &numVoxelsTotal,
    //                 /*! desired range of *cells* (not voxels) to load from this
    //                   volume, *including* the lower coordinates but *excluding* the
    //                   upper. 
            
    //                   Eg, for a volume of 10x10 voxels (ie, 9x9 cells) the
    //                   range {(2,2),(4,4)} would span cells (2,2),(3,2),(3,2) and
    //                   (3,3); and to do that wouldread the voxels from (2,2) to
    //                   including (4,4) (ie, the brick would have 2x2 cells and 3x3
    //                   voxels. */
    //                 const box3i &desiredCellRange);
    
    std::string toString() const overrride;

    /*! load a given time step and variable's worth of voxels from given file name */
    void load(CUDAArray<float> &devMem, const std::string &fileName);

    /*! load a given time step and variable's worth of voxels from given file name */
    void load(std::vector<float> &hostMem, const std::string &fileName);
    box3f getDomain() const { return spaceRange; }
    void write(std::ostream &out) const override;

    /*! loads this brick's voxels - ie, only a range of the full file - from a raw file */
    template<typename T=float>
    std::vector<float> loadRegionRAW(const std::string &rawFileName);
    std::vector<Shard> makeShards(int numShards) override;
    
    box3i cellRange;
    box3i voxelRange;
    box3f spaceRange;
    vec3i numVoxels;
    vec3i numCells;
    vec3i numVoxelsParent;
    std::vector<float> scalars;
  };

  template<typename T>
  StructuredBrick::SP makeBrickRaw(const box3i &cellRange,
                                   const vec3i &numVoxels,
                                   const std::string &rawFileName);
  
  struct StructuredModel : public Model {
    typedef std::shared_ptr<StructuredModel> SP;

    box3f getBounds() const {
      if (reduce_min(numVoxelsTotal) <= 0)
        throw std::runtime_error("invalid model...");
      return { vec3f(0.f), vec3f(numVoxelsTotal-1) };
    }
  };
  
}

