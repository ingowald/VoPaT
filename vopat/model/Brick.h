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

  /*! a "brick" refers to a sub-range of a larger (structured)
    volume, such that the entirety of all bricks conver all of the
    input volume's cells, and as such all share one "boundary"
    layer of voxels */
  struct Brick {
    typedef std::shared_ptr<Brick> SP;
    
    static SP create(int ID) { return std::make_shared<Brick>(ID); }
    static SP create(int ID,
                     const vec3i &numVoxelsTotal,
                     const box3i &desiredCellRange)
    { return std::make_shared<Brick>(ID,numVoxelsTotal,desiredCellRange); }
    
    Brick(int ID) : ID(ID) {};
    Brick(int ID,
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
          const box3i &desiredCellRange);

    std::string toString() const;
    
    /*! loads this brick's voxels - ie, only a range of the full file - from a raw file */
    template<typename T=float>
    std::vector<float> loadRegionRAW(const std::string &rawFileName);

    /*! load a given time step and variable's worth of voxels from given file name */
    std::vector<float> load(const std::string &fileName);
    
    /*! linear numbering of this brick, relative to all bricks in the parent model */
    const int ID;
    box3i voxelRange;
    box3i cellRange;
    box3f spaceRange;
    vec3i numVoxels;
    vec3i numCells;
    vec3i numVoxelsParent;
  };
  
}
