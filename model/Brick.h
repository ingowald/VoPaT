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

#include "common/CUDAArray.h"
#include "umesh/UMesh.h"

namespace vopat {

  using range1f = interval<float>;
  
  struct Shard {
    box3f domain;
    range1f valueRange;
  };
  
  /*! a "brick" refers to one of the parts of a distribtued
      model. this can be a umesh that's part of a larger umesh, or a
      set of structured-volume voxels that are a region of the model
      (possibly includign ghost layers), etc */
  struct Brick : public std::enable_shared_from_this<Brick> {
    typedef std::shared_ptr<Brick> SP;

    Brick(int ID) : ID(ID) {};

    template<typename T>
    std::shared_ptr<T> as() 
    { return std::dynamic_pointer_cast<T>(shared_from_this()); }

    virtual std::string toString() const = 0;
    
    // ------------------------------------------------------------------
    // interface for the BUILDER/SPLITTER 
    // ------------------------------------------------------------------
    virtual void writeConstantData(const std::string &fileName) const = 0;
    virtual void writeTimeStep(const std::string &fileName) const = 0;

    // ------------------------------------------------------------------
    // interface for the RENDERER
    // ------------------------------------------------------------------
    
    /*! load all of the time-step and variabel independent data */
    virtual void loadConstantData(const std::string &fileName) = 0;
    
    /*! load a given time step and variable's worth of voxels from given file name */
    virtual void loadTimeStep(const std::string &fileName) = 0;
    
    /*! on a given rank, create shards for _exactly this_ brick (ranks
        can then exchange those as required */
    virtual std::vector<Shard> makeShards(int numShards) = 0;
    
    const int ID;
  };

}
