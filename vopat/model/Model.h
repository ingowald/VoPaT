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

#include "vopat/model/Brick.h"

namespace vopat {

  /*! a model made up of multiple bricks; usually one per rank */
  struct Model {
    typedef std::shared_ptr<Model> SP;

    static SP create() { return std::make_shared<Model>(); }
    static SP load(const std::string &fileName);

    void save(const std::string &fileName);

    /*! given a base file name prefix (including directory name, if
        desired), return a canonical file name for the master model
        file */
    static std::string canonicalMasterFileName(const std::string &baseName);
    
    /*! given a base file name prefix (including directory name, if
        desired), return a canonical file name for the data file for
        the 'rankID'th rank */
    static std::string canonicalRankFileName(const std::string &baseName,
                                             int rankID,
                                             const std::string &variable = "unknown",
                                             int timeStep = 0);

    std::vector<Brick::SP>   bricks;
    
    vec3i                  numVoxelsTotal;
  };

}
