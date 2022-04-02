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

#include "vopat/render/DistributedRendererBase.h"
#include "vopat/model/Model.h"

namespace vopat {
  
  /*! creates a renderer from the given name (e.g., "woodcock" or
    "cell-march") */
  Renderer *createRenderer(const std::string &rendererName,
                           CommBackend *comm,
                           Model::SP model,
                           const std::string &fileNameBase,
                           // this is the rank WITHIN THE ISLAND
                           int rank);
}
