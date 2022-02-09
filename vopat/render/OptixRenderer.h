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

#include "vopat/render/Renderer.h"
#include "vopat/model/Model.h"

namespace vopat {

  struct OptixRenderer : public Renderer {
    OptixRenderer(CommBackend *comm,
                  ModelMeta::SP meta,
                  RankData::SP rankData,
                  int numSPP)
      : Renderer(comm,numSPP)
    { PING; }
    void render()  override { PING; }
    void resizeFrameBuffer(const vec2i &newSize)  override { PING; }
    void resetAccumulation()  override { PING; }
    void setCamera(const Camera &camera)  override { PING; }
  };
  
}
