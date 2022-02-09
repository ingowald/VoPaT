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
#include "vopat/render/Comms.h"
#include "vopat/render/Camera.h"

// #include "vopat/scene/Scene.h"
// #include "vopat/scene/PartialScene.h"
// #include "vopat/scene/MasterScene.h"
// #include "miniScene/Serialized.h"

// #include "vopat/common/RankInfo.h"
// #include "vopat/common/JobTally.h"
// #include "vopat/common/RayState.h"
// #include "vopat/common/Camera.h"

// #include "vopat/render/Comms.h"
// #include "vopat/render/LaunchData.h"

// ------------------------------------------------------------------
// kernels
// ------------------------------------------------------------------
// #include "composeRegion.h"

namespace vopat {

  struct Renderer {

    static std::string screenShotFileName;

    Renderer(CommBackend *comm,
             int numSPP);
    
    int myRank() { return comm->isMaster?-1:comm->worker.withinIsland->rank; }
    
    virtual void render() = 0;
    virtual void resizeFrameBuffer(const vec2i &newSize) = 0;
    virtual void resetAccumulation() = 0;
    virtual void setCamera(const Camera &camera) = 0;
    vec3f *getLocalAccumBuffer() const { return localAccumBuffer; }
    
    vec3f          *localAccumBuffer;
    vec2i           fbSize;
    int             accumID = 0;
    const int       numSPP;
    CommBackend    *const comm;
  };

}
