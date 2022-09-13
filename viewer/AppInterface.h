// ======================================================================== //
// Copyright 2022-2022 Ingo Wald                                            //
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

#include "common/mpi/Comms.h"
#include "vopat/VopatRenderer.h"

namespace vopat {

  struct AppInterface {
    
    AppInterface(CommBackend *comm,
                 VopatRenderer::SP renderer);

    /*! the 'main loop' that receives and executes cmmands sent by the master */
    void runWorker();

    // ------------------------------------------------------------------
    // interface of how app talks to this rank-parallel renderer
    // ------------------------------------------------------------------
    void screenShot();
    void resetAccumulation();
    void terminate();
    void renderFrame(uint32_t *fbPointer);
    void resizeFrameBuffer(const vec2i &newSize);
    void setCamera(const vec3f &from,
                   const vec3f &at,
                   const vec3f &up,
                   float fovy);
    void setTransferFunction(const std::vector<vec4f> &cm,
                             const interval<float> &range,
                             const float density);
    void setISO(int numActive,
                const std::vector<int> &active,
                const std::vector<float> &values,
                const std::vector<vec3f> &colors);

    void setLights(float ambient,
                   const std::vector<DirectionalLight> &dirLights);
    
  private:
    template<typename T>
    void sendToWorkers(const std::vector<T> &t);
    
    template<typename T>
    void sendToWorkers(const T &t);
    
    /*! @{ command handlers - each corresponds to exactly one command
        sent my the master */
    void cmd_terminate();
    void cmd_renderFrame();
    void cmd_resizeFrameBuffer();
    void cmd_resetAccumulation();
    void cmd_setCamera();
    void cmd_setTransferFunction();
    void cmd_setISO();
    void cmd_setShadeMode();
    void cmd_setNodeSelection();
    void cmd_screenShot();
    void cmd_setLights();
    /* @} */

    template<typename T>
    void fromMaster(std::vector<T> &t);
    template<typename T>
    void fromMaster(T &t);
    
    CommBackend *comm;
    VopatRenderer::SP renderer;

  private:
    int eomIdentifierBase = 0x12345;
    void checkEndOfMessage();
    void sendEndOfMessage();
  };

}
