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

#include "common/mpi/MPIBackend.h"

namespace vopat {

  struct AppInterface {
    
    static void runWorker(int ac, char **av);
    
    AppInterface(int ac, char **av);

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
                   const std::vector<vec3f> &dirs,
                   const std::vector<vec3f> &pows);
    
  private:
    template<typename T>
    void toWorkers(const std::vector<T> &t);
    
    template<typename T>
    void toWorkers(const T &t);
    
    MPIBackend mpi;
  };

}
