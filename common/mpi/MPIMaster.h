// ======================================================================== //
// Copyright 2018-2022 Ingo Wald                                            //
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

#include "common/mpi/MPICommon.h"
#include "common/mpi/MPIRenderer.h"

namespace vopat {

  // /*! mpi rendering interface for the master rank that runs the
  //     app/viewer. basically this will take the 'rendering' commands,
  //     broadcast them to all ranks, and on each rank (including the
  //     master) excute them to the virtual renderer */
  // struct MPIMaster : public MPICommon {
  //   MPIMaster(MPIBackend &mpi, MPIRenderer *renderer);

  //   void screenShot();
  //   void resetAccumulation();
  //   void terminate();
  //   void renderFrame(uint32_t *fbPointer);
  //   void resizeFrameBuffer(const vec2i &newSize);
  //   void setCamera(const vec3f &from,
  //                  const vec3f &at,
  //                  const vec3f &up,
  //                  float fovy);
  //   void setTransferFunction(const std::vector<vec4f> &cm,
  //                            const interval<float> &range,
  //                            const float density);
  //   void setISO(int numActive,
  //               const std::vector<int> &active,
  //               const std::vector<float> &values,
  //               const std::vector<vec3f> &colors);

  //   void backdoor(const std::string &command);
  //   void setShadeMode(int mode);
  //   void setLights(float ambient,
  //                  const std::vector<MPIRenderer::DirectionalLight> &dirLights);
    
  //   MPIRenderer *renderer = nullptr;
  //   MPIBackend  &mpi;
  // };
  
} // ::vopat
