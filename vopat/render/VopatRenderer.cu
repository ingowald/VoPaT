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

#include "vopat/render/VopatRenderer.h"
#include "3rdParty/stb_image//stb/stb_image_write.h"
#include "3rdParty/stb_image//stb/stb_image.h"
#include "owl/owl_device.h"
#include <sstream>

namespace vopat {

  __global__ void writeLocalFB(vec2i fbSize,
                               small_vec3f *localFB,
                               vec3f *accumBuffer,
                               int numAccumFrames)
  {
    int ix = threadIdx.x + blockIdx.x*blockDim.x;
    int iy = threadIdx.y + blockIdx.y*blockDim.y;
    if (ix >= fbSize.x) return;
    if (iy >= fbSize.y) return;

    int i = ix + iy * fbSize.x;
    
    localFB[i] = to_half(accumBuffer[i] * (1.f/(numAccumFrames)));
  }
    
}
