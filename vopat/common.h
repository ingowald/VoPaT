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

#include "owl/common/math/AffineSpace.h"
#include "owl/common/arrayND/array3D.h"
#include "owl/common/parallel/parallel_for.h"
#include "owl/helper/cuda.h"
// std
#include <sstream>
#include <string>
#include <string.h>
#include <mutex>
#include <stdexcept>
#include <set>
#include <map>
#include <vector>
#include <queue>
#include "cuda_fp16.h"

#define NOTIMPLEMENTED throw std::runtime_error(std::string(__PRETTY_FUNCTION__)+" not implemented")

#define TERM_COLOR_RED "\033[1;31m"
#define TERM_COLOR_GREEN "\033[1;32m"
#define TERM_COLOR_YELLOW "\033[1;33m"
#define TERM_COLOR_BLUE "\033[1;34m"
#define TERM_COLOR_RESET "\033[0m"
#define TERM_COLOR_DEFAULT TERM_COLOR_RESET
#define TERM_COLOR_BOLD "\033[1;1m"
     
namespace vopat {
  using namespace owl;
  using namespace owl::common;
  
  namespace common {
    
    inline bool endsWith(const std::string &s, const std::string &suffix)
    {
      return s.substr(s.size()-suffix.size(),suffix.size()) == suffix;
    }

    inline int iDivUp(int a, int b) { return (a+b-1)/b; }
  }


  
#define CUDA_CALL(a) OWL_CUDA_CALL(a)
#define CUDA_SYNC_CHECK() OWL_CUDA_SYNC_CHECK()


  struct small_vec3f { half x, y, z; };
  
  inline __both__ float from_half(half h) { return (float)h; }

  inline __both__ vec3f from_half(small_vec3f v)
  {
    return { from_half(v.x),from_half(v.y),from_half(v.z) };
  }

  inline __both__ half to_half(float f)
  {
    half h = f;
    return h;
  }

  inline __both__ small_vec3f to_half(vec3f v)
  {
    return { to_half(v.x),to_half(v.y),to_half(v.z) };
  }
  
} // ::mini

