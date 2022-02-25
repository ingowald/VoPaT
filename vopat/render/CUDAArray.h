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

namespace vopat {
  
  template<typename T>
  struct CUDAArray {

    void resize(int64_t N)
    {
      if (N < 0)
        throw std::runtime_error("invalid array size!?");
      
      if (this->N == N) return;
      this->N = N;
      if (devMem) CUDA_CALL(Free(devMem));
      devMem = 0;
#if 0
      CUDA_CALL(Malloc(&devMem,N*sizeof(T)));
#else
      CUDA_CALL(MallocManaged(&devMem,N*sizeof(T)));
#endif
      assert(devMem);
    }

    inline size_t numBytes() const { return N * sizeof(T); }
    inline T operator*() const { return *devMem; }
    inline operator bool() const { return devMem; }
    
    std::vector<T> download() const
    {
      std::vector<T> host(N);
      CUDA_CALL(Memcpy(host.data(),devMem,N*sizeof(T),cudaMemcpyDefault));
      CUDA_SYNC_CHECK();
      return host;
    }
    
    inline void upload(const std::vector<T> &vt) {
      resize(vt.size());
      CUDA_CALL(Memcpy(devMem,vt.data(),vt.size()*sizeof(T),cudaMemcpyDefault));
      CUDA_SYNC_CHECK();
    }
    inline void upload(const std::vector<T> &vt, size_t ofsInElements) {
      resize(vt.size());
      CUDA_CALL(Memcpy(devMem+ofsInElements,vt.data(),vt.size()*sizeof(T),cudaMemcpyDefault));
      CUDA_SYNC_CHECK();
    }
    inline void bzero() {
      CUDA_CALL(Memset(devMem,0,N*sizeof(T)));
    }
    T *get() const { return devMem; }
    
    T     *devMem = 0;
    size_t N      = 0;
  };

}
