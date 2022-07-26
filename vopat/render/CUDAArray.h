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

    /*! number of bytes allocated on the device */
    inline size_t numBytes() const { return N * sizeof(T); }

    /*! returns a (device side!-)reference to the first element. This
      primarily exists to allow things like "sizeof(*myDevMem)"; the
      address should not be dereferenced on the host */
    inline T operator*() const { return *devMem; }

    /*! returns if the device-memory is non-null */
    inline operator bool() const { return devMem; }
    
    /*! re-sizes device memory to specified number of elements,
      invalidating the current content. if device mem is already of
      exactly this size this is a no-op, otherwise old memory will
      be freed and new one allocated */
    void resize(int64_t N);

    /*! allocates a host-vector of the same size, downloads device
      data, and returns that vector */
    std::vector<T> download() const;

    /*! resizes (if required) the device memory to the same size of
      the vector, then uploads this host data to device */
    inline void upload(const std::vector<T> &vt);

    /*! upload host vector to given offset (counted in elements, not
      in bytes). Note that uplike resize(vector<>) this variant will
      _not_ do any resize, so this requires the device vector to be
      pre-allocated to sufficient size */
    inline void upload(const std::vector<T> &vt, size_t ofsInElements);

    /*! clears to allocated memory region to 0 */
    inline void bzero();
    T *get() const { return devMem; }

    /*! allocated device pointer. this can change upon resizing. */
    T     *devMem = 0;
    size_t N      = 0;
  };

  /*! clears to allocated memory region to 0 */
  template<typename T>
  inline void CUDAArray<T>::bzero()
  {
    CUDA_CALL(Memset(devMem,0,N*sizeof(T)));
  }

  /*! upload host vector to given offset (counted in elements, not
    in bytes). Note that uplike resize(vector<>) this variant will
    _not_ do any resize, so this requires the device vector to be
    pre-allocated to sufficient size */
  template<typename T>
  inline void CUDAArray<T>::upload(const std::vector<T> &vt, size_t ofsInElements)
  {
    assert((ofsInElements + vt.size()) <= N);
    CUDA_CALL(Memcpy(devMem+ofsInElements,vt.data(),vt.size()*sizeof(T),cudaMemcpyDefault));
    CUDA_SYNC_CHECK();
  }

  /*! resizes (if required) the device memory to the same size of
    the vector, then uploads this host data to device */
  template<typename T>
  inline void CUDAArray<T>::upload(const std::vector<T> &vt)
  {
    resize(vt.size());
    PRINT(vt.size());
    PRINT(sizeof(T));
    size_t sz = vt.size() * sizeof(T);
    PRINT((int*)sz);
    CUDA_CALL(Memcpy(devMem,vt.data(),vt.size()*sizeof(T),cudaMemcpyDefault));
    CUDA_SYNC_CHECK();
  }

  /*! re-sizes device memory to specified number of elements,
    invalidating the current content. if device mem is already of
    exactly this size this is a no-op, otherwise old memory will
    be freed and new one allocated */
  template<typename T>
  void CUDAArray<T>::resize(int64_t N)
  {
    if (N < 0)
      throw std::runtime_error("invalid array size!?");
      
    if (this->N == N) return;
    this->N = N;
    if (devMem) CUDA_CALL(Free(devMem));
    devMem = 0;
#if 1
    CUDA_CALL(Malloc(&devMem,N*sizeof(T)));
#else
    CUDA_CALL(MallocManaged(&devMem,N*sizeof(T)));
#endif
    assert(devMem);
  }

  /*! allocates a host-vector of the same size, downloads device
    data, and returns that vector */
  template<typename T>
  std::vector<T> CUDAArray<T>::download() const
  {
    std::vector<T> host(N);
    CUDA_CALL(Memcpy(host.data(),devMem,N*sizeof(T),cudaMemcpyDefault));
    CUDA_SYNC_CHECK();
    return host;
  }
} // ::vopat
