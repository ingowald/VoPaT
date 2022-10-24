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

#include "vopat/volume/StructuredVolume.h"
#include "vopat/LaunchParams.h"

namespace vopat {

  int StructuredVolume::mcWidth = 8;

  __global__ void buildMCs(MacroCell *mcData,
                           vec3i      mcDims,
                           int        mcWidth,
                           StructuredVolume::DD volume)
  {
    vec3i mcID(threadIdx.x+blockIdx.x*blockDim.x,
               threadIdx.y+blockIdx.y*blockDim.y,
               threadIdx.z+blockIdx.z*blockDim.z);
    
    if (mcID.x >= mcDims.x) return;
    if (mcID.y >= mcDims.y) return;
    if (mcID.z >= mcDims.z) return;
    
    int mcIdx = mcID.x + mcDims.x*(mcID.y + mcDims.y*mcID.z);
    auto &mc = mcData[mcIdx];

    /* compute begin/end of VOXELS for this macro-cell */
    vec3i begin = mcID*mcWidth;
    vec3i end = min(begin + mcWidth + /* plus one for tri-lerp!*/1,
                    volume.dims);
    interval<float> valueRange;
    for (int iz=begin.z;iz<end.z;iz++)
      for (int iy=begin.y;iy<end.y;iy++)
        for (int ix=begin.x;ix<end.x;ix++) {
          float f;
          tex3D(&f,volume.texObjNN,ix,iy,iz);
          valueRange.extend(f);
// #else
//           valueRange.extend(volume.voxels[ix+volume.dims.x*(iy+volume.dims.y*size_t(iz))]);
// #endif
        }
    mc.inputRange = valueRange;
    mc.maxOpacity = 1.f;
  }

  void StructuredVolume::buildMCs(MCGrid &mcGrid) 
  {
    std::cout << OWL_TERMINAL_BLUE
              << "#vopat.structured: building macro cells .."
              << OWL_TERMINAL_DEFAULT
              << std::endl;
    PRINT(globals.dims);
    mcGrid.dd.dims = divRoundUp(globals.dims,vec3i(mcWidth));
    vec3ui bs = 4;
    vec3ui nb = divRoundUp(vec3ui(mcGrid.dd.dims),bs);
    PRINT(nb);
    PRINT(bs);
    
    mcGrid.cells.resize(owl::common::volume(mcGrid.dd.dims));
    dim3 _nb{nb.x,nb.y,nb.z};
    dim3 _bs{bs.x,bs.y,bs.z};
    vopat::buildMCs<<<_nb,_bs>>>
      (mcGrid.cells.get(),
       mcGrid.dd.dims,
       mcWidth,
       globals);
    CUDA_SYNC_CHECK();
    std::cout << OWL_TERMINAL_GREEN
              << "#vopat.structured: done building macro cells .."
              << OWL_TERMINAL_DEFAULT
              << std::endl;
  }
    
  void StructuredVolume::build(OWLContext owl,
                               OWLModule owlDevCode) 
  {
    std::vector<float> &hostVoxels = myBrick->scalars;
    
    // Copy voxels to cuda array
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
    cudaExtent extent{(unsigned)myBrick->numVoxels.x,
                      (unsigned)myBrick->numVoxels.y,
                      (unsigned)myBrick->numVoxels.z};
    cudaArray_t voxelArray;
    CUDA_CALL(Malloc3DArray(&voxelArray,&desc,extent,0));
    cudaMemcpy3DParms copyParms;
    memset(&copyParms,0,sizeof(copyParms));
    copyParms.srcPtr = make_cudaPitchedPtr(hostVoxels.data(),
                                           (size_t)myBrick->numVoxels.x*sizeof(float),
                                           (size_t)myBrick->numVoxels.x,
                                           (size_t)myBrick->numVoxels.y);
    copyParms.dstArray = voxelArray;
    copyParms.extent   = extent;
    copyParms.kind     = cudaMemcpyHostToDevice;
    CUDA_CALL(Memcpy3D(&copyParms));

    // Create a texture object
    cudaResourceDesc resourceDesc;
    memset(&resourceDesc,0,sizeof(resourceDesc));
    resourceDesc.resType         = cudaResourceTypeArray;
    resourceDesc.res.array.array = voxelArray;

    cudaTextureDesc textureDesc;
    memset(&textureDesc,0,sizeof(textureDesc));
    textureDesc.addressMode[0]   = cudaAddressModeClamp;
    textureDesc.addressMode[1]   = cudaAddressModeClamp;
    textureDesc.addressMode[2]   = cudaAddressModeClamp;
    textureDesc.filterMode       = cudaFilterModeLinear;
    textureDesc.readMode         = cudaReadModeElementType;
    textureDesc.normalizedCoords = false;

    CUDA_CALL(CreateTextureObject(&globals.texObj,&resourceDesc,&textureDesc,0));

    // 2nd texture object for nearest filtering in macro cell generation
    textureDesc.filterMode       = cudaFilterModePoint;
    CUDA_CALL(CreateTextureObject(&globals.texObjNN,&resourceDesc,&textureDesc,0));
    globals.dims = myBrick->numVoxels;
  }
  
  void StructuredVolume::setDD(OWLLaunchParams lp) 
  {
    owlParamsSetRaw(lp,"volumeSampler.structured",&globals);
  }
  
  void StructuredVolume::addLPVars(std::vector<OWLVarDecl> &lpVars) 
  {
    lpVars.push_back({"volumeSampler.structured",OWL_USER_TYPE(DD),
                      OWL_OFFSETOF(LaunchParams,volumeSampler.structured)});
  }
    
}
