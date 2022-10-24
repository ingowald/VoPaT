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

    // // 2nd texture object for nearest filtering
    // textureDesc.filterMode       = cudaFilterModePoint;
    // CUDA_CALL(CreateTextureObject(&globals.texObjNN,&resourceDesc,&textureDesc,0));
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
