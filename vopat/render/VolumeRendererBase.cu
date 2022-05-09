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

#include <string.h>
#include "VolumeRendererBase.h"

namespace vopat {

  VolumeRenderer::VolumeRenderer(Model::SP model,
                                 const std::string &baseFileName,
                                 int islandRank)
    : model(model), islandRank(islandRank)
  {
    if (islandRank < 0)
      return;
      
    // ------------------------------------------------------------------
    // upload per-rank boxes
    // ------------------------------------------------------------------
    std::vector<box3f> hostRankBoxes;
    for (auto brick : model->bricks)
      hostRankBoxes.push_back(
#if VOPAT_UMESH
                              brick->domain
#else
                              brick->spaceRange
#endif
                              );
    rankBoxes.upload(hostRankBoxes);
    globals.rankBoxes = rankBoxes.get();

    myBrick = model->bricks[islandRank];
    const std::string fileName = Model::canonicalRankFileName(baseFileName,islandRank);
#if VOPAT_UMESH
    myBrick->load(fileName);
    globals.myRegion      = myBrick->domain;
#else
# if VOPAT_VOXELS_AS_TEXTURE
    std::vector<float> hostVoxels;
    myBrick->load(hostVoxels,fileName);

    // Copy voxels to cuda array
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
    cudaExtent extent{(unsigned)myBrick->numVoxels.x,
                      (unsigned)myBrick->numVoxels.y,
                      (unsigned)myBrick->numVoxels.z};
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

    CUDA_CALL(CreateTextureObject(&globals.volume.texObj,&resourceDesc,&textureDesc,0));

    // 2nd texture object for nearest filtering
    textureDesc.filterMode       = cudaFilterModePoint;
    CUDA_CALL(CreateTextureObject(&globals.volume.texObjNN,&resourceDesc,&textureDesc,0));
# else
#  if 1
    myBrick->load(voxels,fileName);
#  else
    std::vector<float> loadedVoxels = myBrick->load(fileName);
    voxels.upload(loadedVoxels);
#  endif
    globals.volume.voxels = voxels.get();
# endif
    globals.volume.dims   = myBrick->numVoxels;//voxelRange.size();
    globals.myRegion      = myBrick->spaceRange;
#endif
    /* initialize to model value range; xf editor may mess with that
       later on */
    globals.xf.domain = model->valueRange;
    globals.islandRank = islandRank;
    globals.islandSize = model->bricks.size();
  
    globals.gradientDelta = vec3f(1.f);

    initMacroCells();
  }

  void VolumeRenderer::initMacroCells()
  {
#if VOPAT_UMESH
    std::cout << "need to rebuild macro cells .." << std::endl;
#else
    globals.mc.dims = divRoundUp(myBrick->numCells,vec3i(mcWidth));
    mcData.resize(volume(globals.mc.dims));
    globals.mc.data  = mcData.get();
    globals.mc.width = mcWidth;
  
    VoxelData voxelData = *(VoxelData*)&globals.volume;
    initMacroCell<<<(dim3)globals.mc.dims,(dim3)vec3i(4)>>>
      (globals.mc.data,globals.mc.dims,mcWidth,voxelData);
#endif
  }

  
  void VolumeRenderer::setTransferFunction(const std::vector<vec4f> &cm,
                                           const interval<float> &xfDomain,
                                           const float density)
  {
    if (islandRank < 0) {
    } else {
      colorMap.upload(cm);
      globals.xf.values = colorMap.get();
      globals.xf.numValues = cm.size();
      globals.xf.domain = xfDomain;
      globals.xf.density = density;

      mapMacroCell<<<(dim3)globals.mc.dims,(dim3)vec3i(4)>>>
        (mcData.get(),globals.mc.dims,
         globals.xf.values,
         globals.xf.numValues,
         globals.xf.domain);
    }
  }
    
}

