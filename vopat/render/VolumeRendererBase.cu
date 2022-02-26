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

#include "VolumeRendererBase.h"

namespace vopat {

  VolumeRenderer::VolumeRenderer(Model::SP model,
                                 const std::string &baseFileName,
                                 int myRank)
    : model(model), myRank(myRank)
  {
    if (myRank < 0)
      return;
      
    // ------------------------------------------------------------------
    // upload per-rank boxes
    // ------------------------------------------------------------------
    std::vector<box3f> hostRankBoxes;
    for (auto brick : model->bricks)
      hostRankBoxes.push_back(brick->spaceRange);
    rankBoxes.upload(hostRankBoxes);
    globals.rankBoxes = rankBoxes.get();

    myBrick = model->bricks[myRank];
    const std::string fileName = Model::canonicalRankFileName(baseFileName,myRank);
#if 1
    myBrick->load(voxels,fileName);
#else
    std::vector<float> loadedVoxels = myBrick->load(fileName);
    voxels.upload(loadedVoxels);
#endif
      
    globals.volume.voxels = voxels.get();
    globals.volume.dims   = myBrick->numVoxels;//voxelRange.size();
    globals.myRegion      = myBrick->spaceRange;
    /* initialize to model value range; xf editor may mess with that
       later on */
    globals.xf.domain = model->valueRange;
    globals.myRank = myRank;
    globals.numRanks = model->bricks.size();
    
    initMacroCells();
  }

  void VolumeRenderer::initMacroCells()
  {
    globals.mc.dims = divRoundUp(myBrick->numCells,vec3i(mcWidth));
    mcData.resize(volume(mcDims));
    globals.mc.data  = mcData.get();
    
    initMacroCell<<<(dim3)mcDims,(dim3)vec3i(4)>>>
      (globals.mc.data,globals.mc.dims,mcWidth,
       voxels.get(),globals.volume.dims);
  }

  
  void VolumeRenderer::setTransferFunction(const std::vector<vec4f> &cm,
                                           const interval<float> &xfDomain,
                                           const float density)
  {
    if (myRank < 0) {
    } else {
      // __global__ void mapMacroCell(DeviceKernelsBase::MacroCell *mcData,
      //                              vec3i mcDims,
      //                              vec4f *xfValues,
      //                              int numXfValues,
      //                              interval<float> xfDomain);
      mapMacroCell<<<(dim3)mcDims,(dim3)vec3i(4)>>>
        (mcData.get(),mcDims,
         globals.xf.values,
         globals.xf.numValues,
         globals.xf.domain);
    }
  }
    
}

