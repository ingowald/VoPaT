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
#include "vopat/volume/UMeshVolume.h"

namespace vopat {
  
  Volume::SP Volume::createFrom(Brick::SP brick)
  {
    if (!brick)
      // this is OK - we're probably on master, which doesn't have a brick
      return {};
    
    if (StructuredBrick::SP typedBrick = brick->as<StructuredBrick>())
      return StructuredVolume::create(typedBrick);
    else if (UMeshBrick::SP typedBrick = brick->as<UMeshBrick>())
      return UMeshVolume::create(typedBrick);
    else
      throw std::runtime_error("un-recognized model type !?");
  }

  void Volume::setTransferFunction(const std::vector<vec4f> &cm,
                                   const interval<float> &xfDomain,
                                   const float density)
  {
    if (!brick)
      // master ...
      return;
    
    xf.colorMap.upload(cm);
    // globals.xf.values = colorMap.get();
    // xf.numValues = cm.size();
    xf.domain = xfDomain;
    xf.density = density;

    xfGlobals.values    = this->xf.colorMap.get();
    xfGlobals.numValues = this->xf.colorMap.N;
    xfGlobals.domain    = this->xf.domain;
    xfGlobals.density   = this->xf.density;
  }
    

} // ::vopat

