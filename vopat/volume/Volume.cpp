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
  
  Volume::SP Volume::createFrom(Model::SP model)
  {
    if (StructuredModel::SP sm = model->as<StructuredModel>())
      return StructuredVolume::create(sm);
    else if (UMeshModel::SP sm = model->as<UMeshModel>())
      return UMeshVolume::create(sm);
    else
      throw std::runtime_error("un-recognized model type !?");
  }

} // ::vopat

