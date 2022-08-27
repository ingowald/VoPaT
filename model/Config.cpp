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

#include "model/Config.h"
#include "model/IO.h"

namespace vopat {

  constexpr size_t modelConfigVersion   = 4ull;
  constexpr size_t modelConfigMagicBase = 0x12354522ull;
  constexpr size_t modelConfigMagic     = modelConfigMagicBase + modelConfigVersion;
  
  void ModelConfig::save(const std::string &fileName)
  {
    std::ofstream out(fileName,std::ios::binary);
    size_t magic = modelConfigMagic;
    write(out,magic);
    write(out,xf.absDomain);
    write(out,xf.relDomain);
    write(out,xf.colorMap);
    write(out,xf.opacityScale);
    write(out,iso.active);
    write(out,iso.colors);
    write(out,iso.values);
    write(out,camera);
    write(out,lights.ambient);
    write(out,lights.directional);
    PRINT(lights.directional.size());
    if (lights.directional.size() > 0)
      PRINT(lights.directional[0].dir);
  }
  
  ModelConfig ModelConfig::load(const std::string &fileName)
  {
    std::ifstream in(fileName,std::ios::binary);
    size_t magic;
    ModelConfig mc;
    read(in,magic);
    if (magic < modelConfigMagicBase + 3ull) {
      PRINT(modelConfigMagic);
      PRINT(magic);
      throw std::runtime_error("invalid or outdated .vtp file");
    }
    read(in,mc.xf.absDomain);
    read(in,mc.xf.relDomain);
    read(in,mc.xf.colorMap);
    read(in,mc.xf.opacityScale);
    if (modelConfigMagic >= modelConfigMagicBase + 4ull) {
      read(in,mc.iso.active);
      read(in,mc.iso.colors);
      read(in,mc.iso.values);
    }
    read(in,mc.camera);
    read(in,mc.lights.ambient);
    read(in,mc.lights.directional);
    PRINT(mc.lights.directional.size());
    if (mc.lights.directional.size() > 0)
      PRINT(mc.lights.directional[0].dir);
    
    return mc;
  }
  
}
