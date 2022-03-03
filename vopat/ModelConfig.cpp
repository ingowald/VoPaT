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

#include "vopat/ModelConfig.h"
#include "vopat/model/IO.h"

namespace vopat {

  const size_t modelConfigMagic = 0x12354522ull + 3;
  
  void ModelConfig::save(const std::string &fileName)
  {
    std::ofstream out(fileName);
    size_t magic = modelConfigMagic;
    write(out,magic);
    write(out,xf.absDomain);
    write(out,xf.relDomain);
    write(out,xf.colorMap);
    write(out,xf.opacityScale);
    write(out,camera);
    write(out,lights);
  }
  
  ModelConfig ModelConfig::load(const std::string &fileName)
  {
    std::ifstream in(fileName);
    size_t magic;
    ModelConfig mc;
    read(in,magic);
    if (magic != modelConfigMagic)
      throw std::runtime_error("invalid or outdated .vtp file");
    read(in,mc.xf.absDomain);
    read(in,mc.xf.relDomain);
    read(in,mc.xf.colorMap);
    read(in,mc.xf.opacityScale);
    read(in,mc.camera);
    PRINT(mc.xf.opacityScale);
    PRINT(mc.camera.up);
    read(in,mc.lights);
    return mc;
  }
  
}
