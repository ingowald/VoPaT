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

#include "vopat/model/Model.h"
#include <fstream>

using namespace vopat;

int main(int ac, char **av)
{
  std::vector<std::string> inFileNames;
  std::string outFileName;
  for (int i=1;i<ac;i++) {
    const std::string arg = av[i];
    if (arg == "-o")
      outFileName = av[++i];
    else
      inFileNames.push_back(arg);
  }
  if (inFileNames.size() != 3 || outFileName.empty())
    throw std::runtime_error("usage: ./vopatMakeMagField u.raw v.raw w.raw -o mag.raw");

  std::ifstream in_x(inFileNames[0],std::ios::binary);
  if (!in_x) throw std::runtime_error("could not open "+inFileNames[0]);
  std::ifstream in_y(inFileNames[1],std::ios::binary);
  if (!in_y) throw std::runtime_error("could not open "+inFileNames[1]);
  std::ifstream in_z(inFileNames[2],std::ios::binary);
  if (!in_z) throw std::runtime_error("could not open "+inFileNames[2]);

  std::ofstream out(outFileName,std::ios::binary);
  size_t num = 0;
  while (!in_x.eof()) {
    vec3f v;
    in_x.read((char*)&v.x,sizeof(v.x));
    if (!in_x) break;
    in_y.read((char*)&v.y,sizeof(v.y));
    in_z.read((char*)&v.z,sizeof(v.z));

    float vm = length(v);
    out.write((const char *)&vm,sizeof(vm));
    ++num;
  }
  std::cout << "done... read and written 0x" << (int*)num << " scalars" << std::endl;
  return 0;
}
