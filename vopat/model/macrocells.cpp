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
#include "vopat/model/IO.h"

#include "owl/common/math/box.h"

#include <fstream>

namespace vopat {

  void usage(const std::string &error)
  {
    if (!error.empty())
      std::cerr << OWL_TERMINAL_RED
                << "Error: " << error
                << OWL_TERMINAL_DEFAULT
                << std::endl << std::endl;;
    std::cout << "./vopatGenMacrocells inFileBase <args>\n";
    std::cout << "\n";
    // std::cout << "-o <outpath>        : specifies common base part of output file names\n";
    // std::cout << "-n|--num-bricks     : num bricks to create\n";
    // std::cout << "-is|-ir|--input-res : resolution of input raw file\n";
    // std::cout << "-if|--input-format : format of input raw file\n";
    exit((error == "")?0:1);
  }
}

using namespace vopat;
using namespace owl;
using namespace owl::common;
typedef interval<float> range1f;

int main(int argc, char **argv)
{
  try {
    std::string inFileBase = "";

    int macrocellSize = 32;
    
    for (int i=1;i<argc;i++) {
      const std::string arg = argv[i];
      if (arg[0] != '-') {
        inFileBase = arg;
      } else if (arg == "-ms") {
        macrocellSize = std::stoi(argv[++i]);
      } else {
        usage("unknown cmdline arg '"+arg+"'");
      }
    }

    Model::SP model = Model::load(Model::canonicalMasterFileName(inFileBase));

    std::cout<<"Generating macrocells of size " << macrocellSize << "^3" <<std::endl;
    
    // For each brick, we will generate a collection of macrocells
    for (int bid = 0; bid < model->bricks.size(); ++bid) {
      Brick::SP brick = model->bricks[bid];
      std::cout<<"Processing brick " << bid << std::endl;
      std::cout<<"Number of voxels: " << brick->numVoxels.x << " " << brick->numVoxels.y << " " << brick->numVoxels.z << std::endl;

      vec3i numMacrocells = {
        (brick->numVoxels.x + (macrocellSize - 1)) / macrocellSize,
        (brick->numVoxels.y + (macrocellSize - 1)) / macrocellSize,
        (brick->numVoxels.z + (macrocellSize - 1)) / macrocellSize};
      std::cout<<"Number of macrocells to generate: " << numMacrocells.x << " " << numMacrocells.y << " " << numMacrocells.z << std::endl;
      std::vector<range1f> macrocells(numMacrocells.x * numMacrocells.y * numMacrocells.z);

      const std::string fileName = Model::canonicalRankFileName(inFileBase,bid);
      std::vector<float> scalars = brick->load(fileName);        
      for (int z = 0; z < brick->numVoxels.z; ++z) {
        for (int y = 0; y < brick->numVoxels.y; ++y) {
          for (int x = 0; x < brick->numVoxels.x; ++x) {
            int mcX = x / macrocellSize;
            int mcY = y / macrocellSize;
            int mcZ = z / macrocellSize;
            auto &mc = macrocells[
              mcX +
              mcY * numMacrocells.x +
              mcZ * numMacrocells.x * numMacrocells.y
            ];
            mc.extend(scalars[x + y * brick->numVoxels.x + z * brick->numVoxels.x * brick->numVoxels.y]);
          }
        }
      }
      std::cout<<"Done!" << std::endl;


      std::string macrocellFilename = fileName + ".mc";
      std::cout << OWL_TERMINAL_BLUE
              << "#writing macrocells of brick " << bid << " to " << macrocellFilename
              << OWL_TERMINAL_DEFAULT << std::endl;
      std::ofstream out(macrocellFilename);
      write(out, numMacrocells);
      write(out, macrocellSize);
      write(out, macrocells);
      std::cout << OWL_TERMINAL_GREEN
              << "#done writing macrocells to " << macrocellFilename
              << OWL_TERMINAL_DEFAULT << std::endl;
    }

  } catch (std::exception &e) {
    std::cout << "Fatal runtime error: " << e.what() << std::endl;
    exit(1);
  } 
}
