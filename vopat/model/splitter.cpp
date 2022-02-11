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

namespace vopat {

  Brick::SP kdCreateBrick(int rank, int numRanks, const vec3i &inputSize)
  {
    box3i region{vec3i(0),inputSize-vec3i(1)};
    while (1) {
      if (numRanks == 1) return Brick::create(inputSize,region);
      
      int lCount = numRanks / 2;
      int rCount = numRanks - lCount;
      
      float relSplit = lCount / float(numRanks);
      
      int dim = arg_max(region.size());
      box3i lRegion = region; 
      box3i rRegion = region;
      lRegion.upper[dim] = rRegion.lower[dim] =
        int(region.lower[dim] + relSplit * region.size()[dim]);
      if (rank >= lCount) {
        region   =  rRegion;
        rank     -= lCount;
        numRanks -= lCount;
      } else {
        region   =  lRegion;
        rank     -= 0;
        numRanks -= rCount;
      }
    }
  }


  void usage(const std::string &error)
  {
    std::cout << "./umeshSplitRaw inFileName.raw <args>\n";
    std::cout << "\n";
    std::cout << "-o <outpath>        : specifies common base part of output file names\n";
    std::cout << "-n|--num-bricks     : num bricks to create\n";
    std::cout << "-is|-ir|--input-res : resolution of input raw file\n";
    exit((error == "")?0:1);
  }
}

using namespace vopat;

int main(int ac, char **av)
{
  int numBricks = 0;
  vec3i inputSize = 0;
  std::string outFileBase;
  std::string inFileName;
  
  for (int i=1;i<ac;i++) {
    const std::string arg = av[i];
    if (arg[0] != '-') {
      inFileName = arg;
    } else if (arg == "-is" || arg == "-ir" || arg == "--input-size" || arg == "--input-res") {
      inputSize.x = std::stoi(av[++i]);
      inputSize.y = std::stoi(av[++i]);
      inputSize.z = std::stoi(av[++i]);
    } else if (arg == "-o") {
      outFileBase = av[++i];
    } else if (arg == "-n" || arg == "-nr" || arg == "--num-bricks" || arg == "-nb" || arg == "--num-ranks") {
      numBricks = std::stoi(av[++i]);
    } else {
      usage("unknown cmdline arg '"+arg+"'");
    }
  }
  if (numBricks < 1) usage("invalid or not-specified number of bricks");
  if (reduce_min(inputSize) <= 0) usage("invalid or not-specified input model size");
  if (inFileName.empty()) usage("invalid or not-specified input file name");

  Model::SP model = Model::create();
  for (int brickID=0;brickID<numBricks;brickID++) {
    model->bricks.push_back(kdCreateBrick(brickID,numBricks,inputSize));
    std::cout << "... created brick " << model->bricks.back()->toString() << std::endl;
  }
  model->save(outFileBase+".vopat");
}
