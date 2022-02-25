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

namespace vopat {

  Brick::SP kdCreateBrick(int rank, int numRanks, const vec3i &inputSize)
  {
    const int inputRank = rank;
    box3i region{vec3i(0),inputSize-vec3i(1)};
    while (1) {
      if (numRanks == 1)
        return Brick::create(inputRank,inputSize,region);
      
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
    if (!error.empty())
      std::cerr << OWL_TERMINAL_RED
                << "Error: " << error
                << OWL_TERMINAL_DEFAULT
                << std::endl << std::endl;;
    std::cout << "./umeshSplitRaw inFileName.raw <args>\n";
    std::cout << "\n";
    std::cout << "-o <outpath>        : specifies common base part of output file names\n";
    std::cout << "-n|--num-bricks     : num bricks to create\n";
    std::cout << "-is|-ir|--input-res : resolution of input raw file\n";
    std::cout << "-if|--input-format : format of input raw file\n";
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
  std::string inFormat;
  
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
    } else if (arg == "-if" || arg == "--input-format") {
      inFormat = av[++i];
    } else if (arg == "-n" || arg == "-nr" || arg == "--num-bricks" || arg == "-nb" || arg == "--num-ranks") {
      numBricks = std::stoi(av[++i]);
    } else {
      usage("unknown cmdline arg '"+arg+"'");
    }
  }
  if (numBricks < 1) usage("invalid or not-specified number of bricks");
  if (reduce_min(inputSize) <= 0) usage("invalid or not-specified input model size");
  if (inFileName.empty()) usage("invalid or not-specified input file name");
  if (inFormat == "") usage("input format not specified");
  if (inFormat != "uint8" && inFormat != "uint16" && inFormat != "float") usage("unknown input format (allowed 'uint8' 'uint16' 'float')");
  Model::SP model = Model::create();
  model->numVoxelsTotal = inputSize;
  for (int brickID=0;brickID<numBricks;brickID++) {
    model->bricks.push_back(kdCreateBrick(brickID,numBricks,inputSize));
    std::cout << "... created brick " << model->bricks.back()->toString() << std::endl;
  }
  std::cout << "saving meta..." << std::endl;
  model->save(Model::canonicalMasterFileName(outFileBase));

  int timeStep = 0;
  std::string variable = "unknown";
  for (auto brick : model->bricks) {
    std::vector<float> scalars;
    std::cout << "extracting var '" << variable << "', time step " << timeStep << ", brick " << brick->ID << std::endl;
    if (inFormat == "float")
      scalars = brick->loadRegionRAW<float>(inFileName);
    else if (inFormat == "uint8")
      scalars = brick->loadRegionRAW<uint8_t>(inFileName);
    else if (inFormat == "uint16")
      scalars = brick->loadRegionRAW<uint16_t>(inFileName);
    else
      throw std::runtime_error("unsupported raw format");
    for (auto v : scalars)
      model->valueRange.extend(v);
    std::string outFileName = Model::canonicalRankFileName(outFileBase,brick->ID);
    std::ofstream out(outFileName,std::ios::binary);
    // write(out,scalars);
    out.write((const char *)scalars.data(),scalars.size()*sizeof(scalars[0]));
    std::cout << OWL_TERMINAL_GREEN
              << " -> " << outFileName
              << OWL_TERMINAL_DEFAULT << std::endl;
  }
  
}
