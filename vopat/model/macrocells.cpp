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

int main(int argc, char **argv)
{
  try {
    std::string inFileBase = "";

    int numBricks = 0;
    vec3i inputSize = 0;
    std::string outFileBase;
    std::string inFileName;
    std::string inFormat;
    
    for (int i=1;i<argc;i++) {
      const std::string arg = argv[i];
      if (arg[0] != '-') {
        inFileBase = arg;
      } else {
        usage("unknown cmdline arg '"+arg+"'");
      }
    }

    Model::SP model = Model::load(Model::canonicalMasterFileName(inFileBase));
  } catch (std::exception &e) {
    std::cout << "Fatal runtime error: " << e.what() << std::endl;
    exit(1);
  } 
}
