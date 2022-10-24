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

#include "model/StructuredModel.h"
#include "model/UMeshModel.h"
#include "model/IO.h"
#include <fstream>

namespace vopat {

  size_t file_format_version = /*increment this for every change:*/4;
  size_t file_format_magic = 0x33441232340888ull + file_format_version;
  
  /*! given a base file name prefix (including directory name, if
    desired), return a canonical file name for the master model
    file */
  std::string Model::canonicalMasterFileName(const std::string &baseName)
  {
    return baseName+".vopat";
  }
  
  std::string Model::canonicalBrickFileName(const std::string &baseName,
                                            int brickID)
  {
    char bid[100];
    sprintf(bid,"%05i",brickID);
    return baseName+".b"+bid+".unvar.brick";
  }

  /*! given a base file name prefix (including directory name, if
    desired), return a canonical file name for the data file for
    the 'rankID'th rank */
  std::string Model::canonicalTimeStepFileName(const std::string &baseName,
                                               int rankID,
                                               const std::string &variable,
                                               int timeStep)
  {
    char ts[100];
    sprintf(ts,"%05i",timeStep);
    char bid[100];
    sprintf(bid,"%05i",rankID);
    return baseName+"__"+variable+"__t"+ts+".b"+bid+".brick";
    // #endif
  }

  void Model::save(const std::string &fileName)
  {
    std::cout << OWL_TERMINAL_BLUE
              << "#writing model of " << numBricks << " bricks to " << fileName
              << OWL_TERMINAL_DEFAULT << std::endl;

    std::ofstream out(fileName,std::ios::binary);
    size_t fileMagic = file_format_magic;
    write(out,fileMagic);

    write(out,type);

    write(out,numBricks);
    write(out,numTimeSteps);
    write(out,domain);
    write(out,valueRange);
    std::cout << OWL_TERMINAL_GREEN
              << "#done writing model to " << fileName
              << OWL_TERMINAL_DEFAULT << std::endl;
  }
  
  Model::SP Model::load(const std::string &fileName)
  {
    std::ifstream in(fileName,std::ios::binary);
    if (!in.good())
      throw std::runtime_error("could not open '"+fileName+"'");

    size_t fileMagic;
    read(in,fileMagic);
    if (fileMagic != file_format_magic)
      throw std::runtime_error("invalid model file, or wrong brick/model file format version; please rebuild your model");
    std::string type = read<std::string>(in);

    Model::SP model;
    if (type == "UMeshModel/Spatial")
      model = UMeshModel::create();
    else  if (type == "StructuredModel<float>")
      model = StructuredModel::create();
    else
      throw std::runtime_error("unknown/unsupported model type '"+type+"'");
    
    read(in,model->numBricks);
    read(in,model->numTimeSteps);
    read(in,model->domain);
    read(in,model->valueRange);
    std::cout << OWL_TERMINAL_GREEN
              << "#done loading model meta info, expecting " << model->numBricks
              << " bricks..."
              << OWL_TERMINAL_DEFAULT << std::endl;
    return model;
  }

}
