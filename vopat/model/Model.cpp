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
#include <fstream>

namespace vopat {

  void Model::save(const std::string &fileName)
  {
    std::cout << OWL_TERMINAL_BLUE
              << "#writing model of " << bricks.size() << " bricks to " << fileName
              << OWL_TERMINAL_DEFAULT << std::endl;

    std::ofstream out(fileName);
    write(out,numVoxelsTotal);
    write(out,int(bricks.size()));
    for (int i=0;i<bricks.size();i++) {
      Brick::SP brick = bricks[i];
      write(out,brick->voxelRange);
      write(out,brick->cellRange);
      write(out,brick->spaceRange);
      write(out,brick->numVoxels);
      write(out,brick->numCells);
      write(out,brick->numVoxelsParent);
    }
    std::cout << OWL_TERMINAL_GREEN
              << "#done writing model to " << fileName
              << OWL_TERMINAL_DEFAULT << std::endl;
  }
  
  Model::SP Model::load(const std::string &fileName)
  {
    Model::SP model = std::make_shared<Model>();
    
    std::ifstream in(fileName);
    read(in,model->numVoxelsTotal);
    int numBricks = read<int>(in);
    for (int i=0;i<numBricks;i++) {
      Brick::SP brick = Brick::create();
      read(in,brick->voxelRange);
      read(in,brick->cellRange);
      read(in,brick->spaceRange);
      read(in,brick->numVoxels);
      read(in,brick->numCells);
      read(in,brick->numVoxelsParent);
      model->bricks.push_back(brick);
    }
    std::cout << OWL_TERMINAL_GREEN
              << "#done loading, found " << model->bricks.size()
              << " bricks..."
              << OWL_TERMINAL_DEFAULT << std::endl;
    return model;
  }

}
