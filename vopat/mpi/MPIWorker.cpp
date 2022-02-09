// ======================================================================== //
// Copyright 2022-2022 Ingo Wald                                            //
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

#include "vopat/mpi/MPIWorker.h"
// #define STB_IMAGE_WRITE_IMPLEMENTATION 1
#include "3rdParty/stb_image/stb/stb_image_write.h"
#define STB_IMAGE_IMPLEMENTATION 1
#include "3rdParty/stb_image//stb/stb_image.h"

namespace vopat {
  
  MPIWorker::MPIWorker(MPIBackend &mpi, Renderer *renderer)
    : mpi(mpi),
      renderer(renderer)
  {}

  void MPIWorker::cmd_terminate()
  {
    // ------------------------------------------------------------------
    // get args....
    // ------------------------------------------------------------------
    ;
    // ------------------------------------------------------------------
    // and do our own....
    // ------------------------------------------------------------------
    MPI_Finalize();
    exit(0);
  }


  void MPIWorker::cmd_renderFrame()
  {
    // ------------------------------------------------------------------
    // get args....
    // ------------------------------------------------------------------
    ;
    // ------------------------------------------------------------------
    // and execute
    // ------------------------------------------------------------------
    renderer->render();

    // throw std::runtime_error("HARD EXIT FOR DEBUG");
  }
  void MPIWorker::cmd_resizeFrameBuffer()
  {
    // ------------------------------------------------------------------
    // get args....
    // ------------------------------------------------------------------
    vec2i newSize;
    MPI_Bcast((void*)&newSize,sizeof(newSize),MPI_BYTE,0,MPI_COMM_WORLD);
    // ------------------------------------------------------------------
    // and execute
    // ------------------------------------------------------------------
    renderer->resizeFrameBuffer(newSize);

    mpi.barrierAll();
  }

  void MPIWorker::cmd_resetAccumulation()
  {
    // ------------------------------------------------------------------
    // get args....
    // ------------------------------------------------------------------
    ;
    // ------------------------------------------------------------------
    // and execute
    // ------------------------------------------------------------------
    renderer->resetAccumulation();
  }
    
  void MPIWorker::cmd_setCamera()
  {
    // ------------------------------------------------------------------
    // get args....
    // ------------------------------------------------------------------
    Camera camera;
    MPI_Bcast(&camera,sizeof(camera),MPI_BYTE,0,MPI_COMM_WORLD);
    // ------------------------------------------------------------------
    // and execute
    // ------------------------------------------------------------------
    renderer->setCamera(camera);
  }

  // void MPIWorker::cmd_setShadeMode()
  // {
  //   // ------------------------------------------------------------------
  //   // get args....
  //   // ------------------------------------------------------------------
  //   int shadeMode;
  //   MPI_Bcast(&shadeMode,sizeof(shadeMode),MPI_BYTE,0,MPI_COMM_WORLD);
  //   // ------------------------------------------------------------------
  //   // and execute
  //   // ------------------------------------------------------------------
  //   renderer->setShadeMode(shadeMode);
  // }

  // void MPIWorker::cmd_setNodeSelection()
  // {
  //   // ------------------------------------------------------------------
  //   // get args....
  //   // ------------------------------------------------------------------
  //   int nodeSelection;
  //   MPI_Bcast(&nodeSelection,sizeof(nodeSelection),MPI_BYTE,0,MPI_COMM_WORLD);
  //   // ------------------------------------------------------------------
  //   // and execute
  //   // ------------------------------------------------------------------
  //   renderer->setNodeSelection(nodeSelection);
  // }

  void MPIWorker::cmd_screenShot()
  {
    // ------------------------------------------------------------------
    // get args....
    // ------------------------------------------------------------------
    ;
    // ------------------------------------------------------------------
    // and execute
    // ------------------------------------------------------------------
    const vec2i fbSize = renderer->fbSize;
    vec3f *fb = renderer->getLocalAccumBuffer();
      
    char fileName[10000];
    sprintf(fileName,"%s-rank%02i.png",
            Renderer::screenShotFileName.c_str(),
            renderer->myRank());
      
    std::vector<uint32_t> pixels;
    for (int y=0;y<fbSize.y;y++) {
      const vec3f *line = fb + (fbSize.y-1-y)*fbSize.x;
      for (int x=0;x<fbSize.x;x++) {
        vec3f col = sqrt(line[x] / renderer->accumID);
        int r = int(min(255.f,255.f*col.x));
        int g = int(min(255.f,255.f*col.y));
        int b = int(min(255.f,255.f*col.z));
          
        pixels.push_back((r << 0) |
                         (g << 8) |
                         (b << 16) |
                         (0xff << 24));
      }
    }
    stbi_write_png(fileName,fbSize.x,fbSize.y,4,
                   pixels.data(),fbSize.x*sizeof(uint32_t));
    std::cout << "screenshot saved in '" << fileName << "'" << std::endl;
  }
    
  void MPIWorker::run()
  {
    while (1) {
      int cmd;
      MPI_Bcast(&cmd,1,MPI_INT,0,MPI_COMM_WORLD);
      switch(cmd) {
      case SET_CAMERA:
        cmd_setCamera();
        break;
      // case SET_SHADE_MODE:
      //   cmd_setShadeMode();
      //   break;
      // case SET_NODE_SELECTION:
      //   cmd_setNodeSelection();
      //   break;
      case TERMINATE:
        cmd_terminate();
        break;
      case RESIZE_FRAME_BUFFER:
        cmd_resizeFrameBuffer();
        break;
      case RENDER_FRAME:
        cmd_renderFrame();
        break;
      case RESET_ACCUMULATION:
        cmd_resetAccumulation();
        break;
      case SCREEN_SHOT:
        cmd_screenShot();
        break;
      default:
        throw std::runtime_error("unknown command ...");
      }
    }
  }

} // ::vopat
