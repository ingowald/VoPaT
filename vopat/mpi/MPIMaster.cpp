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

#include "vopat/mpi/MPIMaster.h"
#include "vopat/render/Renderer.h"
// #define STB_IMAGE_WRITE_IMPLEMENTATION 1
#include "3rdParty/stb_image//stb/stb_image_write.h"
#include "3rdParty/stb_image//stb/stb_image.h"

namespace vopat {
  
  MPIMaster::MPIMaster(MPIBackend &mpi, Renderer   *renderer)
    : mpi(mpi), renderer(renderer)
  {}

  void MPIMaster::screenShot()
  {
    PING;
    // ------------------------------------------------------------------
    // send request....
    // ------------------------------------------------------------------
    int cmd = SCREEN_SHOT;
    MPI_Bcast(&cmd,1,MPI_INT,0,MPI_COMM_WORLD);

#if 1
    renderer->screenShot();
#else
    // ------------------------------------------------------------------
    // and do our own....
    // ------------------------------------------------------------------
    const uint32_t *fb
      = this->getFB();
      
    char fileName[10000];
    sprintf(fileName,"%s-master.png",
            Renderer::screenShotFileName.c_str());
    PRINT(fileName);
      
    std::vector<uint32_t> pixels;
    for (int y=0;y<fbSize.y;y++) {
      const uint32_t *line = fb + (fbSize.y-1-y)*fbSize.x;
      for (int x=0;x<fbSize.x;x++) {
        pixels.push_back(line[x] | (0xff << 24));
      }
    }
    stbi_write_png(fileName,fbSize.x,fbSize.y,4,
                   pixels.data(),fbSize.x*sizeof(uint32_t));
    std::cout << "screenshot saved in '" << fileName << "'" << std::endl;
#endif
  }
    
  void MPIMaster::resetAccumulation()
  {
    PING;
    // ------------------------------------------------------------------
    // send request....
    // ------------------------------------------------------------------
    int cmd = RESET_ACCUMULATION;
    MPI_Bcast(&cmd,1,MPI_INT,0,MPI_COMM_WORLD);
      
    // ------------------------------------------------------------------
    // and do our own....
    // ------------------------------------------------------------------
    renderer->resetAccumulation();
  }

  void MPIMaster::terminate()
  {
    PING;
    // ------------------------------------------------------------------
    // send request....
    // ------------------------------------------------------------------
    int cmd = TERMINATE;
    MPI_Bcast(&cmd,1,MPI_INT,0,MPI_COMM_WORLD);
      
    // ------------------------------------------------------------------
    // and do our own....
    // ------------------------------------------------------------------
    MPI_Finalize();
    exit(0);
  }
    
  void MPIMaster::renderFrame(uint32_t *fbPointer)
  {
    PING;
    // ------------------------------------------------------------------
    // send request....
    // ------------------------------------------------------------------
    int cmd = RENDER_FRAME;
    MPI_Bcast(&cmd,1,MPI_INT,0,MPI_COMM_WORLD);
      
    // ------------------------------------------------------------------
    // and do our own....
    // ------------------------------------------------------------------
    renderer->render(fbPointer);//Frame(fbPointer);
    // this->collectRankResults(fbPointer);
  }

  void MPIMaster::resizeFrameBuffer(const vec2i &newSize)
  {
    PING;
    // ------------------------------------------------------------------
    // send request....
    // ------------------------------------------------------------------
    int cmd = RESIZE_FRAME_BUFFER;
    MPI_Bcast(&cmd,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast((void*)&newSize,sizeof(newSize),MPI_BYTE,0,MPI_COMM_WORLD);
    // ------------------------------------------------------------------
    // and do our own....
    // ------------------------------------------------------------------
    // fbSize = newSize;
    // this->resize(fbSize);
    renderer->resizeFrameBuffer(newSize);

    mpi.barrierAll();
  }

  void MPIMaster::setCamera(const Camera &camera)
  {
    PING;
    // ------------------------------------------------------------------
    // send request....
    // ------------------------------------------------------------------
    int cmd = SET_CAMERA;
    MPI_Bcast(&cmd,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast((void*)&camera,sizeof(camera),MPI_BYTE,0,MPI_COMM_WORLD);
    // ------------------------------------------------------------------
    // and do our own....
    // ------------------------------------------------------------------
  }

  // void MPIMaster::setShadeMode(int shadeMode)
  // {
  //   // ------------------------------------------------------------------
  //   // send request....
  //   // ------------------------------------------------------------------
  //   int cmd = SET_SHADE_MODE;
  //   MPI_Bcast(&cmd,1,MPI_INT,0,MPI_COMM_WORLD);
  //   MPI_Bcast((void*)&shadeMode,sizeof(shadeMode),MPI_BYTE,0,MPI_COMM_WORLD);
  //   // ------------------------------------------------------------------
  //   // and do our own....
  //   // ------------------------------------------------------------------
  // }

  // void MPIMaster::setNodeSelection(int nodeSelection)
  // {
  //   // ------------------------------------------------------------------
  //   // send request....
  //   // ------------------------------------------------------------------
  //   int cmd = SET_NODE_SELECTION;
  //   MPI_Bcast(&cmd,1,MPI_INT,0,MPI_COMM_WORLD);
  //   MPI_Bcast((void*)&nodeSelection,sizeof(nodeSelection),MPI_BYTE,0,MPI_COMM_WORLD);
      
  //   // ------------------------------------------------------------------
  //   // and do our own....
  //   // ------------------------------------------------------------------
  //   // optix->setNodeSelection(nodeSelection);
  // }

//   void MPIMaster::collectRankResults(uint32_t *fbPointer)
//   {
//     PING;
// // #if USE_APP_FB
//     this->appFB = fbPointer;
// // #endif
// //     mpi.master.toWorkers->indexedGather
// //       ((uint32_t*)getFB(),
// //        fbSize.x*sizeof(uint32_t),
// //        fbSize.y);
// // #if USE_APP_FB
// // #else
// //     memcpy(fbPointer,fullyAssembledFrame.data(),
// //            fullyAssembledFrame.size()*sizeof(fullyAssembledFrame[0]));
// // #endif
//   }
  
//   void MPIMaster::resize(const vec2i &newSize)
//   {
//     fbSize = newSize;
// // #if USE_APP_FB
// // #else
// //     cudaDeviceSynchronize();
// //     fullyAssembledFrame.resize(area(fbSize));
// //     cudaDeviceSynchronize();
// // #endif
//   }


} // ::vopat
