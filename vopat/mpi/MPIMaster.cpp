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

namespace vopat {
  
  MPIMaster::MPIMaster(MPIBackend &mpi,
                       MPIRenderer *renderer)
    : mpi(mpi),
      renderer(renderer)
  {}

  void MPIMaster::screenShot()
  {
    // ------------------------------------------------------------------
    // send request....
    // ------------------------------------------------------------------
    int cmd = SCREEN_SHOT;
    MPI_Bcast(&cmd,1,MPI_INT,0,MPI_COMM_WORLD);

    // ------------------------------------------------------------------
    // and do our own....
    // ------------------------------------------------------------------
    renderer->screenShot();
  }
    
  void MPIMaster::resetAccumulation()
  {
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
    // ------------------------------------------------------------------
    // send request....
    // ------------------------------------------------------------------
    int cmd = RENDER_FRAME;
    MPI_Bcast(&cmd,1,MPI_INT,0,MPI_COMM_WORLD);
      
    // ------------------------------------------------------------------
    // and do our own....
    // ------------------------------------------------------------------
    renderer->render(fbPointer);
  }

  void MPIMaster::resizeFrameBuffer(const vec2i &newSize)
  {
    // ------------------------------------------------------------------
    // send request....
    // ------------------------------------------------------------------
    int cmd = RESIZE_FRAME_BUFFER;
    MPI_Bcast(&cmd,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast((void*)&newSize,sizeof(newSize),MPI_BYTE,0,MPI_COMM_WORLD);
    // ------------------------------------------------------------------
    // and do our own....
    // ------------------------------------------------------------------
    renderer->resizeFrameBuffer(newSize);

    mpi.barrierAll();
  }

  void MPIMaster::setCamera(const vec3f &from,
                            const vec3f &at,
                            const vec3f &up,
                            const float fovy)
  {
    // ------------------------------------------------------------------
    // send request....
    // ------------------------------------------------------------------
    int cmd = SET_CAMERA;
    MPI_Bcast(&cmd,1,MPI_INT,0,MPI_COMM_WORLD);
    // MPI_Bcast((void*)&camera,sizeof(camera),MPI_BYTE,0,MPI_COMM_WORLD);
    MPI_Bcast((void*)&from,sizeof(from),MPI_BYTE,0,MPI_COMM_WORLD);
    MPI_Bcast((void*)&at,sizeof(at),MPI_BYTE,0,MPI_COMM_WORLD);
    MPI_Bcast((void*)&up,sizeof(up),MPI_BYTE,0,MPI_COMM_WORLD);
    MPI_Bcast((void*)&fovy,sizeof(fovy),MPI_BYTE,0,MPI_COMM_WORLD);
    // ------------------------------------------------------------------
    // and do our own....
    // ------------------------------------------------------------------
    renderer->setCamera(from,at,up,fovy);
  }

  void MPIMaster::setTransferFunction(const std::vector<vec4f> &cm,
                                      const interval<float> &range,
                                      const float density)
  {
    // ------------------------------------------------------------------
    // send request....
    // ------------------------------------------------------------------
    int cmd = SET_TRANSFER_FUNCTION;
    MPI_Bcast(&cmd,1,MPI_INT,0,MPI_COMM_WORLD);
    // MPI_Bcast((void*)&camera,sizeof(camera),MPI_BYTE,0,MPI_COMM_WORLD);
    int count = (int)cm.size();
    MPI_Bcast((void*)&count,sizeof(count),MPI_BYTE,0,MPI_COMM_WORLD);
    MPI_Bcast((void*)cm.data(),count*sizeof(*cm.data()),MPI_BYTE,0,MPI_COMM_WORLD);
    MPI_Bcast((void*)&range,sizeof(range),MPI_BYTE,0,MPI_COMM_WORLD);
    MPI_Bcast((void*)&density,sizeof(density),MPI_BYTE,0,MPI_COMM_WORLD);
    // ------------------------------------------------------------------
    // and do our own....
    // ------------------------------------------------------------------
    renderer->setTransferFunction(cm,range,density);
  }

  void MPIMaster::setLights(float ambient,
                            const std::vector<MPIRenderer::DirectionalLight> &dirLights)
  {
    // ------------------------------------------------------------------
    // send request....
    // ------------------------------------------------------------------
    int cmd = SET_LIGHTS;
    MPI_Bcast(&cmd,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast((void*)&ambient,sizeof(ambient),MPI_BYTE,0,MPI_COMM_WORLD);
    int count = dirLights.size();
    MPI_Bcast((void*)&count,sizeof(count),MPI_BYTE,0,MPI_COMM_WORLD);
    for (int i=0;i<count;i++) {
      MPI_Bcast((void*)&dirLights[i],sizeof(dirLights[i]),MPI_BYTE,0,MPI_COMM_WORLD);
    }
    
    // ------------------------------------------------------------------
    // and do our own....
    // ------------------------------------------------------------------
    renderer->setLights(ambient,dirLights);
  }

  void MPIMaster::setShadeMode(int value)
  {
    // ------------------------------------------------------------------
    // send request....
    // ------------------------------------------------------------------
    int cmd = SET_SHADE_MODE;
    MPI_Bcast(&cmd,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast((void*)&value,sizeof(value),MPI_BYTE,0,MPI_COMM_WORLD);
    // ------------------------------------------------------------------
    // and do our own....
    // ------------------------------------------------------------------
    renderer->setShadeMode(value);
  }

  
} // ::vopat
