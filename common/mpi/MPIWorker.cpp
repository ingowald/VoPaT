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

#include "common/mpi/MPIWorker.h"

namespace vopat {
  
  MPIWorker::MPIWorker(MPIBackend &mpi, MPIRenderer *renderer)
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
    renderer->render(0);

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

  void MPIWorker::cmd_setShadeMode()
  {
    // ------------------------------------------------------------------
    // get args....
    // ------------------------------------------------------------------
    int value;
    MPI_Bcast((void*)&value,sizeof(value),MPI_BYTE,0,MPI_COMM_WORLD);
    
    // ------------------------------------------------------------------
    // and execute
    // ------------------------------------------------------------------
    renderer->setShadeMode(value);
  }

  void MPIWorker::cmd_setLights()
  {
    // ------------------------------------------------------------------
    // get args....
    // ------------------------------------------------------------------
    float ambient;
    MPI_Bcast((void*)&ambient,sizeof(ambient),MPI_BYTE,0,MPI_COMM_WORLD);
    int count;
    MPI_Bcast((void*)&count,sizeof(count),MPI_BYTE,0,MPI_COMM_WORLD);
    std::vector<MPIRenderer::DirectionalLight> dirLights;
    dirLights.resize(count);
    for (int i=0;i<count;i++) {
      MPI_Bcast((void*)&dirLights[i],sizeof(dirLights[i]),MPI_BYTE,0,MPI_COMM_WORLD);
    }
    
    // ------------------------------------------------------------------
    // and execute
    // ------------------------------------------------------------------
    renderer->setLights(ambient,dirLights);
  }

  void MPIWorker::cmd_script()
  {
    // ------------------------------------------------------------------
    // get args....
    // ------------------------------------------------------------------
    int len;
    MPI_Bcast((void*)&len,sizeof(len),MPI_BYTE,0,MPI_COMM_WORLD);
    // std::string cmd(len);
    std::vector<char> cmd(len+1);
    MPI_Bcast((void *)cmd.data(),len,MPI_BYTE,0,MPI_COMM_WORLD);
    cmd[len] = 0;
    
    // ------------------------------------------------------------------
    // and execute
    // ------------------------------------------------------------------
    renderer->script(std::string(cmd.data()));
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
    vec3f from, at, up;
    float fovy;
    MPI_Bcast(&from,sizeof(from),MPI_BYTE,0,MPI_COMM_WORLD);
    MPI_Bcast(&at,sizeof(at),MPI_BYTE,0,MPI_COMM_WORLD);
    MPI_Bcast(&up,sizeof(up),MPI_BYTE,0,MPI_COMM_WORLD);
    MPI_Bcast(&fovy,sizeof(fovy),MPI_BYTE,0,MPI_COMM_WORLD);

    // Camera camera;
    // MPI_Bcast(&camera,sizeof(camera),MPI_BYTE,0,MPI_COMM_WORLD);
    // ------------------------------------------------------------------
    // and execute
    // ------------------------------------------------------------------
    renderer->setCamera(from,at,up,fovy);
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
    renderer->screenShot();
  }
    
  void MPIWorker::cmd_setTransferFunction()
  {
    // ------------------------------------------------------------------
    // get args....
    // ------------------------------------------------------------------
    int count;
    std::vector<vec4f> cm;
    interval<float> range;
    float density;
    MPI_Bcast((void*)&count,sizeof(count),MPI_BYTE,0,MPI_COMM_WORLD);
    cm.resize(count);
    MPI_Bcast((void*)cm.data(),count*sizeof(*cm.data()),MPI_BYTE,0,MPI_COMM_WORLD);
    MPI_Bcast((void*)&range,sizeof(range),MPI_BYTE,0,MPI_COMM_WORLD);
    MPI_Bcast((void*)&density,sizeof(density),MPI_BYTE,0,MPI_COMM_WORLD);

    // ------------------------------------------------------------------
    // and execute
    // ------------------------------------------------------------------
    renderer->setTransferFunction(cm,range,density);
  }
  
  void MPIWorker::cmd_setISO()
  {
    // ------------------------------------------------------------------
    // get args....
    // ------------------------------------------------------------------
    int count;
    int numActive;
    std::vector<int> active;
    std::vector<float> values;
    std::vector<vec3f> colors;
    MPI_Bcast((void*)&count,sizeof(count),MPI_BYTE,0,MPI_COMM_WORLD);
    active.resize(count);
    values.resize(count);
    colors.resize(count);
    MPI_Bcast((void*)&numActive,sizeof(numActive),MPI_BYTE,0,MPI_COMM_WORLD);
    MPI_Bcast((void*)active.data(),count*sizeof(*active.data()),MPI_BYTE,0,MPI_COMM_WORLD);
    MPI_Bcast((void*)values.data(),count*sizeof(*values.data()),MPI_BYTE,0,MPI_COMM_WORLD);
    MPI_Bcast((void*)colors.data(),count*sizeof(*colors.data()),MPI_BYTE,0,MPI_COMM_WORLD);

    // ------------------------------------------------------------------
    // and execute
    // ------------------------------------------------------------------
    renderer->setISO(numActive,active,values,colors);
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
      case SET_TRANSFER_FUNCTION:
        cmd_setTransferFunction();
        break;
      case SET_ISO:
        cmd_setISO();
        break;
      case SET_LIGHTS:
        cmd_setLights();
        break;
      case SET_SHADE_MODE:
        cmd_setShadeMode();
        break;
      case CALL_SCRIPT:
        cmd_script();
        break;
      default:
        throw std::runtime_error("unknown command ...");
      }
    }
  }

} // ::vopat
