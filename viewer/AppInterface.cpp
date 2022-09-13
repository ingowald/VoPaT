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

#pragma once

#include "AppInterface.h"

namespace vopat {
  
  typedef enum
    {
     SET_CAMERA = 0 ,
     SET_LIGHTS,
     RESIZE_FRAME_BUFFER,
     RENDER_FRAME,
     SCREEN_SHOT,
     TERMINATE,
     SET_TRANSFER_FUNCTION,
     SET_ISO,
     RESET_ACCUMULATION
    } CommandTag;


  /*! mpi rendering interface for the *receivers* on the workers;
      these recive the broadcasts from the MPIMaster, and execute them
      on their (virutal) renderer. */
  struct Worker
  {
    Worker(CommBackend *comms);
    
    /*! the 'main loop' that receives and executes cmmands sent by the master */
    void run();

    /*! @{ command handlers - each corresponds to exactly one command
        sent my the master */
    void cmd_terminate();
    void cmd_renderFrame();
    void cmd_resizeFrameBuffer();
    void cmd_resetAccumulation();
    void cmd_setCamera();
    void cmd_setTransferFunction();
    void cmd_setISO();
    void cmd_setShadeMode();
    void cmd_setNodeSelection();
    void cmd_screenShot();
    void cmd_setLights();
    /* @} */

    template<typename T>
    void fromMaster(std::vector<T> &t);
    template<typename T>
    void fromMaster(T &t);
    
    Renderer *renderer = nullptr;
    CommBackend *comms;
  };


  AppInterface::AppInterface(MPIBackend &mpi,
                       MPIRenderer *renderer)
    : mpi(mpi),
      renderer(renderer)
  {}

  MPIWorker::MPIWorker(MPIBackend &mpi, MPIRenderer *renderer)
    : mpi(mpi),
      renderer(renderer)
  {}

  // ==================================================================
  
  void AppInterface::screenShot()
  {
    // ------------------------------------------------------------------
    // send request....
    // ------------------------------------------------------------------
    int cmd = SCREEN_SHOT;
    sendToWorkers(cmd);

    // ------------------------------------------------------------------
    // and do our own....
    // ------------------------------------------------------------------
    renderer->screenShot();
  }

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
    

  // ==================================================================
  
  void AppInterface::resetAccumulation()
  {
    // ------------------------------------------------------------------
    // send request....
    // ------------------------------------------------------------------
    int cmd = RESET_ACCUMULATION;
    sendToWorkers(cmd);
      
    // ------------------------------------------------------------------
    // and do our own....
    // ------------------------------------------------------------------
    renderer->resetAccumulation();
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
    

  // ==================================================================
  void AppInterface::terminate()
  {
    // ------------------------------------------------------------------
    // send request....
    // ------------------------------------------------------------------
    int cmd = TERMINATE;
    sendToWorkers(cmd);
      
    // ------------------------------------------------------------------
    // and do our own....
    // ------------------------------------------------------------------
    MPI_Finalize();
    exit(0);
  }
    
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



  // ==================================================================

  void AppInterface::renderFrame(uint32_t *fbPointer)
  {
    // ------------------------------------------------------------------
    // send request....
    // ------------------------------------------------------------------
    int cmd = RENDER_FRAME;
    sendToWorkers(cmd);
      
    // ------------------------------------------------------------------
    // and do our own....
    // ------------------------------------------------------------------
    renderer->render(fbPointer);
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

  // ==================================================================

  void AppInterface::resizeFrameBuffer(const vec2i &newSize)
  {
    // ------------------------------------------------------------------
    // send request....
    // ------------------------------------------------------------------
    int cmd = RESIZE_FRAME_BUFFER;
    sendToWorkers(cmd);
    sendToWorkers(newSize);
    
    // ------------------------------------------------------------------
    // and do our own....
    // ------------------------------------------------------------------
    renderer->resizeFrameBuffer(newSize);

    mpi.barrierAll();
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

  // ==================================================================

  void AppInterface::setCamera(const vec3f &from,
                            const vec3f &at,
                            const vec3f &up,
                            const float fovy)
  {
    // ------------------------------------------------------------------
    // send request....
    // ------------------------------------------------------------------
    int cmd = SET_CAMERA;
    sendToWorkers(cmd);
    sendToWorkers(from);
    sendToWorkers(at);
    sendToWorkers(up);
    sendToWorkers(fovy);
    // ------------------------------------------------------------------
    // and do our own....
    // ------------------------------------------------------------------
    renderer->setCamera(from,at,up,fovy);
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

    // ------------------------------------------------------------------
    // and execute
    // ------------------------------------------------------------------
    renderer->setCamera(from,at,up,fovy);
  }

  // ==================================================================

  void AppInterface::setTransferFunction(const std::vector<vec4f> &cm,
                                      const interval<float> &range,
                                      const float density)
  {
    // ------------------------------------------------------------------
    // send request....
    // ------------------------------------------------------------------
    int cmd = SET_TRANSFER_FUNCTION;
    sendToWorkers(cmd);
    sendToWorkers(cm);
    sendToWorkers(range);
    sendToWorkers(density);
    // ------------------------------------------------------------------
    // and do our own....
    // ------------------------------------------------------------------
    renderer->setTransferFunction(cm,range,density);
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
  
  // ==================================================================

  void AppInterface::setISO(int numActive,
                         const std::vector<int> &active,
                         const std::vector<float> &values,
                         const std::vector<vec3f> &colors)
  {
    // ------------------------------------------------------------------
    // send request....
    // ------------------------------------------------------------------
    int cmd = SET_ISO;
    sendToWorkers(cmd);
    sendToWorkers(active);
    sendToWorkers(values);
    sendToWorkers(colors);
    // ------------------------------------------------------------------
    // and do our own....
    // ------------------------------------------------------------------
    renderer->setISO(numActive,active,values,colors);
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

  // ==================================================================

  void AppInterface::setLights(float ambient,
                               const std::vector<vec3f> &dirs,
                               const std::vector<vec3f> &pows)
  {
    // ------------------------------------------------------------------
    // send request....
    // ------------------------------------------------------------------
    int cmd = SET_LIGHTS;
    sendToWorkers(cmd);
    sendToWorkers(dirs);
    sendToWorkers(pows);
    
    // ------------------------------------------------------------------
    // and do our own....
    // ------------------------------------------------------------------
    renderer->setLights(ambient,dirLights);
  }

  void MPIWorker::cmd_setLights()
  {
    // ------------------------------------------------------------------
    // get args....
    // ------------------------------------------------------------------
    float ambient;
    fromMaster(ambient);
    std::vector<vec3f> dirs,pows;
    fromMaster(dirs);
    fromMaster(pows);
    
    // ------------------------------------------------------------------
    // and execute
    // ------------------------------------------------------------------
    renderer->setLights(ambient,dirs,pows);
  }

  // ==================================================================

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
      default:
        throw std::runtime_error("unknown command ...");
      }
    }
  }
  
}
