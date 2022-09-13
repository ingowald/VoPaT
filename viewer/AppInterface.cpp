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

  template<typename T>
  void AppInterface::fromMaster(T &t)
  {
    comm->worker.toMaster->bc_recv(&t,sizeof(T));
  }
  
  template<typename T>
  void AppInterface::fromMaster(std::vector<T> &t)
  {
    size_t s;
    fromMaster(s);
    t.resize(s);
    comm->worker.toMaster->bc_recv(t.data(),s*sizeof(T));
  }
  
  template<typename T>
  void AppInterface::sendToWorkers(const std::vector<T> &t)
  {
    size_t s = t.size();
    sendToWorkers(s);
    comm->master.toWorkers->broadcast(t.data(),s*sizeof(T));
  }
  
  template<typename T>
  void AppInterface::sendToWorkers(const T &t)
  {
    comm->master.toWorkers->broadcast(&t,sizeof(T));
  }
    
    

  AppInterface::AppInterface(CommBackend *comm,
                             VopatRenderer::SP renderer)
    : comm(comm),
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

  void AppInterface::cmd_screenShot()
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

  void AppInterface::cmd_resetAccumulation()
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
    // MPI_Finalize();
    comm->finalize();
    exit(0);
  }
    
  void AppInterface::cmd_terminate()
  {
    // ------------------------------------------------------------------
    // get args....
    // ------------------------------------------------------------------
    ;
    // ------------------------------------------------------------------
    // and do our own....
    // ------------------------------------------------------------------
    // MPI_Finalize();
    comm->finalize();
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
    renderer->renderFrame(fbPointer);
  }

  void AppInterface::cmd_renderFrame()
  {
    // ------------------------------------------------------------------
    // get args....
    // ------------------------------------------------------------------
    ;
    // ------------------------------------------------------------------
    // and execute
    // ------------------------------------------------------------------
    renderer->renderFrame(0);

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

    comm->barrierAll();
  }

  void AppInterface::cmd_resizeFrameBuffer()
  {
    // ------------------------------------------------------------------
    // get args....
    // ------------------------------------------------------------------
    vec2i newSize;
    fromMaster(newSize);
    // ------------------------------------------------------------------
    // and execute
    // ------------------------------------------------------------------
    renderer->resizeFrameBuffer(newSize);

    comm->barrierAll();
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

  void AppInterface::cmd_setCamera()
  {
    // ------------------------------------------------------------------
    // get args....
    // ------------------------------------------------------------------
    vec3f from, at, up;
    float fovy;
    fromMaster(from);
    fromMaster(at);
    fromMaster(up);
    fromMaster(fovy);

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

  void AppInterface::cmd_setTransferFunction()
  {
    // ------------------------------------------------------------------
    // get args....
    // ------------------------------------------------------------------
    int count;
    std::vector<vec4f> cm;
    interval<float> range;
    float density;
    fromMaster(cm);
    fromMaster(range);
    fromMaster(density);

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
    std::cout << "skipping iso for now; not yet implemented in renderer ... " << std::endl;
    // renderer->setISO(numActive,active,values,colors);
  }

  void AppInterface::cmd_setISO()
  {
    // ------------------------------------------------------------------
    // get args....
    // ------------------------------------------------------------------
    int count;
    int numActive;
    std::vector<int> active;
    std::vector<float> values;
    std::vector<vec3f> colors;
    fromMaster(active);
    fromMaster(values);
    fromMaster(colors);

    // ------------------------------------------------------------------
    // and execute
    // ------------------------------------------------------------------
    std::cout << "skipping iso for now; not yet implemented in renderer ... " << std::endl;
    // renderer->setISO(numActive,active,values,colors);
  }

  // ==================================================================

  void AppInterface::setLights(float ambient,
                               const std::vector<DirectionalLight> &dirLights)
  {
    // ------------------------------------------------------------------
    // send request....
    // ------------------------------------------------------------------
    int cmd = SET_LIGHTS;
    sendToWorkers(cmd);
    sendToWorkers(dirLights);
    
    // ------------------------------------------------------------------
    // and do our own....
    // ------------------------------------------------------------------
    renderer->setLights(ambient,dirLights);
  }

  void AppInterface::cmd_setLights()
  {
    // ------------------------------------------------------------------
    // get args....
    // ------------------------------------------------------------------
    float ambient;
    fromMaster(ambient);
    std::vector<DirectionalLight> dirLights;
    fromMaster(dirLights);
    
    // ------------------------------------------------------------------
    // and execute
    // ------------------------------------------------------------------
    renderer->setLights(ambient,dirLights);
  }

  // ==================================================================

  void AppInterface::runWorker()
  {
    while (1) {
      int cmd;
      fromMaster(cmd);
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
