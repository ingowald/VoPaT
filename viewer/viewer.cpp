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

#include "common/mpi/MPIMaster.h"
#include "common/mpi/MPIWorker.h"
#include "vopat/Renderer.h"
#include "model/ModelConfig.h"
#include <math.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

#include "samples/common/owlViewer/OWLViewer.h"

namespace vopat {

  std::string rendererName = "wc";
    
  struct {
    int spp = 1; //4;
    struct {
      // auto-generate
      vec3f vp = vec3f(0.f);
      vec3f vu = vec3f(0.f);
      vec3f vi = vec3f(0.f);
      float fovy = 70;
    } camera;
    vec2i windowSize  = vec2i(1024,1024);
    float windowScale = 1.f;
    std::string configFileName = "";
  } cmdline;
  
  void usage(const std::string &msg)
  {
    if (msg != "") std::cerr << "Error: " << msg << std::endl << std::endl;
    std::cout << "Usage: ./brxViewer <inputfile.pbf> -p <partSpec.ps>" << std::endl;
    exit(msg != "");
  }

  struct VoPaTViewer : public owl::viewer::OWLViewer
  {
    typedef OWLViewer inherited;

    MPIMaster &master;
    
    VoPaTViewer(MPIMaster &master)
      : master(master)
    {}
    
    void screenShot()
    {
      master.screenShot();
    }
    
    // /*! this function gets called whenever the viewer widget changes camera settings */
    virtual void cameraChanged() override 
    {
      inherited::cameraChanged();
      auto &camera = inherited::getCamera();

      const vec3f from = camera.position;
      const vec3f at = camera.getPOI();
      const vec3f up = camera.upVector;
      const float fovy = camera.fovyInDegrees;
     // master.setCamera(Camera(getWindowSize(),
      //                         from,at,up,fovy));
      master.setCamera(from,at,up,fovy);
      master.resetAccumulation();
      // glutPostRedisplay();
    }
    
    /*! window notifies us that we got resized */
    virtual void resize(const vec2i &newSize) override
    {
      this->fbSize = newSize;
      cudaDeviceSynchronize();
      master.resizeFrameBuffer(newSize);
      // optix->resizeFrameBuffer(newSize);
      
      // ... tell parent to resize (also resizes the pbo in the window)
      inherited::resize(newSize);
      
      // ... and finally: update the camera's aspect
      setAspect(newSize.x/float(newSize.y));

      cameraChanged();
      // update camera as well, since resize changed both aspect and
      // u/v pixel delta vectors ...
      updateCamera();
      cudaDeviceSynchronize();
    }
    
    /*! gets called whenever the viewer needs us to re-render out widget */
    virtual void render() override
    {
      if (fbSize.x < 0) return;
      
      static double t_last = -1;

      {
        // static double t0 = getCurrentTime();
        master.renderFrame(fbPointer);
      }      
      
      double t_now = getCurrentTime();
      static double avg_t = 0.;
      if (t_last >= 0)
        avg_t = 0.8*avg_t + 0.2*(t_now-t_last);

      if (displayFPS) {
        char title[1000];
        sprintf(title,"owlVoPaT - %.2f FPS",(1.f/avg_t));
        glfwSetWindowTitle(this->handle,title);
      }
      t_last = t_now;
    }
    
    /*! this gets called when the user presses a key on the keyboard ... */
    virtual void key(char key, const vec2i &where)
    {
      switch (key) {
      case '!':
        screenShot();
        break;
      case 'V':
        displayFPS = !displayFPS;
        break;
      case 'C': {
        auto &fc = getCamera();
        std::cout << "(C)urrent camera:" << std::endl;
        std::cout << "- from :" << fc.position << std::endl;
        std::cout << "- poi  :" << fc.getPOI() << std::endl;
        std::cout << "- upVec:" << fc.upVector << std::endl; 
        std::cout << "- frame:" << fc.frame << std::endl;
        std::cout.precision(10);
        std::cout << "cmdline: --camera "
                  << fc.position.x << " "
                  << fc.position.y << " "
                  << fc.position.z << " "
                  << fc.getPOI().x << " "
                  << fc.getPOI().y << " "
                  << fc.getPOI().z << " "
                  << fc.upVector.x << " "
                  << fc.upVector.y << " "
                  << fc.upVector.z << " "
                  << "-fovy " << fc.fovyInDegrees
                  << std::endl;
      } break;
      default:
        inherited::key(key,where);
      }
    }

    vec2i fbSize { -1,-1 };
    bool  displayFPS = true;
  };

  extern "C" int main(int argc, char **argv)
  {
    try {
      std::string inFileBase = "";
      int islandSize = -1;
      for (int i=1;i<argc;i++) {
        const std::string arg = argv[i];
        if (arg[0] != '-') {
          inFileBase = arg;
        } 
        else if (arg == "--renderer" || arg == "-r") {
          rendererName = argv[++i];
        }
        else if (arg == "-fovy") {
          cmdline.camera.fovy = std::atof(argv[++i]);
        }
        else if (arg == "--island-size" || arg == "-is") {
          islandSize = std::atoi(argv[++i]);
        }
        else if (arg == "-c" || "--config") {
          cmdline.configFileName = argv[++i];
        }
        else if (arg == "--camera") {
          cmdline.camera.vp.x = std::atof(argv[++i]);
          cmdline.camera.vp.y = std::atof(argv[++i]);
          cmdline.camera.vp.z = std::atof(argv[++i]);
          cmdline.camera.vi.x = std::atof(argv[++i]);
          cmdline.camera.vi.y = std::atof(argv[++i]);
          cmdline.camera.vi.z = std::atof(argv[++i]);
          cmdline.camera.vu.x = std::atof(argv[++i]);
          cmdline.camera.vu.y = std::atof(argv[++i]);
          cmdline.camera.vu.z = std::atof(argv[++i]);
        }
        else if (arg == "-win" || arg == "--size") {
          cmdline.windowSize.x = std::atoi(argv[++i]);
          cmdline.windowSize.y = std::atoi(argv[++i]);
        }
        else if (arg == "-o") {
          Renderer::screenShotFileName = argv[++i];
        }
        else if (arg == "-spp") {
          cmdline.spp = std::stoi(argv[++i]);
        }
        else
          usage("unknown cmdline arg '"+arg+"'");
      }


      // ******************************************************************
      // all input loaded, and all parameters parsed ... set-up comms
      // ******************************************************************
      MPIBackend mpiBackend(argc,argv,islandSize);
      Model::SP model = Model::load(Model::canonicalMasterFileName(inFileBase));
      if (model->bricks.size() != mpiBackend.workersSize)
        throw std::runtime_error("incompatible number of bricks and workers");
      const bool isMaster = mpiBackend.isMaster;
      int myRank = mpiBackend.islandRank();
      if (!isMaster)
        CUDA_CALL(SetDevice(mpiBackend.worker.gpuID));
      Renderer *renderer
        = createRenderer(rendererName,
                         &mpiBackend,model,
                         inFileBase,myRank,cmdline.spp);
      
      if (!isMaster) {
        MPIWorker worker(mpiBackend,renderer);
        worker.run();
        exit(0);
      }

      MPIMaster master(mpiBackend,renderer);
    
      VoPaTViewer viewer(master);
      box3f sceneBounds = model->getBounds();
      PRINT(sceneBounds);
      viewer.enableFlyMode();
      viewer.enableInspectMode();


      PING; PRINT(cmdline.configFileName);
      if (!cmdline.configFileName.empty()) {
        ModelConfig mc = ModelConfig::load(cmdline.configFileName);
        viewer.setCameraOrientation(mc.camera.from,
                                    mc.camera.at,
                                    mc.camera.up,
                                    mc.camera.fovy);
        viewer.camera.setUpVector(mc.camera.up);

        PRINT(mc.xf.getRange());
        PRINT(mc.xf.getDensity());
        PRINT(mc.xf.colorMap.size());
        master.setTransferFunction(mc.xf.colorMap,
                                      mc.xf.getRange(),
                                      mc.xf.getDensity());
      } 

      if (cmdline.camera.vu != vec3f(0.f)) {
        std::cout << "Camera from command line!"
                  << std::endl;
        viewer.setCameraOrientation(/*origin   */cmdline.camera.vp,
                                    /*lookat   */cmdline.camera.vi,
                                    /*up-vector*/cmdline.camera.vu,
                                    /*fovy(deg)*/cmdline.camera.fovy);
      } else if (cmdline.configFileName.empty()) {
        std::cout << "No camera in model, nor on command line - generating from bounds ...."
                  << std::endl;
        viewer.setCameraOrientation(/*origin   */
                                    sceneBounds.center()
                                    + vec3f(-.7f,.3f,+1.f)*1.5f*sceneBounds.span(),
                                    // + vec3f(-.3f,.7f,+1.f)*1.5f*sceneBounds.span(),
                                    /*lookat   */sceneBounds.center(),
                                    /*up-vector*/vec3f(0.f, 1.f, 0.f),
                                    /*fovy(deg)*/cmdline.camera.fovy);
      }
      viewer.setWorldScale(.1f*length(sceneBounds.span()));
      viewer.showAndRun();

    } catch (std::exception &e) {
      std::cout << "Fatal runtime error: " << e.what() << std::endl;
      exit(1);
    }
  }
} // ::vopat
