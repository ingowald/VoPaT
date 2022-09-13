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

// #include "brix/mpi/MPIMaster.h"

#include "viewer/headless.h"
#ifndef HEADLESS
#include "viewer/isoDialog.h"
#endif
// #include "common/mpi/MPIMaster.h"
// #include "common/mpi/MPIWorker.h"
#include "viewer/AppInterface.h"
#include "model/Config.h"
#include <math.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

#include "submodules/cuteeOWL/qtOWL/OWLViewer.h"
#include "submodules/cuteeOWL/qtOWL/XFEditor.h"
#include "common/mpi/MPIBackend.h"

//#include "samples/common/owlViewer/OWLViewer.h"

namespace vopat {

  std::string rendererName = "wc";
    
  using qtOWL::range1f;
  
  struct {
    int spp = 1; //4;
    vec2i windowSize  = vec2i(1024,1024);
    float windowScale = 1.f;
  } cmdline;
  
  void usage(const std::string &msg)
  {
    if (msg != "") std::cerr << "Error: " << msg << std::endl << std::endl;
    std::cout << "Usage: ./brxViewer <inputfile.pbf> -p <partSpec.ps>" << std::endl;
    exit(msg != "");
  }

#ifdef HEADLESS
  struct VoPaTViewer : public Headless
  {
  public:
    typedef Headless inherited;
#else
  struct VoPaTViewer : public qtOWL::OWLViewer
  {
  public:
    typedef qtOWL::OWLViewer inherited;
#endif

    AppInterface &master;
    Model::SP model;
    VoPaTViewer(AppInterface &master,
                Model::SP model,
                ModelConfig::SP _modelConfig,
                qtOWL::XFEditor *xfEditor)
      : inherited("",cmdline.windowSize),
        master(master),
        model(model),
        xfEditor(xfEditor),
        modelConfig(_modelConfig)
    {
      // xfRange = model->valueRange;
    }
    
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
      
      modelConfig->camera.from = from;
      modelConfig->camera.at = at;
      modelConfig->camera.up = up;
      modelConfig->camera.fovy = fovy;

      master.setCamera(from,at,up,fovy);
      master.resetAccumulation();
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
        static double t0 = getCurrentTime();
        master.renderFrame(fbPointer);
      }      
      
      double t_now = getCurrentTime();
      static double avg_t = 0.;
      if (t_last >= 0)
        avg_t = 0.8*avg_t + 0.2*(t_now-t_last);

      if (displayFPS) {
        char title[1000];
        sprintf(title,"owlVoPaT - %.2f FPS",(1.f/avg_t));
        setTitle(title);
      }
      t_last = t_now;
    }
    

    void setKeyLight(uint32_t lightID)
    {
      PING;
      std::vector<vec3f> lightDirs = {
                                      vec3f(1.f,.1f,.1f),
                                      vec3f(-1.f,.1f,.1f),
                                      vec3f(.1f,+1.f,.1f),
                                      vec3f(.1f,-1.f,.1f),
                                      vec3f(.1f,.1f,+1.f),
                                      vec3f(.1f,.1f,-1.f),
                                      vec3f(-1.f,-1.f,-1.f),
                                      vec3f(-1.f,-1.f,+1.f),
                                      vec3f(-1.f,+1.f,-1.f),
                                      vec3f(-1.f,+1.f,+1.f),
                                      vec3f(+1.f,-1.f,-1.f),
                                      vec3f(+1.f,-1.f,+1.f),
                                      vec3f(+1.f,+1.f,-1.f),
                                      vec3f(+1.f,+1.f,+1.f),
      };
      vec3f lightDir = lightDirs[lightID % lightDirs.size()];
      modelConfig->lights.directional.resize(1);
      modelConfig->lights.directional[0].dir = lightDir;
      updateLights();
    }
    
    /*! this gets called when the user presses a key on the keyboard ... */
    virtual void key(char key, const vec2i &where)
    {
      PING;
      static uint32_t keyLightID = 0;
      
      switch (key) {
      case 'l':
        setKeyLight(++keyLightID);
        break;
      case 'L':
        setKeyLight(--keyLightID);
        break;
      case '!':
        screenShot();
        break;
      case 'V':
        displayFPS = !displayFPS;
        break;
      case '@': {
        const std::string xfFileName = "vopat.vpt";
        std::cout << "('@' key:) dumping transfer function/vopat model config to " << xfFileName << std::endl;
        modelConfig->save(xfFileName);
      } break;
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

    void updateLights()
    {
      master.setLights(modelConfig->lights.ambient,
                       modelConfig->lights.directional);
    }
   
#ifndef HEADLESS
    vec2i fbSize { -1,-1 };
#endif
    bool  displayFPS = true;

    // signals:
    //   ;
  public slots:
    void colorMapChanged(qtOWL::XFEditor *xf)
    {
      modelConfig->xf.colorMap = xf->getColorMap();
      master.setTransferFunction(modelConfig->xf.colorMap,
                                 modelConfig->xf.getRange(),
                                 modelConfig->xf.getDensity());
    };
    void rangeChanged(range1f r) 
    {
      modelConfig->xf.relDomain = xfEditor->getRelDomain();
      modelConfig->xf.absDomain = xfEditor->getAbsDomain();
      master.setTransferFunction(modelConfig->xf.colorMap,
                                 modelConfig->xf.getRange(),
                                 modelConfig->xf.getDensity());
    };
    /*! 'scale' is actually a percentage, with 100 meaning 'default' */
    void opacityScaleChanged(double scale)
    {
      modelConfig->xf.opacityScale = xfEditor->getOpacityScale();
      master.setTransferFunction(modelConfig->xf.colorMap,
                                 modelConfig->xf.getRange(),
                                 modelConfig->xf.getDensity());
    };
    void isoToggled(int iso, bool enabled)
    {
      if (iso>=0 && iso<ModelConfig::maxISOs) {
        modelConfig->iso.active[iso] = (int)enabled;
      }
      int numActive = (int)std::count_if(modelConfig->iso.active.begin(),
                                         modelConfig->iso.active.end(),
                                         [](int i) { return i!=0; });
      master.setISO(numActive,
                    modelConfig->iso.active,
                    modelConfig->iso.values,
                    modelConfig->iso.colors);
    }
    void isoColorChanged(int iso, QColor clr)
    {
      if (iso>=0 && iso<ModelConfig::maxISOs) {
        modelConfig->iso.colors[iso] = {(float)clr.redF(),
                                        (float)clr.greenF(),
                                        (float)clr.blueF()};
      }
      int numActive = (int)std::count_if(modelConfig->iso.active.begin(),
                                         modelConfig->iso.active.end(),
                                         [](int i) { return i!=0; });
      master.setISO(numActive,
                    modelConfig->iso.active,
                    modelConfig->iso.values,
                    modelConfig->iso.colors);
    }
    void isoValueChanged(int iso, float value)
    {
      if (iso>=0 && iso<ModelConfig::maxISOs) {
        modelConfig->iso.values[iso] = value;
      }
      int numActive = (int)std::count_if(modelConfig->iso.active.begin(),
                                         modelConfig->iso.active.end(),
                                         [](int i) { return i!=0; });
      master.setISO(numActive,
                    modelConfig->iso.active,
                    modelConfig->iso.values,
                    modelConfig->iso.colors);
    }
                                     
  public:
    qtOWL::XFEditor *xfEditor;
    ModelConfig::SP modelConfig;
  };

  extern "C" int main(int argc, char **argv)
  {
    cudaFree(0);
    CUDA_SYNC_CHECK();
    
    // ******************************************************************
    // parse OUR stuff
    // ******************************************************************
    ModelConfig::SP modelConfig = std::make_shared<ModelConfig>(); 
    int islandSize = -1;
    try {
      std::string inFileBase = "";
      for (int i=1;i<argc;i++) {
        const std::string arg = argv[i];
        if (arg[0] != '-') {
          inFileBase = arg;
        } 
        else if (arg == "--renderer" || arg == "-r") {
          rendererName = argv[++i];
        }
        else if (arg == "--island-size" || arg == "-is") {
          islandSize = std::stoi(argv[++i]);
        }
        else if (arg == "-c" || arg == "--config") {
          const std::string configFileName = argv[++i];
          try {
            *modelConfig = ModelConfig::load(configFileName);
          } catch (...) {
            // master might not be able to load, that's OK
          }
        }
        else if (arg == "--camera") {
          modelConfig->camera.from.x = std::atof(argv[++i]);
          modelConfig->camera.from.y = std::atof(argv[++i]);
          modelConfig->camera.from.z = std::atof(argv[++i]);
          modelConfig->camera.at.x = std::atof(argv[++i]);
          modelConfig->camera.at.y = std::atof(argv[++i]);
          modelConfig->camera.at.z = std::atof(argv[++i]);
          modelConfig->camera.up.x = std::atof(argv[++i]);
          modelConfig->camera.up.y = std::atof(argv[++i]);
          modelConfig->camera.up.z = std::atof(argv[++i]);
        }
        else if (arg == "-fovy") {
          modelConfig->camera.fovy = std::atof(argv[++i]);
        }
        else if (arg == "-win" || arg == "--size") {
          cmdline.windowSize.x = std::atoi(argv[++i]);
          cmdline.windowSize.y = std::atoi(argv[++i]);
        }
        else if (arg == "-o") {
          AddLocalFBsLayer::screenShotFileName = argv[++i];
        }
        else if (arg == "-spp") {
          cmdline.spp = std::stoi(argv[++i]);
        }
        else
          usage("unknown cmdline arg '"+arg+"'");
      }

      // ******************************************************************
      // cmdline parsed, set up mpi
      // ******************************************************************
      MPIBackend mpiBackend(argc,argv,islandSize);
    
      // ******************************************************************
      // load model, and check that it meets our mpi config
      // ******************************************************************
      Model::SP model = Model::load(Model::canonicalMasterFileName(inFileBase));
      if (model->numBricks != mpiBackend.numRanksPerIsland)//workersSize)
        throw std::runtime_error("incompatible number of bricks and workers");

      // ******************************************************************
      // create renderer, workers, etc.
      // ******************************************************************
      const bool isMaster = mpiBackend.isMaster;
      int myRank = mpiBackend.islandRank();
// #if VOPAT_UMESH
//       if (isMaster) {
//         std::cout << "got the following brick domains: " << std::endl;
//         for (auto brick : model->bricks)
//           std:: cout << "  - " << brick->domain << std::endl;
//       }
// #endif

      Brick::SP brick
        = isMaster
        ? Brick::SP()
        : model->createBrick(myRank);
      if (brick) {
        brick->loadConstantData(Model::canonicalBrickFileName(inFileBase,brick->ID));
        brick->loadTimeStep(Model::canonicalTimeStepFileName(inFileBase,brick->ID,
                                                             "unknown",0));
      }
      
      Volume::SP volume = Volume::createFrom(brick);
        
      VopatRenderer::SP renderer
        = VopatRenderer::create(&mpiBackend,volume);

      AppInterface appInterface(&mpiBackend,renderer);
      if (!isMaster) {
        CUDA_CALL(SetDevice(mpiBackend.worker.gpuID));
        appInterface.runWorker();
        exit(0);
      }
      CUDA_SYNC_CHECK();

      // ******************************************************************
      // initialize all not-yet-set values in our model config from model
      // ******************************************************************
      if (modelConfig->xf.absDomain.is_empty())
        modelConfig->xf.absDomain = model->valueRange;
      if (modelConfig->xf.colorMap.empty()) {
        auto cm = qtOWL::ColorMapLibrary().getMap(0);
        modelConfig->xf.colorMap = cm;
      }
      box3f sceneBounds = model->getBounds();
      if (modelConfig->camera.up == vec3f(0.f)) {
        modelConfig->camera.from
          = sceneBounds.center()
          + vec3f(-.7f,.3f,+1.f)*1.5f*sceneBounds.span();
        modelConfig->camera.at = sceneBounds.center();
        modelConfig->camera.up = vec3f(0.f, 1.f, 0.f);
        modelConfig->camera.fovy = 70.f;
      }

      QApplication app(argc,argv);
      
#ifndef HEADLESS
      // ******************************************************************
      // this is the master - set up window
      // ******************************************************************
      QMainWindow guiWindow;
      qtOWL::XFEditor *xfEditor = new qtOWL::XFEditor(model->valueRange);

      // -------------------------------------------------------
      // set up the main viewer class
      // -------------------------------------------------------
      VoPaTViewer viewer(appInterface,model,modelConfig,xfEditor);
      viewer.enableFlyMode();
      viewer.enableInspectMode();
      viewer.setCameraOrientation(modelConfig->camera.from,
                                  modelConfig->camera.at,
                                  modelConfig->camera.up,
                                  modelConfig->camera.fovy);
      viewer.camera.setUpVector(modelConfig->camera.up);

      viewer.setWorldScale(.1f*length(sceneBounds.span()));

      // -------------------------------------------------------
      // initialize gui widgets from saved config
      // -------------------------------------------------------
      guiWindow.setCentralWidget(xfEditor);

      QObject::connect(xfEditor,&qtOWL::XFEditor::colorMapChanged,
                       &viewer, &VoPaTViewer::colorMapChanged);
      QObject::connect(xfEditor,&qtOWL::XFEditor::rangeChanged,
                       &viewer, &VoPaTViewer::rangeChanged);
      QObject::connect(xfEditor,&qtOWL::XFEditor::opacityScaleChanged,
                       &viewer, &VoPaTViewer::opacityScaleChanged);

      // save both here, because the first call to the set() function
      // below will overwrite BOTH
      const interval<float> relDomain = modelConfig->xf.relDomain;
      const interval<float> absDomain = modelConfig->xf.absDomain;
      const float opacityScale = modelConfig->xf.opacityScale;
      
      xfEditor->setColorMap(modelConfig->xf.colorMap);
      xfEditor->setOpacityScale(opacityScale);
      xfEditor->setRelDomain(relDomain);
      xfEditor->setAbsDomain(absDomain);

      interval<float> isoRange = absDomain;
      if (const char *env = getenv("VOPAT_ISO_RANGE"))
        sscanf(env,"%f %f",&isoRange.lower,&isoRange.upper);
      // xfEditor->cmSelectionChanged(0);
      // xfEditor->opacityScaleChanged(xfEditor->getOpacityScale());

      
      IsoDialog isoDialog(isoRange);
      QObject::connect(&isoDialog,&IsoDialog::isoToggled,
                       &viewer, &VoPaTViewer::isoToggled);
      QObject::connect(&isoDialog,&IsoDialog::isoColorChanged,
                       &viewer, &VoPaTViewer::isoColorChanged);
      QObject::connect(&isoDialog,&IsoDialog::isoValueChanged,
                       &viewer, &VoPaTViewer::isoValueChanged);

    
      isoDialog.setISOs(modelConfig->iso.active,
                        modelConfig->iso.values,
                        modelConfig->iso.colors);

      // -------------------------------------------------------
      // and create the window ...
      // -------------------------------------------------------
      viewer.show();
      viewer.updateLights();
      
      guiWindow.show();
      isoDialog.raise();
      isoDialog.show();
      
      return app.exec();
#else
      // Headless viewer
      VoPaTViewer viewer(appInterface,model,modelConfig,NULL);
      viewer.setCameraOrientation(modelConfig->camera.from,
                                  modelConfig->camera.at,
                                  modelConfig->camera.up,
                                  modelConfig->camera.fovy);
      viewer.camera.setUpVector(modelConfig->camera.up);

      viewer.setWorldScale(.1f*length(sceneBounds.span()));
      viewer.resize(cmdline.windowSize);
      viewer.updateLights();

      // Emulate what slots do initially, when xfeditor is initialized..
      {
        appInterface.setTransferFunction(modelConfig->xf.colorMap,
                                   modelConfig->xf.getRange(),
                                   modelConfig->xf.getDensity());

        int numActive = (int)std::count_if(modelConfig->iso.active.begin(),
                                           modelConfig->iso.active.end(),
                                           [](int i) { return i!=0; });
        appInterface.setISO(numActive,
                      modelConfig->iso.active,
                      modelConfig->iso.values,
                      modelConfig->iso.colors);
      }
      viewer.run();
      return 0;
#endif
    } catch (std::exception &e) {
      std::cout << "Fatal runtime error: " << e.what() << std::endl;
      exit(1);
    }
  }
} // ::vopat
