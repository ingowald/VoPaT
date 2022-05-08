// ======================================================================== //
// Copyright 2020-2020 Ingo Wald                                            //
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

#include <QApplication>
#include <QDesktopWidget>
#include <cuda_runtime.h>
#include "headless.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION 1
#include "stb/stb_image_write.h"

using namespace owl;

namespace vopat {

  Headless::Headless(const std::string &title, const vec2i &initWindowSize)
    : fbSize(initWindowSize)
    , title(title)
  {
    owl::vec2i ws = initWindowSize;
    if (ws == vec2i(0,0)) {
      QRect rec = QApplication::desktop()->screenGeometry();
      int height = rec.height();
      int width = rec.width();
      ws = vec2i{width,height};
    }

    cudaMalloc(&fbPointer,sizeof(uint32_t)*ws.x*ws.y);
    fbSize = ws;
  }

  Headless::~Headless()
  {
    cudaFree(fbPointer);
  }

  void Headless::run()
  {
    int frameID=0;
    int screenshotID=10;//-1
    int stopID=50;
    std::string screenshotFileName = "";
    while (++frameID) {

      double t1 = getCurrentTime();
      render();
      double t2 = getCurrentTime();
      // std::cout << frameID << ';' << t2-t1 << '\n';

      if (frameID==screenshotID) {
        // Save png
        std::string fileName = screenshotFileName.empty() ? "offline.png" : screenshotFileName;
        std::vector<uint32_t> pixels(fbSize.x*fbSize.y);
        cudaError_t err = cudaMemcpy(pixels.data(),fbPointer,fbSize.x*fbSize.y*sizeof(uint32_t),
                                     cudaMemcpyDeviceToHost);

        if (err==cudaSuccess) {
          stbi_write_png(fileName.c_str(),fbSize.x,fbSize.y,4,
                         pixels.data(),fbSize.x*sizeof(uint32_t));
          std::cout << "#owl.viewer: frame buffer written to " << fileName << std::endl;
        } else {
          std::cerr << cudaGetErrorString(cudaGetLastError()) << '\n';
        }
      }

      if (frameID==stopID)
        break;
    }
  }

  void Headless::resize(const owl::vec2i &newSize)
  {
    if (newSize==fbSize)
      return;

    fbSize = newSize;

    cudaFree(fbPointer);
    cudaMalloc(&fbPointer,fbSize.x*fbSize.y*sizeof(uint32_t));
  }

  void Headless::setTitle(const std::string &s)
  {
    title = s;
    std::cout << title << '\n';
  }

  void Headless::setWorldScale(const float worldScale)
  {
    camera.motionSpeed = worldScale / sqrtf(3.f);
  }

    /*! re-computes the 'camera' from the 'cameracontrol', and notify
    app that the camera got changed */
  void Headless::updateCamera()
  {
    // camera.digestInto(simpleCamera);
    // if (isActive)
    camera.lastModified = getCurrentTime();
  }

    /*! set a new orientation for the camera, update the camera, and
    notify the app */
  void Headless::setCameraOrientation(/* camera origin    : */const vec3f &origin,
                                      /* point of interest: */const vec3f &interest,
                                      /* up-vector        : */const vec3f &up,
                                      /* fovy, in degrees : */float fovyInDegrees)
  {
    camera.setOrientation(origin,interest,up,fovyInDegrees,false);
    updateCamera();
  }
} // ::vopat
