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

#pragma once

#include <string>
#include "owl/common/math/vec.h"
#include "qtOWL/Camera.h"

namespace vopat {

  struct Headless {

    Headless(const std::string &title = "OWL Sample Viewer",
             const owl::vec2i &initWindowSize=owl::vec2i(1200,800));

    ~Headless();

    void run();

    virtual void resize(const owl::vec2i &newSize);

    void setTitle(const std::string &s);

    virtual void render() {}

    virtual void key(char key, const owl::vec2i &/*where*/) {}

    virtual void cameraChanged() {}

    owl::vec2i getWindowSize() const { return fbSize; }

    qtOWL::Camera &getCamera() { return camera; }

    virtual void setWorldScale(const float worldScale);

    /*! set a new window aspect ratio for the camera, update the
      camera, and notify the app */
    void setAspect(const float aspect)
    {
      camera.setAspect(aspect);
      updateCamera();
    }

    void updateCamera();

        /*! set a new orientation for the camera, update the camera, and
      notify the app */
    void setCameraOrientation(/* camera origin    : */const owl::vec3f &origin,
                              /* point of interest: */const owl::vec3f &interest,
                              /* up-vector        : */const owl::vec3f &up,
                              /* fovy, in degrees : */float fovyInDegrees);

  // protected:


    owl::vec2i  fbSize { 0 };
    uint32_t    *fbPointer { nullptr };
    std::string title;
    qtOWL::Camera camera;

  };

} // ::vopat
