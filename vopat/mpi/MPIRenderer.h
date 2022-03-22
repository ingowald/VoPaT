// ======================================================================== //
// Copyright 2018-2022 Ingo Wald                                            //
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

namespace vopat {

  /*! base abstraction for a parallel renderer that the
      mpimaster/mpiworker framework can talk to. _what_ that renderer
      does with the commands (like 'resize', 'render' etc), is up the
      derived class */
  struct MPIRenderer {
    struct DirectionalLight {
      vec3f dir;
      vec3f rad;
    };
    
    /*! render into the provided application frame buffer; this
        pointer will be 0 on workers */
    virtual void render(uint32_t *appFbPointer)          = 0;
    virtual void resizeFrameBuffer(const vec2i &newSize) = 0;
    virtual void resetAccumulation()                     = 0;
    virtual void setCamera(const vec3f &from,
                           const vec3f &at,
                           const vec3f &up,
                           float fovy) = 0;
    virtual void setTransferFunction(const std::vector<vec4f> &cm,
                                     const interval<float> &range,
                                     const float density) {}
    virtual void setISO(const std::vector<int> &active,
                        const std::vector<float> &values,
                        const std::vector<vec3f> &colors) {}
    /*! dump a screenshot, possibhly including some per-rank debugging
        screenshots (mostly for debugging; it should be the app that
        provides the "real" screen shots) */
    virtual void screenShot()                            = 0;

    /*! unformatted script-like command - assuems renderer can parse this */
    virtual void script(const std::string &s) {};
    virtual void setShadeMode(int mode) {};
    virtual void setLights(float ambient,
                           const std::vector<MPIRenderer::DirectionalLight> &dirLights) {};
  };
  
}
