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

#include "common/mpi/MPIRenderer.h"

namespace vopat {

XXX

struct VopatMPIRenderer : public MPIRenderer
  {
    static std::string screenShotFileName;

    VopatMPIRenderer(CommBackend *comm);

    /*! returns rank of this renderer; is -1 on master, and worker ID on workers */
    int myRank() const { return comm->isMaster?-1:comm->worker.withinIsland->rank; }
    int islandIndex() const { return comm->islandIndex(); }
    int islandCount() const { return comm->islandCount(); }
    
    /*! returns true if this is running on the master node */
    bool isMaster() const { return comm->isMaster; }
    
    /*! render a given frame; fbPointer will be null on workers, and
      point to app frame buffer on master */
    void render(uint32_t *fbPointer) override { ++accumID; }
    void resizeFrameBuffer(const vec2i &newSize) override
    {
      worldFbSize = newSize;
      if (isMaster())
        islandFbSize = 0;
      else {
        islandFbSize.x = newSize.x;
        islandFbSize.y
          = (newSize.y / islandCount())
          + (islandIndex() < (newSize.y % islandCount()));
      }
    }
    void resetAccumulation() override { accumID = -1; };
    void setCamera(const vec3f &from,
                   const vec3f &at,
                   const vec3f &up,
                   const float fovy) {
      // this->camera = camera;
      this->camera = Camera(worldFbSize,from,at,up,fovy);
    };

    virtual void screenShot() = 0;

    CommBackend    *const comm;
    int             accumID = -1;
    vec2i           islandFbSize;
    vec2i           worldFbSize;
    Camera          camera;
  };  
}
