// ======================================================================== //
// Copyright 2022++ Ingo Wald                                               //
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

#include "vopat/common.h"
#include "vopat/mpi/MPIRenderer.h"
#include "vopat/mpi/Comms.h"
#include "vopat/render/CUDAArray.h"
#include "vopat/render/Camera.h"
#include <sstream>

namespace vopat {

#if 0
  struct Prof {
    enum { is_active = 0 };
    Prof(const std::string &, int) {};
    void enter() {}
    void leave() {}
  };
#else
  struct Prof {
    enum { is_active = 1 };
    Prof(const std::string &name, int rank) : name(name), rank(rank) {};
    void enter()
    {
      t_enter = getCurrentTime();
      if (t0 < 0.) return;
    }
    void leave()
    {
      double t = getCurrentTime();
      if (t0 < 0.f) {
        t0 = t;
      } else {
        t_inside += (t-t_enter);
        numInside++;
        while (numInside >= nextPing) {
          std::stringstream ss;
          ss << "#(" << rank << ") " << name << "\t avg time in:" << prettyDouble(t_inside/numInside) << ", count " << numInside << ", relative " << prettyDouble(100.f*t_inside/(t-t0)) << "%";
          nextPing *= 2;
          printf("%s\n",ss.str().c_str()); fflush(0);
        }
      }
    }

    const int rank;
    const std::string name;
    double t0 = -1.;
    double t_enter;
    double t_inside = 0.;
    int numInside = 0;
    int nextPing = 1;
  };
#endif
  
  /*! base abstraction for a renderer, not specifying _how_ rendering
      works, just what kind of interface it offers to an
      application */
  struct Renderer : public MPIRenderer {

    static std::string screenShotFileName;

    Renderer(CommBackend *comm);

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
