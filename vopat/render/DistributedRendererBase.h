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

namespace vopat {

  /*! base abstraction for a renderer, not specifying _how_ rendering
      works, just what kind of interface it offers to an
      application */
  struct Renderer : public MPIRenderer {

    static std::string screenShotFileName;

    Renderer(CommBackend *comm);

    /*! returns rank of this renderer; is -1 on master, and worker ID on workers */
    int myRank() const { return comm->isMaster?-1:comm->worker.withinIsland->rank; }
    
    /*! returns true if this is running on the master node */
    bool isMaster() const { return comm->isMaster; }
    
    /*! render a given frame; fbPointer will be null on workers, and
      point to app frame buffer on master */
    void render(uint32_t *fbPointer) override { ++accumID; }
    void resizeFrameBuffer(const vec2i &newSize) override { fbSize = newSize; }
    void resetAccumulation() override { accumID = -1; };
    void setCamera(const vec3f &from,
                   const vec3f &at,
                   const vec3f &up,
                   const float fovy) {
      // this->camera = camera;
      this->camera = Camera(fbSize,from,at,up,fovy);
    };

    virtual void screenShot() = 0;

    CommBackend    *const comm;
    int             accumID = -1;
    vec2i           fbSize;
    Camera          camera;
  };

  /*! a renderer where worker nodes fill a *local* frame buffer, and
      the master collects the final *added* images; _how_ a client
      renders its local frame buffer is still virtual, but all the
      frame buffer handling, compositing, and collecting of pixels is
      done by this class */
  struct AddWorkersRenderer
    : public Renderer
  {
    AddWorkersRenderer(CommBackend *comm) : Renderer(comm) {}
    
    void resizeFrameBuffer(const vec2i &newSize) override;
    void render(uint32_t *fbPointer) override;

    static void composeRegion(uint32_t *results,
                              const vec2i &ourRegionSize,
                              const small_vec3f *inputs,
                              int numRanks);

    // ==================================================================
    
    /*! asks the _workers_ to render locally; will not get called on
        master, but assumes that all workers render their local frame
        buffer(s) that we'll then merge */
    virtual void renderLocal() = 0;

    /*! on workres, this is the result of 'renderLocal()', and is
        supposed to contain what each workers wants to contribute to
        the final image (the final image is the *addition* of all
        these local fbs). On the master this isn't used at all, as the
        master directly gathers final results into masterFB */
    CUDAArray<small_vec3f> localFB;
    
    /*! "temporary" buffer where current node receives all the lines
        that it has to compose */
    // small_vec3f    *compInputsMemory = nullptr;
    CUDAArray<small_vec3f> compInputsMemory;//localAccumBuffer;

    /*! temporary memory where this node writes its composed lines
        to, and from where those can then be sent on to the master */
    // uint32_t       *compResultMemory = nullptr;
    CUDAArray<uint32_t> compResultMemory;
    vec2i           islandFbSize;
    vec2i           fullFbSize;

    CUDAArray<uint32_t> masterFB;
  };
}
