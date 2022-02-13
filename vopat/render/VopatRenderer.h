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

#include "vopat/render/DistributedRendererBase.h"
#include "vopat/model/Model.h"

namespace vopat {

  struct Ray {
    struct {
      uint32_t    pixelID : 31;
      uint32_t    isShadow:  1;
    };
    vec3f       origin;
    small_vec3f direction;
    small_vec3f throughput;
  };
  
  struct Globals {
    int          myRank, numWorkers;
    int          sampleID;
    Camera       camera;
    vec2i        fbSize;
    small_vec3f *fbPointer;
    int         *perRankSendOffsets;
    int         *perRankSendCounts;
    int         *rayNextNode;
    box3f       *rankBoxes;
    Ray         *rayQueueIn;
    Ray         *rayQueueOut;
  };

  struct VopatRenderer : public AddWorkersRenderer {
    VopatRenderer(CommBackend *comm,
                  Model::SP model,
                  int numSPP);

    // ==================================================================
    // main things we NEED to implement
    // ==================================================================
    
    /*! asks the _workers_ to render locally; will not get called on
        master, but assumes that all workers render their local frame
        buffer(s) that we'll then merge */
    void renderLocal() override;
    void screenShot() override;

    // ==================================================================
    // things we intercept to know what to do
    // ==================================================================
    
    void resizeFrameBuffer(const vec2i &newSize)  override;
    void setCamera(const vec3f &from,
                   const vec3f &at,
                   const vec3f &up,
                   const float fov) override;

    void traceRaysLocally();
    void createSendQueue();
    int  exchangeRays();
    
    
    /*! one box per rank, which rays can use to find neext rank to send to */
    CUDAArray<box3f>       rankBoxes;

    /*! ray queues we are expected to trace in the next step */
    CUDAArray<Ray>         rayQueueIn;
    
    /*! ray queues for sending out; in this one rays are sorted by
        rank they are supposed to go to */
    CUDAArray<Ray>         rayQueueOut;
    
    /*! one int per ray, which - after tracing locally, says which
        rank to go to next; -1 meaning "done" */
    CUDAArray<int>         rayNextNode;
    
    /*! one entry per ray, telling how many rays _we_ want to send to
        that given rank */
    CUDAArray<int>         perRankSendCounts;
    CUDAArray<int>         perRankSendOffsets;
    
    Globals globals;
    int numRaysInQueue;
  };
  
}
