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

#include "Woodcock.h"
#include "NodeRenderer.h"

namespace vopat {

  Renderer *createRenderer_Woodcock(CommBackend *comm,
                                    Model::SP model,
                                    const std::string &fileNameBase,
                                    int rank,
                                    int numSPP)
  {
    VopatNodeRenderer *nodeRenderer
      = new VopatNodeRenderer
      (model,fileNameBase,rank,comm->worker.gpuID);
    return new RayForwardingRenderer(comm,nodeRenderer,numSPP);
  }

} // ::vopat
