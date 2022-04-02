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

#include "Renderer.h"

namespace vopat {
  
  Renderer *createRenderer_Woodcock(CommBackend *comm,
                                    Model::SP model,
                                    const std::string &fileNameBase,
                                    int rank, int numSPP);
  Renderer *createRenderer_CellMarch(CommBackend *comm,
                                     Model::SP model,
                                     const std::string &fileNameBase,
                                     int rank, int numSPP);
  
  /*! woodcock-"style" renderer without forwarding of shadow rays, as
      if one did woodcock in each rank, and used compositing */
  Renderer *createRenderer_WrongShadows(CommBackend *comm,
                                        Model::SP model,
                                        const std::string &fileNameBase,
                                        int rank, int numSPP);

  /*! woodcock-"style" renderer that doesn't do any shadows, just
      emisison-absoption style rendering */
  Renderer *createRenderer_NoShadows(CommBackend *comm,
                                     Model::SP model,
                                     const std::string &fileNameBase,
                                     int rank, int numSPP);

  /*! creates a renderer from the given name (e.g., "woodcock" or
      "cell-march") */
  Renderer *createRenderer(const std::string &rendererName,
                           CommBackend *comm,
                           Model::SP model,
                           const std::string &fileNameBase,
                           int rank, int numSPP)
  {
    if (rendererName == "wc" || rendererName == "woodock")
      return createRenderer_Woodcock(comm,model,fileNameBase,rank,numSPP);
    if (rendererName == "ws" || rendererName == "wrong-shadows")
      return createRenderer_WrongShadows(comm,model,fileNameBase,rank,numSPP);
    if (rendererName == "ns" || rendererName == "no-shadows")
      return createRenderer_NoShadows(comm,model,fileNameBase,rank,numSPP);
    if (rendererName == "cm" || rendererName == "cell-march")
      return createRenderer_CellMarch(comm,model,fileNameBase,rank,numSPP);
    throw std::runtime_error("unknown renderer mode '"+rendererName+"'");
  }

}

