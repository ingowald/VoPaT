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

#pragma once

#include "vopat/mpi/MPICommon.h"
#include "vopat/mpi/MPIRenderer.h"

namespace vopat {
  
  /*! mpi rendering interface for the *receivers* on the workers;
      these recive the broadcasts from the MPIMaster, and execute them
      on their (virutal) renderer. */
  struct MPIWorker : public MPICommon
  {
    MPIWorker(MPIBackend &mpi, MPIRenderer *renderer);

    // int myRank() const { return mpi.myRank(); }
    int islandRank() const { return mpi.islandRank(); }
    
    /*! the 'main loop' that receives and executes cmmands sent by the master */
    void run();

    /*! @{ command handlers - each corresponds to exactly one command
        sent my the master */
    void cmd_terminate();
    void cmd_renderFrame();
    void cmd_resizeFrameBuffer();
    void cmd_resetAccumulation();
    void cmd_setCamera();
    void cmd_setTransferFunction();
    void cmd_setISO();
    void cmd_setShadeMode();
    void cmd_setNodeSelection();
    void cmd_screenShot();
    void cmd_setLights();
    void cmd_script();
    /* @} */
    
    MPIRenderer *renderer = nullptr;
    MPIBackend  &mpi;
  };
  

} // ::vopat
