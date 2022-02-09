// ======================================================================== //
// Copyright 2018-2020 Ingo Wald                                            //
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

#include "brix/mpi/MPIBackend.h"
#include <sstream>

using namespace brix;

int main(int ac, char **av)
{
  int numRanksPerIsland = 2;
  for (int i=1;i<ac;i++) {
    const std::string arg = av[i];
    if (arg == "-npi" || arg == "-rpi")
      numRanksPerIsland = atoi(av[++i]);
    else throw std::runtime_error("unknown arg "+arg);
  }
  MPIBackend mpi(ac,av,numRanksPerIsland);

  mpi.barrierAll();
  if (mpi.isMaster) std::cout << "Hello from master" << std::endl;
  mpi.barrierAll();
  
  if (mpi.isWorker) {
    std::vector<int> islandPeers(mpi.numRanksPerIsland);
    mpi.worker.withinIsland->allGather(islandPeers,mpi.workersRank);
    std::stringstream ss;
    ss << "worker rank " << mpi.workersRank
       << " in island " << mpi.worker.islandIdx << " with peers:";
    for (auto peer : islandPeers)
      ss << peer << " ";
    ss << std::endl;
    std::cout << ss.str();
  }
  
  mpi.finalize();
}
