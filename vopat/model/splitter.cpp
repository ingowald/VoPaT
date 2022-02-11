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

#include "vopat/model/Model.h"
#include <fstream>

namespace vopat {

  box3i kdComputeRankRegion(int rank, int numRanks,
                            box3i region)
  {
    while (1) {
      if (numRanks == 1) return region;
      
      int lCount = numRanks / 2;
      int rCount = numRanks - lCount;
      
      float relSplit = lCount / float(numRanks);
      
      int dim = arg_max(region.size());
      box3i lRegion = region; 
      box3i rRegion = region;
      lRegion.upper[dim] = rRegion.lower[dim] =
        int(region.lower[dim] + relSplit * region.size()[dim]);
      if (rank >= lCount) {
        region   =  rRegion;
        rank     -= lCount;
        numRanks -= lCount;
      } else {
        region   =  lRegion;
        rank     -= 0;
        numRanks -= rCount;
      }
    }
  }
  
}

using namespace vopat;

int main(int ac, char **av)
{
}
