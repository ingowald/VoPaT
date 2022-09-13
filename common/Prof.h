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

#include "common/vopat.h"

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
  
}
