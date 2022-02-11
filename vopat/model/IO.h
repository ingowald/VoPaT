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
#include <fstream>

namespace vopat {

  template<typename T> inline T read(std::ifstream &in);
  template<typename T> inline void write(std::ofstream &out, const std::vector<T> &t);
  template<typename T> inline void write(std::ofstream &out, const T &t);
  
  template<typename T> inline void read(std::ifstream &in, T &t) { t = read<T>(in); }

  // ==================================================================
  template<typename T> inline void write(std::ofstream &out, const std::string &t)
  {
    write(out,int(t.size())); out.write((char*)t.data(),t.size());
  }
  template<> inline std::string read<std::string>(std::ifstream &in)
  {
    int len = read<int>(in);
    std::string s; s.resize(len);
    in.read((char*)s.data(),len);
    return s;
  }

  // ==================================================================
  template<typename T> inline void write(std::ofstream &out, const T &t)
  { out.write((char*)&t,sizeof(t)); }
  
  template<typename T> inline T read(std::ifstream &in)
  { T t; in.read((char*)&t,sizeof(t)); return t; }

  template<typename T> inline void write(std::ofstream &out, const std::vector<T> &t)
  { write(out,t.size()); out.write((const char*)t.data(),sizeof(T)*t.size()); }
  
}
