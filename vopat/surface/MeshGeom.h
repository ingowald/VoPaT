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

#include "common/vopat.h"
#include "miniScene/Scene.h"
#include <owl/owl.h>

namespace vopat {

  struct MeshGeom {

    /*! the device-side struct used for the actual OWL geometry */
    struct DD {
      vec3f *vertices;
      vec3i *indices;
      vec3f  diffuseColor;
    };

    struct PRD {
      vec3f diffuseColor;
      vec3f N;
      float t;
    };

    static OWLGeom createGeom(OWLContext owl, mini::Mesh::SP mesh);
    
    /*! defines 'our' geometry type in the given owl context */
    static void defineGeometryType(OWLContext owl, OWLModule devCode);
    
    /*! geometry for actual triangle mesh surface geometry */
    static OWLGeomType meshGT;
  };
  
} // :: vopat
