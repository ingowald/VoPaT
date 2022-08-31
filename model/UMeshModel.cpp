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

#include "model/UMeshModel.h"
#include "model/IO.h"
#include <fstream>

namespace vopat {

  std::vector<box4f> Brick::makeShards(int numShards)
  {
    struct ShardNode {
      box3f domain;
      umesh::range1f valueRange;
      int childID = -1;
      int numShards;
    };

    // ------------------------------------------------------------------
    // BUILD kd-tree that defines the shards
    // ------------------------------------------------------------------
    std::vector<ShardNode> nodes;
    nodes.resize(1);
    nodes[0].domain = getDomain();
    nodes[0].childID = -1;
    nodes[0].numShards = numShards;
    std::stack<int> todo; todo.push(0);
    while (!todo.empty()) {
      int nodeID = todo.top();todo.pop();
      if (nodes[nodeID].numShards == 1)
        continue;
      int childID = nodes.size();
      nodes[nodeID].childID = childID;
      nodes.push_back({});
      nodes.push_back({});

      int Nl = numShards/2;
      int Nr = numShards - Nl;
      float ratio = float(Nl)/float(numShards);
      auto &pDomain = nodes[nodeID].domain;
      auto &lDomain = nodes[childID+0].domain;
      auto &rDomain = nodes[childID+1].domain;

      int dim = arg_max(pDomain.size());
      float mid = pDomain.lower[dim] + pDomain.size()[dim]*ratio;
      
      lDomain = rDomain = domain;
      lDomain.upper[dim] = rDomain.lower[dim] = mid;

      nodes[childID+0].numShards = Nl;
      nodes[childID+1].numShards = Nr;

      if (Nl > 1) todo.push(childID+0);
      if (Nr > 1) todo.push(childID+1);      
    };
    
    // ------------------------------------------------------------------
    // RASTER the umesh
    // ------------------------------------------------------------------
    std::vector<umesh::UMesh::PrimRef> prims = umesh->createVolumePrimRefs();
    for (auto prim : prims) {
      umesh::box3f primBounds = umesh->getBounds(prim);
      umesh::range1f primRange = umesh->getValueRange(prim);

      todo.push(0);
      while(!todo.empty()) {
        int nodeID = todo.top(); todo.pop();
        const int childID = nodes[nodeID].childID;
        if (childID >= 0) {
          for (int c=0;c<2;c++)
            if (nodes[childID+c].domain.overlaps((const box3f&)primBounds))
              todo.push(nodes[nodeID].childID+c);
          
        }
        else
          nodes[nodeID].valueRange.extend(primRange);
      }
    }
    
    // ------------------------------------------------------------------
    // gather the leaves
    // ------------------------------------------------------------------
    std::vector<box4f> result;
    for (auto &node : nodes) {
      if (node.childID >= 0) continue;
      result.push_back({ { node.domain.lower.x,
                           node.domain.lower.y,
                           node.domain.lower.z,
                           node.valueRange.lower },
                         { node.domain.upper.x,
                           node.domain.upper.y,
                           node.domain.upper.z,
                           node.valueRange.upper }});
    }
    return result;
  }

  void UMeshBrick::UMeshBrick(const std::string &constantDataFileName)
  {
    umesh = umesh::UMesh::loadFrom(constantDataFileName);
    valueRange = {};
    for (auto v : umesh->perVertex->values)
      valueRange.extend(v);
    PRINT(umesh->toString());
  }
  
  void UMeshBrick::write(std::ostream &out) const
  {
// #if VOPAT_UMESH
//       write(out,brick->domain);
// #else
//       write(out,brick->voxelRange);
//       write(out,brick->cellRange);
//       write(out,brick->spaceRange);
//       write(out,brick->numVoxels);
//       write(out,brick->numCells);
//       write(out,brick->numVoxelsParent);
// #endif
  }
  
}
