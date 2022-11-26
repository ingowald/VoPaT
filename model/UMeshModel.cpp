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

  std::string UMeshBrick::toString() const
  {
    std::stringstream ss;
    ss << "UMeshBrick #"<<ID << " " << domain
       << " " << (umesh?umesh->toString():std::string("<no mesh>"));
    
    return ss.str();
  }
  
  std::vector<VolumeProxy> UMeshBrick::makeVolumeProxies(int init_numVolumeProxies)
  {
    struct VolumeProxyNode {
      box3f domain;
      range1f valueRange;
      int childID = -1;
      int numVolumeProxies = 0;
    };

    // ------------------------------------------------------------------
    // BUILD kd-tree that defines the volumeProxies
    // ------------------------------------------------------------------
    std::vector<VolumeProxyNode> nodes;
    nodes.resize(1);
    nodes[0].domain = this->domain;//umesh->getBounds();
    nodes[0].childID = -1;
    nodes[0].numVolumeProxies = init_numVolumeProxies;
    {
      std::stack<int> todo; todo.push(0);
      while (!todo.empty()) {
        int nodeID = todo.top();todo.pop();
        if (nodes[nodeID].numVolumeProxies == 1)
          continue;
        int childID = nodes.size();
        nodes[nodeID].childID = childID;
        nodes.push_back({});
        nodes.push_back({});

        int Nl = nodes[nodeID].numVolumeProxies/2;
        int Nr = nodes[nodeID].numVolumeProxies - Nl;
        float ratio = float(Nl)/float(nodes[nodeID].numVolumeProxies);

        auto pDomain = nodes[nodeID].domain;

        int dim = arg_max(pDomain.size());
        float mid = pDomain.lower[dim] + pDomain.size()[dim]*ratio;

        
        box3f lDomain = pDomain; lDomain.upper[dim] = mid;
        box3f rDomain = pDomain; rDomain.lower[dim] = mid;

        nodes[childID+0].numVolumeProxies = Nl;
        nodes[childID+1].numVolumeProxies = Nr;
        nodes[childID+0].domain = lDomain;
        nodes[childID+1].domain = rDomain;

        if (Nl > 1) todo.push(childID+0);
        if (Nr > 1) todo.push(childID+1);      
      };
    }

    // ------------------------------------------------------------------
    // RASTER the umesh
    // ------------------------------------------------------------------
    std::vector<umesh::UMesh::PrimRef> prims = umesh->createVolumePrimRefs();
    for (auto prim : prims) {
      umesh::box3f primBounds = umesh->getBounds(prim);
      umesh::range1f primRange = umesh->getValueRange(prim);
      if (primRange.lower > primRange.upper)
        throw std::runtime_error("invalid prim!?");
      std::stack<int> todo; todo.push(0);
      while(!todo.empty()) {
        int nodeID = todo.top(); todo.pop();
        const int childID = nodes[nodeID].childID;
        if (childID >= 0) {
          for (int c=0;c<2;c++) {
            if (nodes[childID+c].domain.overlaps((const box3f&)primBounds))
              todo.push(childID+c);
          }
        }
        else {
          nodes[nodeID].valueRange.extend((const range1f&)primRange);
        }
      }
    }

    // ------------------------------------------------------------------
    // gather the leaves
    // ------------------------------------------------------------------
    std::vector<VolumeProxy> result;
    int dbgID = 0;
    for (auto node : nodes) {
      int nodeID = dbgID++;
      if (node.childID >= 0) continue;
      if (node.valueRange.upper < node.valueRange.lower) {
        std::cout << "WARNING: got a volumeProxy with zero elements here !?" << std::endl;
        continue;
      }
      result.push_back(VolumeProxy{ node.domain, node.valueRange });
    }
    return result;
  }
  
  void UMeshBrick::writeUnvaryingData(const std::string &fileName) const
  {
    std::cout << " .... writing un-varying brick data to " << (fileName) << std::endl;
    std::ofstream out(fileName,std::ios::binary);
    write(out,domain);

    std::cout << " .... writing umesh to " << (fileName+".umesh") << std::endl;
    umesh->finalize();
    umesh->saveTo(fileName+".umesh");
  }
  
  void UMeshBrick::writeTimeStep(const std::string &fileName) const
  {
    std::cout << " .... writing scalars to " << fileName << std::endl;
    std::ofstream out(fileName,std::ios::binary);
    write(out,umesh->perVertex->values);
    write(out,umesh->perVertex->valueRange);
  }
  
  /*! load all of the time-step and variabel independent data */
  void UMeshBrick::loadUnvaryingData(const std::string &fileName) 
  {
    this->umesh = umesh::UMesh::loadFrom(fileName+".umesh");
    
    std::ifstream in(fileName,std::ios::binary);
    read(in,domain);
  }
    
  /*! load a given time step and variable's worth of voxels from given file name */
  void UMeshBrick::loadTimeStep(const std::string &fileName) 
  {
    std::ifstream in(fileName,std::ios::binary);
    read(in,umesh->perVertex->values);
    read(in,umesh->perVertex->valueRange);
    PRINT(umesh->perVertex->valueRange);
  }
    
}
