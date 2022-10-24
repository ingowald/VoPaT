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

/* computes a *spatial* partitioning of a mesh into sub-bricks (up
   until either max numbre of bricks is reahed, or bricks' sizes fall
   below user-specified max size). stores resulting bricks in one file
   per brick, plus one file containing one box per brick that
   specifies the "domain" of that brick (iw, which region of space it
   is responsible for. domains do not overlap. */

#include "umesh/io/ugrid32.h"
#include "umesh/io/UMesh.h"
#include "umesh/RemeshHelper.h"
#include <queue>
#include "model/UMeshModel.h"

namespace umesh {

  // bool primRefsOnly = false;
  
  void usage(const std::string &error = "")
  {
    if (error != "")
      std::cout << UMESH_TERMINAL_RED
                << "\n*********** Fatal error: " << error << " ***********" << std::endl
                << UMESH_TERMINAL_DEFAULT << std::endl;
    std::cout << "./umeshPartitionSpatially <in.umesh> <args>" << std::endl;
    std::cout << "w/ Args: " << std::endl;
    std::cout << "-o <baseName>\n\tbase path for all output files (there will be multiple)" << std::endl;
    std::cout << "-n|-mb|--max-bricks <N>\n\tmax number of bricks to create" << std::endl;
    std::cout << "-lt|--leaf-threshold <N>\n\tnum prims at which we make a leaf" << std::endl;
    std::cout << "-pro|--prim-refs-only\n\tdump _only_ the primrefs going into each brick, do not create the actual umeshes" << std::endl;
    std::cout << std::endl;
    std::cout << "generated files are:" << std::endl;
    std::cout << "<baseName>.bricks : one box3f for each generated brick" << std::endl;
    std::cout << "<baseName>_%05d.umesh : the extracted umeshes for each brick" << std::endl;
    exit( error != "");
  }

  struct Brick {
    std::vector<UMesh::PrimRef> prims;
    box3f domain;
  };

  void split(UMesh::SP mesh,
             Brick *in,
             Brick *out[2])
  {
    if (in->domain.lower == in->domain.upper)
      throw std::runtime_error("can't split this any more ...");

    int dim = arg_max(in->domain.size());
    float pos = in->domain.center()[dim];
    std::cout << "splitting brick\tw/ domain " << in->domain << std::endl;
    std::cout << "splitting at " << char('x'+dim) << "=" << pos << std::endl;

    Brick *l = out[0] = new Brick;
    Brick *r = out[1] = new Brick;
    l->domain = r->domain = in->domain;
    l->domain.upper[dim] = pos;
    r->domain.lower[dim] = pos;
    box3f lBounds, rBounds;

    for (auto prim : in->prims) {
      const box3f pb = mesh->getBounds(prim);
      if (pb.overlaps(l->domain)) {
        l->prims.push_back(prim);
        lBounds.extend(pb);
      }
      if (pb.overlaps(r->domain)) {
        r->prims.push_back(prim);
        rBounds.extend(pb);
      }
    }
    l->domain = intersection(l->domain,lBounds);
    r->domain = intersection(r->domain,rBounds);
    
    std::cout << "done splitting " << prettyNumber(in->prims.size()) << " prims\tw/ domain " << in->domain << std::endl;
    std::cout << "into L = " << prettyNumber(out[0]->prims.size()) << " prims\tw/ domain " << out[0]->domain << std::endl;
    std::cout << " and R = " << prettyNumber(out[1]->prims.size()) << " prims\tw/ domain " << out[1]->domain << std::endl;
  }
  
  void createInitialBrick(std::priority_queue<std::pair<int,Brick *>> &bricks,
                          UMesh::SP in)
  {
    Brick *brick = new Brick;
    in->createVolumePrimRefs(brick->prims);
    box3f bounds;
    for (auto prim : brick->prims) {
      const box3f pb = in->getBounds(prim);
      bounds.extend(pb);
    }
    brick->domain = bounds;
    bricks.push({(int)brick->prims.size(),brick});
  }

  void writeBrick(UMesh::SP in,
                  const std::string &outFileBase,
                  Brick *brick,
                  int brickID,
                  range1f &valueRange)
  {
    UMesh::SP out = std::make_shared<UMesh>();
    RemeshHelper indexer(*out);

    valueRange = range1f();
    for (auto pr : brick->prims)
      valueRange.extend(in->getValueRange(pr));
    
    // if (primRefsOnly) {
    //   const std::string fileName = fileBase+".primRefs";
    //   std::cout << "saving out " << fileName
    //             << " w/ " << prettyNumber(brick->prims.size()) << " primsRefs" << std::endl;
    //   std::ofstream out(fileName,std::ios::binary);
    //   io::writeVector(out,brick->prims);
    //   io::writeElement(out,valueRange);
    // } else {
    for (auto prim : brick->prims) 
      indexer.add(in,prim);
    // const std::string fileName
    //   = vopat::Model:://fileBase+".umesh";
    std::cout << "saving out brick #" << brickID << " -> " << out->toString() << std::endl;
    // io::saveBinaryUMesh(fileName,out);
    // }

    vopat::UMeshBrick::SP vopatBrick
      = vopat::UMeshBrick::create(brickID);
    (box3f&)vopatBrick->domain = brick->domain;
    vopatBrick->umesh = out;
    
    vopatBrick->writeUnvaryingData(vopat::Model::canonicalBrickFileName(outFileBase,brickID));
    vopatBrick->writeTimeStep(vopat::Model::canonicalTimeStepFileName(outFileBase,brickID,
                                                               "unknown",0));
  }
  
  extern "C" int main(int ac, char **av)
  {
    std::string inFileName;
    std::string outFileBase;
    int leafThreshold = 1<<30;
    int maxBricks = 1<<30;
    
    for (int i=1;i<ac;i++) {
      const std::string arg = av[i];
      if (arg == "-h")
        usage();
      else if (arg == "-o")
        outFileBase = av[++i];
      else if (arg == "-lt" || arg == "--leaf-threshold")
        leafThreshold = atoi(av[++i]);
      else if (arg == "-n" || arg == "-mb" || arg == "--max-bricks") {
        maxBricks = atoi(av[++i]);
        leafThreshold = 1;
      // else if (arg == "-pro" || arg == "--prim-refs-only")
      //   primRefsOnly = true;
      } else if (arg[0] != '-')
        inFileName = arg;
      else
        usage("unknown arg "+arg);
    }

    if (outFileBase == "") usage("no output file name specified");
    if (inFileName == "") usage("no input file name specified");
    if (leafThreshold == 1<<30 && maxBricks == 1<<30)
      usage("neither leaf threshold nor max bricks specified");

    std::cout << "loading umesh from " << inFileName << std::endl;
    UMesh::SP in = io::loadBinaryUMesh(inFileName);
    std::cout << "done loading, found " << in->toString() << std::endl;
    
    std::priority_queue<std::pair<int,Brick *>> bricks;
    createInitialBrick(bricks,in);
    
    while (bricks.size() < maxBricks) {
      auto biggest = bricks.top(); 
      std::cout << "########### currently having " << bricks.size()
                << " bricks, biggest of which has "
                << prettyNumber(biggest.first) << " prims" << std::endl;
      if (biggest.first < leafThreshold)
        break;
      bricks.pop();

      std::cout << "splitting..." << std::endl;
      Brick *half[2];
      split(in,biggest.second,half);
      for (int i=0;i<2;i++) {
        if (half[i]->prims.empty() ||
            reduce_min(half[i]->domain.size()) == 0.f) {
          std::cout << "WARNING: got invalid split, dropping!" << std::endl;
          delete half[i];
        }
        else
          bricks.push({ (int)half[i]->prims.size(),half[i]});
      }
      delete biggest.second;
    }

    vopat::UMeshModel::SP vopatModel = vopat::UMeshModel::create();
    vopatModel->numBricks = bricks.size();
    vopatModel->numTimeSteps = 1;
    
    std::vector<box3f> brickDomains;
    std::vector<range1f> valueRanges;
    for (int brickID=0;!bricks.empty();brickID++) {
      Brick *brick = bricks.top().second;
      bricks.pop();
      range1f valueRange;
      writeBrick(in,outFileBase,brick,brickID,valueRange);
      brickDomains.push_back(brick->domain);
      vopatModel->domain.extend((const vopat::box3f&)brick->domain);
      vopatModel->valueRange.extend((const vopat::range1f&)valueRange);
        // valueRanges.push_back(valueRange);
      delete brick;
    }

    std::cout << OWL_TERMINAL_BLUE << "saving model..." << OWL_TERMINAL_DEFAULT << std::endl;
    vopatModel->save(vopat::Model::canonicalMasterFileName(outFileBase));
  }
  
} // ::umesh
