// ======================================================================== //
// Copyright 2018-2021 Ingo Wald                                            //
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

/* computes an *object* space partitioing of a mesh into bricks (up
   until either max numbre of bricks is reahed, or bricks' sizes fall
   below user-specified max size). stores resulting bricks in one file
   per brick, plus one file containing the bounding box for each
   brick. brick bounds may overlap, but no prim should ever go into
   more than one brick */

#include "Model.h"
#include "model/UMeshModel.h"
#include "umesh/UMesh.h"
#include "umesh/check.h"
#include "umesh/extractShellFaces.h"
#include "umesh/tetrahedralize.h"
#include "umesh/io/ugrid32.h"
#include "umesh/io/UMesh.h"
#include "umesh/RemeshHelper.h"
#include <queue>

#define PARALLEL_REINDEXING 1

namespace umesh {
  using namespace vopat;
  
  void usage(const std::string &error = "")
  {
    if (error != "")
      std::cout << "Fatal error: " << error << std::endl << std::endl;
    std::cout << "./mpmPartitionObjectSpace <in.umesh> <args>" << std::endl;
    std::cout << "w/ Args: " << std::endl;
    std::cout << "-o <outFileName.mm>\n\tbase path for all output files (there will be multiple)" << std::endl;
    std::cout << "-n|-mb|--max-bricks <N>\n\tmax number of bricks to create" << std::endl;
    std::cout << "-lt|--leaf-threshold <N>\n\tnum prims at which we make a leaf" << std::endl;
    std::cout << std::endl;
    std::cout << "generated files are:" << std::endl;
    std::cout << "<baseName>.bricks : one box3f for each generated brick" << std::endl;
    std::cout << "<baseName>_%05d.umesh : the extracted umeshes for each brick" << std::endl;
    exit( error != "");
  }

  inline bool noDuplicates(const Triangle &tri)
  {
    return
      tri.x != tri.y &&
      tri.x != tri.z &&
      tri.y != tri.z;
  }
  
  inline bool noDuplicates(const Tet &tet)
  {
    return
      tet.x != tet.y &&
      tet.x != tet.z &&
      tet.x != tet.w &&
      tet.y != tet.z &&
      tet.y != tet.w &&
      tet.z != tet.w;
  }

  struct Brick {
    std::vector<UMesh::PrimRef> prims;
    box3f bounds;
    box3f centBounds;

    inline float weight() const { return prims.size(); }
  };


#if 1
  void splitAt(int dim,
               float pos,
               UMesh::SP mesh,
               Brick *in,
               Brick *out[2])
  {
    std::cout << "splitting brick\tw/ bounds " << in->bounds << " cent " << in->centBounds << std::endl;
    std::cout << "splitting at " << char('x'+dim) << "=" << pos << std::endl;

    out[0] = new Brick;
    out[1] = new Brick;
#if 0
    std::mutex mutex[2];
    parallel_for_blocked
      (0ull,in->prims.size(),128*1024ull,
       [&](size_t begin,size_t end){
        std::vector<UMesh::PrimRef> prims[2];
        box3f bounds[2];
        box3f centBounds[2];
        for (size_t i=begin;i<end;i++) {
          auto prim = in->prims[i];
          const box3f pb = mesh->getBounds(prim);
          int side = (pb.center()[dim] < pos) ? 0 : 1;
          prims[side].push_back(prim);
          bounds[side].extend(pb);
          centBounds[side].extend(pb.center());
        }
        for (int side=0;side<2;side++) {
          std::lock_guard<std::mutex> lock(mutex[side]);
          out[side]->bounds.extend(bounds[side]);
          out[side]->centBounds.extend(centBounds[side]);
          for (auto prim : prims[side])
            out[side]->prims.push_back(prim);
        }
      });
#else
    for (auto prim : in->prims) {
      const box3f pb = mesh->getBounds(prim);
      int side = (pb.center()[dim] < pos) ? 0 : 1;
      out[side]->prims.push_back(prim);
      out[side]->bounds.extend(pb);
      out[side]->centBounds.extend(pb.center());
    }
#endif
    std::cout << "done splitting " << prettyNumber(in->prims.size()) << " prims\tw/ bounds " << in->bounds << std::endl;
    std::cout << "into L = " << prettyNumber(out[0]->prims.size()) << " prims\tw/ bounds " << out[0]->bounds << std::endl;
    std::cout << " and R = " << prettyNumber(out[1]->prims.size()) << " prims\tw/ bounds " << out[1]->bounds << std::endl;
  }
  
  void split(UMesh::SP mesh,
             Brick *in,
             Brick *out[2])
  {
    if (in->centBounds.lower == in->centBounds.upper)
      throw std::runtime_error("can't split this any more ...");

    std::cout << "#### test splitting ####" << std::endl;
    std::mutex mutex;
    float bestRatio = -std::numeric_limits<float>::infinity();
    float bestPos;
    int bestDim;
    const int numPlanes = 7;
    // for (int dim=0;dim<3;dim++)
    parallel_for
      (3,
       [&](int dim){
         parallel_for
           (numPlanes,
            [&](int plane) {
              // for (int plane=0;plane<numPlanes;plane++) {
              float f = (plane+1.f)/(numPlanes+1.f);
              float pos = (1.f-f)*in->centBounds.lower[dim]+f*in->centBounds.upper[dim];
              Brick *tmp_out[2];
              splitAt(dim,pos,mesh,in,tmp_out);
              
              float weight0 = tmp_out[0]->weight();
              float weight1 = tmp_out[1]->weight();
              float area0 = area((const owl::common::box3f&)tmp_out[0]->bounds);
              float area1 = area((const owl::common::box3f&)tmp_out[1]->bounds);
              float areaParent = area((const owl::common::box3f&)in->bounds);

              float weight_ratio = (max(weight0,weight1)-min(weight0,weight1))/(weight0+weight1);
              float area_ratio = max(area0,area1)/areaParent;
              float ratio = -((weight_ratio+.1f)*area_ratio);
              // float ratio = min(weight0,weight1) / (weight0+weight1);
              {
                std::lock_guard<std::mutex> lock(mutex);
                if (ratio >= bestRatio) {
                  std::cout << "*** NEW BEST, ratio is " << ratio << std::endl;
                  bestDim = dim;
                  bestPos = pos;
                  bestRatio = ratio;
                }
              }
              delete tmp_out[0];
              delete tmp_out[1];
            });
       });
    std::cout << "=== found BEST split at " << ('x'+bestDim) << " = " << bestPos << std::endl;
    splitAt(bestDim,bestPos,mesh,in,out);
    std::cout << "done *actual* splitting " << prettyNumber(in->prims.size()) << " prims\tw/ bounds " << in->bounds << std::endl;
    std::cout << "into L = " << prettyNumber(out[0]->prims.size()) << " prims\tw/ bounds " << out[0]->bounds << std::endl;
    std::cout << " and R = " << prettyNumber(out[1]->prims.size()) << " prims\tw/ bounds " << out[1]->bounds << std::endl;
  }
#else
  void split(UMesh::SP mesh,
             Brick *in,
             Brick *out[2])
  {
    // PING;
    // PRINT(in->prims.size());
    // PRINT(in->bounds);
    // PRINT(in->centBounds);

    if (in->centBounds.lower == in->centBounds.upper)
      throw std::runtime_error("can't split this any more ...");

    int dim = arg_max(in->centBounds.size());
    float pos = in->centBounds.center()[dim];
    std::cout << "splitting brick\tw/ bounds " << in->bounds << " cent " << in->centBounds << std::endl;
    std::cout << "splitting at " << char('x'+dim) << "=" << pos << std::endl;

    out[0] = new Brick;
    out[1] = new Brick;
    for (auto prim : in->prims) {
      const box3f pb = mesh->getBounds(prim);
      int side = (pb.center()[dim] < pos) ? 0 : 1;
      out[side]->prims.push_back(prim);
      out[side]->bounds.extend(pb);
      out[side]->centBounds.extend(pb.center());
    }
    std::cout << "done splitting " << prettyNumber(in->prims.size()) << " prims\tw/ bounds " << in->bounds << std::endl;
    std::cout << "into L = " << prettyNumber(out[0]->prims.size()) << " prims\tw/ bounds " << out[0]->bounds << std::endl;
    std::cout << " and R = " << prettyNumber(out[1]->prims.size()) << " prims\tw/ bounds " << out[1]->bounds << std::endl;
  }
#endif
  
  void createInitialBrick(std::priority_queue<std::pair<int,Brick *>> &bricks,
                          UMesh::SP in)
  {
    Brick *brick = new Brick;
    in->createVolumePrimRefs(brick->prims);
    for (auto prim : brick->prims) {
      const box3f pb = in->getBounds(prim);
      brick->bounds.extend(pb);
      brick->centBounds.extend(pb.center());
    }
           
    bricks.push({(int)brick->prims.size(),brick});
  }

  // void writeBrick(UMesh::SP in,
  //                 const std::string &fileBase,
  //                 Brick *brick)
  // {
  //   UMesh::SP out = std::make_shared<UMesh>();
  //   RemeshHelper indexer(*out);
  //   for (auto prim : brick->prims) 
  //     indexer.add(in,prim);
  //   const std::string fileName = fileBase+".umesh";
  //   std::cout << "saving out " << fileName
  //             << " w/ " << prettyNumber(out->size()) << " prims" << std::endl;
  //   io::saveBinaryUMesh(fileName,out);
  // }
  
  extern "C" int main(int ac, char **av)
  {
    std::string inFileName;
    std::string outFileName;
    int leafThreshold = 1<<30;
    int maxBricks = 1<<30;

    for (int i=1;i<ac;i++) {
      const std::string arg = av[i];
      if (arg == "-o")
        outFileName = av[++i];
      else if (arg == "-lt" || arg == "--leaf-threshold")
        leafThreshold = atoi(av[++i]);
      else if (arg == "-n" || arg == "-mb" || arg == "--max-bricks") {
        maxBricks = atoi(av[++i]);
        leafThreshold = 1;
      } else if (arg[0] != '-')
        inFileName = arg;
      else
        usage("unknown arg "+arg);
    }
    
    if (outFileName == "") usage("no output file name specified");
    if (inFileName == "") usage("no input file name specified");
    if (leafThreshold == 1<<30 && maxBricks == 1<<30)
      usage("neither leaf threshold nor max bricks specified");
    std::cout << "loading umesh from " << inFileName << std::endl;
    UMesh::SP in = io::loadBinaryUMesh(inFileName);
    std::cout << "done loading, found " << in->toString() << std::endl;
    if (!in->perVertex)
      throw std::runtime_error("can currently only do per-vertex meshes");
    sanityCheck(in);
    assert(in->numVolumeElements() > 0);
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
      bricks.push({(int)half[0]->prims.size(),half[0]});
      bricks.push({(int)half[1]->prims.size(),half[1]});
      delete biggest.second;
    }

    std::cout << "done splitting, starting to make bricks ..." << std::endl;
    char ext[20];
    std::vector<box3f> brickBounds;
    // Model::SP model = std::make_shared<Model>();
    std::vector<Brick *> brickList;
    while (!bricks.empty()) {
      Brick *brick = bricks.top().second;
      bricks.pop();
      brickList.push_back(brick);
    }
    // model->parts.resize(brickList.size());
    
    UMeshModel::SP vopatModel = UMeshModel::create();
    vopatModel->numBricks = brickList.size();
    parallel_for
      (brickList.size(),
       [&](size_t brickID)
       {
         // for (int brickID=0;!bricks.empty();brickID++) {
         // for (int brickID=0;!bricks.empty();brickID++) {
         std::cout << "making brick ID " << brickID << std::endl;
         Brick *brick = brickList[brickID];
         assert(!brick->prims.empty());

         std::cout << "[" << brickID << "] re-indexing all prims" << std::endl;
         UMesh::SP out = std::make_shared<UMesh>();
#if PARALLEL_REINDEXING
         out->vertices = in->vertices;
         out->perVertex = std::make_shared<Attribute>();
         out->perVertex->name = in->perVertex->name;
         out->perVertex->values = in->perVertex->values;
         out->vertexTags.resize(out->vertices.size());
         for (size_t i=0;i<out->vertices.size();i++)
           out->vertexTags[i] = i;
         for (auto primRef : brick->prims) {
           switch (primRef.type) {
           case UMesh::TRI: {
             auto prim = in->triangles[primRef.ID];
             if (noDuplicates(prim))
               out->triangles.push_back(prim);
           } break;
           case UMesh::QUAD: {
             auto prim = in->quads[primRef.ID];
             out->quads.push_back(prim);
           } break;
           case UMesh::TET: {
             auto prim = in->tets[primRef.ID];
             if (noDuplicates(prim))
               out->tets.push_back(prim);
           } break;
           case UMesh::PYR: {
             auto prim = in->pyrs[primRef.ID];
             out->pyrs.push_back(prim);
           } break;
           case UMesh::WEDGE: {
             auto prim = in->wedges[primRef.ID];
             out->wedges.push_back(prim);
           } break;
           case UMesh::HEX: {
             auto prim = in->hexes[primRef.ID];
             out->hexes.push_back(prim);
           } break;
           default:
             throw std::runtime_error("un-implemented prim type?");
           }
         }
         // parallelReIndexing(out);
         removeUnusedVertices(out);
#else
         RemeshHelper indexer(*out,true);
         for (auto prim : brick->prims) 
           indexer.add(in,prim);
#endif


#if 1
         UMeshBrick::SP vopatBrick = UMeshBrick::create(brickID);
         vopatBrick->umesh = out;
         
         box3f brickBounds = out->getBounds();
         PING; PRINT(out->tets.size()); PRINT(brickBounds);
         range1f brickRange = out->getValueRange();
         vopatBrick->domain = (const vopat::box3f&)brickBounds;
         PRINT(vopatBrick->domain);
         
         vopatModel->domain.extend((const vopat::box3f&)brickBounds);
         vopatModel->valueRange.extend((const vopat::range1f&)brickRange);
         // umb->umesh = out;
         // model->bricks.push_back(umb);

         vopatBrick->writeUnvaryingData(vopat::Model::canonicalBrickFileName(outFileName,brickID));
         vopatBrick->writeTimeStep(vopat::Model::canonicalTimeStepFileName(outFileName,brickID,
                                                                           "unknown",0));
#else
         std::cout << "got umesh brick " << out->toString() << std::endl;
         char suffix[100];
         sprintf(suffix,"_part-%04i.umesh",(int)brickID);
         out->saveTo(outFileName+suffix);
#endif
         // Model::Part::SP part = std::make_shared<Model::Part>();
         // part->mesh = out;
         // std::cout << "[" << brickID << "] got mesh " << part->mesh->toString() << std::endl;
         // // std::cout << "sanity-checking the mesh generated fo this part..." << std::endl;
         // sanityCheck(part->mesh);

         // // part->mesh = tetrahedralize_maintainFlatElements(part->mesh);
         // // std::cout << "[" << brickID << "] extracting shell faces" << std::endl;
         // // part->shell = extractShellFaces(part->mesh,1);
         // // sanityCheck(part->shell,CHECK_FLAG_MESH_IS_SURFACE);
      
         // model->parts[brickID] = part;
         // // {
         //   std::lock_guard<std::mutex> lock(modelMutex);
         //   model->parts.push_back(part);
         // }
         delete brick;
       });
    vopatModel->save(Model::canonicalMasterFileName(outFileName));
    std::cout << "done building model, saving it..." << std::endl;
    // model->save(outFileName);
    std::cout << "done wrting mp-model... done all" << std::endl;
  }
  
} // ::umesh
