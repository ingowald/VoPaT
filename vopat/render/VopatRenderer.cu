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

#include "vopat/render/VopatRenderer.h"
#include "3rdParty/stb_image//stb/stb_image_write.h"
#include "3rdParty/stb_image//stb/stb_image.h"
#include "owl/owl_device.h"

namespace vopat {

  inline __both__ uint32_t make_8bit(const float f)
  {
    return min(255,max(0,int(f*256.f)));
  }

  inline __both__ uint32_t make_rgba(const vec3f color)
  {
    return
      (make_8bit(color.x) << 0) +
      (make_8bit(color.y) << 8) +
      (make_8bit(color.z) << 16) +
      (0xffU << 24);
  }
  inline __both__ uint32_t make_rgba(const vec4f color)
  {
    return
      (make_8bit(color.x) << 0) +
      (make_8bit(color.y) << 8) +
      (make_8bit(color.z) << 16) +
      (make_8bit(color.w) << 24);
  }

  VopatRenderer::VopatRenderer(CommBackend *comm,
                               Model::SP model,
                               int numSPP)
    : AddWorkersRenderer(comm)
  {
    if (isMaster()) {
    } else {
      const int numWorkers = comm->numWorkers();

      globals.myRank   = myRank();
      globals.numWorkers = numWorkers;
      perRankSendCounts.resize(numWorkers);
      perRankSendOffsets.resize(numWorkers);
      globals.perRankSendOffsets = perRankSendOffsets.get();
      globals.perRankSendCounts = perRankSendCounts.get();

      // ------------------------------------------------------------------
      // upload per-rank boxes
      // ------------------------------------------------------------------
      std::vector<box3f> hostRankBoxes;
      for (auto brick : model->bricks)
        hostRankBoxes.push_back(brick->spaceRange);
      if (hostRankBoxes.size() != numWorkers)
        throw std::runtime_error("invalid rank boxes!?");
      rankBoxes.upload(hostRankBoxes);
      globals.rankBoxes = rankBoxes.get();
    }
  }


  inline __device__ Ray generateRay(const Globals &globals,
                                    vec2i pixelID,
                                    vec2f pixelPos)
  {
    Ray ray;
    ray.pixelID = pixelID.x + globals.fbSize.x*pixelID.y;
    ray.origin = globals.camera.lens_00;
    vec3f dir
      = globals.camera.dir_00
      + globals.camera.dir_du * (pixelID.x+pixelPos.x)
      + globals.camera.dir_dv * (pixelID.y+pixelPos.y);
    ray.direction = to_half(normalize(dir));
    ray.throughput = to_half(vec3f(1.f));
    return ray;
  }

  void VopatRenderer::resizeFrameBuffer(const vec2i &newSize)
  {
    AddWorkersRenderer::resizeFrameBuffer(newSize);
    if (isMaster()) {
    } else {
      localFB.resize(newSize.x*newSize.y);
      globals.fbPointer = localFB.get();
      globals.fbSize    = fbSize;

      rayQueueIn.resize(fbSize.x*fbSize.y);
      globals.rayQueueIn = rayQueueIn.get();

      rayQueueOut.resize(fbSize.x*fbSize.y);
      globals.rayQueueOut = rayQueueOut.get();

      rayNextNode.resize(fbSize.x*fbSize.y);
      globals.rayNextNode = rayNextNode.get();
    }
  }
  
  void VopatRenderer::setCamera(const vec3f &from,
                                const vec3f &at,
                                const vec3f &up,
                                const float fovy)
  {
    AddWorkersRenderer::setCamera(from,at,up,fovy);
    globals.camera = AddWorkersRenderer::camera;
  }

  inline __device__
  float fixDir(float f) { return (f==0.f)?1e-8f:f; }
  
  inline __device__
  vec3f fixDir(vec3f v)
  { return {fixDir(v.x),fixDir(v.y),fixDir(v.z)}; }
  
  inline __device__
  bool boxTest(box3f box, Ray ray, float &t_min)
  {
    vec3f t_lo = (box.lower - ray.origin) * rcp(fixDir(from_half(ray.direction)));
    vec3f t_hi = (box.upper - ray.origin) * rcp(fixDir(from_half(ray.direction)));
    
    vec3f t_nr = min(t_lo,t_hi);
    vec3f t_fr = max(t_lo,t_hi);

    float t0 = max(0.f,reduce_max(t_nr));
    float t1 = min(t_min,reduce_min(t_fr));

    if (t0 >= t1) return false;
    t_min = t0;
    return true;
  }
  
  inline __device__
  int computeInitialRank(const Globals &globals, Ray ray, bool dbg = false)
  {
    int closest = -1;
    float t_min = CUDART_INF;
    // if (dbg) printf("init ray %f %f %f dir %f %f %f\n",
    //                ray.origin.x,
    //                ray.origin.y,
    //                ray.origin.z,
    //                from_half(ray.direction.x),
    //                from_half(ray.direction.y),
    //                from_half(ray.direction.z)
    //                );
    for (int i=0;i<globals.numWorkers;i++) {
      if (boxTest(globals.rankBoxes[i],ray,t_min)) {
        // if (dbg) printf("(%i) new closest %i %f\n",
        //                 globals.myRank,i,t_min);
        closest = i;
      }
    }
    return closest;
  }

  __global__ void renderFrame(Globals globals)
  {
    int ix = threadIdx.x + blockIdx.x*blockDim.x;
    int iy = threadIdx.y + blockIdx.y*blockDim.y;

    int myRank = globals.myRank;
    Ray ray    = generateRay(globals,vec2i(ix,iy),vec2f(.5f));
    int dest   = computeInitialRank(globals,ray);

    if (dest < 0) {
      /* "nobody" owns this pixel, set to background on rank 0 */
      if (myRank == 0) {
        globals.fbPointer[ray.pixelID] = to_half(vec3f(.5f,.5f,.5f));
      }
      return;
    }
    if (dest != myRank) {
      /* somebody else owns this pixel; we don't do anything */
      return;
    }
    int queuePos = atomicAdd(&globals.perRankSendCounts[myRank],1);
    globals.rayQueueIn[queuePos] = ray;
    globals.fbPointer[ray.pixelID] = to_half(randomColor(myRank));
  }
    
  void VopatRenderer::renderLocal()
  {
    perRankSendCounts.bzero();
    localFB.bzero();
    
    vec2i blockSize(16);
    vec2i numBlocks = divRoundUp(islandFbSize,blockSize);
    renderFrame<<<numBlocks,blockSize>>>(globals);
    CUDA_SYNC_CHECK();
  }
  
  void VopatRenderer::screenShot()
  {
    std::string fileName = Renderer::screenShotFileName;
    std::vector<uint32_t> pixels;
    if (isMaster()) {
      fileName = fileName + "_master.png";
      pixels = masterFB.download();
    } else {
      char suff[100];
      sprintf(suff,"_island%03i_rank%05i.png",
              comm->worker.islandIdx,comm->worker.withinIsland->rank);
      fileName = fileName + suff;
      
      std::vector<small_vec3f> hostFB;
      hostFB = localFB.download();
      for (int y=0;y<fbSize.y;y++) {
        const small_vec3f *line = hostFB.data() + (fbSize.y-1-y)*fbSize.x;
        for (int x=0;x<fbSize.x;x++) {
          vec3f col = from_half(line[x]);
          pixels.push_back(make_rgba(col) | (0xffu << 24));
        }
      }
    }
    
    stbi_write_png(fileName.c_str(),fbSize.x,fbSize.y,4,
                   pixels.data(),fbSize.x*sizeof(uint32_t));
    std::cout << "screenshot saved in '" << fileName << "'" << std::endl;

  }

  
}
