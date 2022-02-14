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
#include <sstream>

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

  inline __device__ void addToFB(vec3f *tgt, vec3f addtl)
  {
    atomicAdd(&tgt->x,addtl.x);
    atomicAdd(&tgt->y,addtl.y);
    atomicAdd(&tgt->z,addtl.z);
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
    ray.pixelID  = pixelID.x + globals.fbSize.x*pixelID.y;
    ray.isShadow = false;
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
      // localFB.resize(newSize.x*newSize.y);
      // globals.fbPointer = localFB.get();
      accumBuffer.resize(newSize.x*newSize.y);
      globals.accumBuffer = accumBuffer.get();
      globals.fbSize    = newSize;

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
  void clipRay(box3f box, Ray ray, float &t_min, float &t_max)
  {
    vec3f t_lo = (box.lower - ray.origin) * rcp(fixDir(from_half(ray.direction)));
    vec3f t_hi = (box.upper - ray.origin) * rcp(fixDir(from_half(ray.direction)));
    
    vec3f t_nr = min(t_lo,t_hi);
    vec3f t_fr = max(t_lo,t_hi);

    float t0 = max(0.f,reduce_max(t_nr));
    float t1 = reduce_min(t_fr);

    t_min = t0;
    t_max = t1;
  }
  
  inline __device__
  int computeInitialRank(const Globals &globals, Ray ray, bool dbg = false)
  {
    int closest = -1;
    float t_min = CUDART_INF;
    for (int i=0;i<globals.numWorkers;i++) {
      if (boxTest(globals.rankBoxes[i],ray,t_min)) {
        closest = i;
      }
    }
    return closest;
  }

  __global__ void generatePrimaryWave(Globals globals)
  {
    int ix = threadIdx.x + blockIdx.x*blockDim.x;
    int iy = threadIdx.y + blockIdx.y*blockDim.y;
    if (ix >= globals.fbSize.x) return;
    if (iy >= globals.fbSize.y) return;

    int myRank = globals.myRank;
    Ray ray    = generateRay(globals,vec2i(ix,iy),vec2f(.5f));
    ray.dbg    = (vec2i(ix,iy) == globals.fbSize/2);
    int dest   = computeInitialRank(globals,ray);

    if (dest < 0) {
      /* "nobody" owns this pixel, set to background on rank 0 */
      if (myRank == 0) {
        globals.accumBuffer[ray.pixelID] += vec3f(.5f,.5f,.5f);
      }
      return;
    }
    if (dest != myRank) {
      /* somebody else owns this pixel; we don't do anything */
      return;
    }
    int queuePos = atomicAdd(&globals.perRankSendCounts[myRank],1);
    globals.rayQueueIn[queuePos] = ray;
  }
    
  __global__ void writeLocalFB(small_vec3f *localFB, Globals globals)
  {
    int ix = threadIdx.x + blockIdx.x*blockDim.x;
    int iy = threadIdx.y + blockIdx.y*blockDim.y;
    if (ix >= globals.fbSize.x) return;
    if (iy >= globals.fbSize.y) return;

    int i = ix + iy * globals.fbSize.x;
    
    localFB[i] = to_half(globals.accumBuffer[i] * (1.f/(globals.sampleID+1)));
  }
    
  void VopatRenderer::renderLocal()
  {
    globals.sampleID = accumID;
    
    vec2i blockSize(16);
    vec2i numBlocks = divRoundUp(islandFbSize,blockSize);
    perRankSendCounts.bzero();
    if (numBlocks != vec2i(0))
      generatePrimaryWave<<<numBlocks,blockSize>>>(globals);
    CUDA_SYNC_CHECK();
    host_sendCounts = perRankSendCounts.download();
    numRaysInQueue = host_sendCounts[myRank()];

// #if 1
//     fflush(0);
//     comm->worker.withinIsland->barrier();
//     printf("(%i) init num rays in q %i color %f %f %f\n",myRank(),numRaysInQueue,
//            randomColor(myRank()).x,
//            randomColor(myRank()).y,
//            randomColor(myRank()).z
//            );
//     comm->worker.withinIsland->barrier();
//     fflush(0);
// #endif

    
    CUDA_SYNC_CHECK();
    while (true) {
      perRankSendCounts.bzero();
      CUDA_SYNC_CHECK();
      traceRaysLocally();
      CUDA_SYNC_CHECK();
      createSendQueue();
      CUDA_SYNC_CHECK();
      int numRaysExchanged = exchangeRays();
      if (numRaysExchanged == 0)
        break;
    }

    CUDA_SYNC_CHECK();
    if (numBlocks != vec2i(0))
      writeLocalFB<<<numBlocks,blockSize>>>(localFB.get(),globals);
    CUDA_SYNC_CHECK();

// #if 1
//     fflush(0);
//     comm->worker.withinIsland->barrier();
//     fflush(0);
// #endif
  }
  
  void VopatRenderer::screenShot()
  {
    std::string fileName = Renderer::screenShotFileName;
    std::vector<uint32_t> pixels;
    if (isMaster()) {
      fileName = fileName + "_master.png";
      pixels = masterFB.download();
      for (int iy=0;iy<fbSize.y/2;iy++) {
        uint32_t *top = pixels.data() + iy * fbSize.x;
        uint32_t *bot = pixels.data() + (fbSize.y-1-iy) * fbSize.x;
        for (int ix=0;ix<fbSize.x;ix++)
          std::swap(top[ix],bot[ix]);
      }
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

  inline __device__ int computeNextNode(const Globals &globals,
                                        const Ray &ray,
                                        const float t_exit)
  {
    vec3f P = ray.origin + from_half(ray.direction) * (t_exit * (1.f+1e-3f));
    // if (ray.dbg)
    //   printf("(%i) next query P %f %f %f\n",
    //          globals.myRank,P.x,P.y,P.z);
    for (int i=0;i<globals.numWorkers;i++)
      if (i != globals.myRank && globals.rankBoxes[i].contains(P))
        return i;
    return -1;
  }
                                     
  __global__ void doTraceRaysLocally(Globals globals,
                                     int numRays)
  {
    int tid = threadIdx.x+blockIdx.x*blockDim.x;
    if (tid >= numRays) return;

    Ray ray = globals.rayQueueIn[tid];
    const box3f myBox = globals.rankBoxes[globals.myRank];
    float t0, t1;
    clipRay(myBox,ray,t0,t1);

    vec3f throughput = from_half(ray.throughput);
    throughput *= randomColor(globals.myRank);
    ray.throughput = to_half(throughput);

    int nextNode = computeNextNode(globals,ray,t1);

    if (nextNode == -1) {
      // path exits volume - deposit to image
      addToFB(&globals.accumBuffer[ray.pixelID],throughput);
      globals.rayNextNode[tid] = -1;
    } else {
      // ray has another node to go to - add to queue
      atomicAdd(&globals.perRankSendCounts[nextNode],1);
      globals.rayQueueIn[tid]  = ray;
      globals.rayNextNode[tid] = nextNode;
    }
  }
  
  void VopatRenderer::traceRaysLocally()
  {
    int blockSize = 1024;
    int numBlocks = divRoundUp(numRaysInQueue,blockSize);
    if (numBlocks)
      doTraceRaysLocally<<<numBlocks,blockSize>>>
        (globals,numRaysInQueue);
    CUDA_SYNC_CHECK();
  }

  __global__ void computeCompactionOffsets(Globals globals)
  {
    if (threadIdx.x != 0) return;
    int ofs = 0;
    for (int i=0;i<globals.numWorkers;i++) {
      globals.perRankSendOffsets[i] = ofs;
      ofs += globals.perRankSendCounts[i];
    }
  }
  
  __global__ void compactRays(Globals globals, int numRays)
  {
    int tid = threadIdx.x+blockIdx.x*blockDim.x;
    if (tid >= numRays) return;

    int dest = globals.rayNextNode[tid];
    if (dest < 0) return;

    int slot = atomicAdd(&globals.perRankSendOffsets[dest],1);
    globals.rayQueueOut[slot] = globals.rayQueueIn[tid];
  }
  
  void VopatRenderer::createSendQueue()
  {
    computeCompactionOffsets<<<1,1>>>
      (globals);
    CUDA_SYNC_CHECK();
    int blockSize = 1024;
    int numBlocks = divRoundUp(numRaysInQueue,blockSize);
    if (numBlocks)
      compactRays<<<numBlocks,blockSize>>>
        (globals,numRaysInQueue);
    CUDA_SYNC_CHECK();
  }

  int  VopatRenderer::exchangeRays()
  {
    host_sendCounts = perRankSendCounts.download();
    const int numWorkers = globals.numWorkers;
    const int myRank = this->myRank();
    auto island = comm->worker.withinIsland;

    /* iw - TODO: change this to use alltoall instead of allgaher;
       with allgather each rank received N*N elements, which describes
       what every node is sending to any other node ... but it would
       actually only need to know how many _it_ receives */
    std::vector<int> allRankSendCounts(numWorkers*numWorkers);
    island->allGather(allRankSendCounts,host_sendCounts);

    // ------------------------------------------------------------------
    // compute SEND counts and offsets
    // ------------------------------------------------------------------
    std::vector<int> sendByteOffsets(numWorkers);
    std::vector<int> sendByteCounts(numWorkers);
    size_t ofs = 0;
    for (int i=0;i<numWorkers;i++) {
      int to_i = allRankSendCounts[myRank*numWorkers+i];
      sendByteOffsets[i] = ofs;
      sendByteCounts[i]  = to_i*sizeof(Ray);
      ofs += sendByteCounts[i];
    }

    // ------------------------------------------------------------------
    // compute RECEIVE counts and offsets
    // ------------------------------------------------------------------
    std::vector<int> recvByteOffsets(numWorkers);
    std::vector<int> recvByteCounts(numWorkers);
    ofs = 0;
    int numReceived = 0;
    for (int i=0;i<numWorkers;i++) {
      int from_i = allRankSendCounts[i*numWorkers+myRank];
      recvByteOffsets[i] = ofs;
      recvByteCounts[i]  = from_i*sizeof(Ray);
      ofs += recvByteCounts[i];
      numReceived += from_i;
    }

    
    // ------------------------------------------------------------------
    // exeute the all2all
    // ------------------------------------------------------------------
    island->allToAll(rayQueueOut.get(),
                     sendByteCounts.data(),
                     sendByteOffsets.data(),
                     rayQueueIn.get(),
                     recvByteCounts.data(),
                     recvByteOffsets.data());
    CUDA_SYNC_CHECK();
    numRaysInQueue = numReceived;

    
    // ------------------------------------------------------------------
    // return how many we've exchanged ACROSS ALL ranks
    // ------------------------------------------------------------------
    int sumAllSends = 0;
    for (auto i : allRankSendCounts) sumAllSends += i;
    // printf("(%i) ----- sum all sends %i\n",myRank,sumAllSends);


    

    // for (int r=0;r<numWorkers;r++) {
    //   comm->worker.withinIsland->barrier();
    //   if (r == globals.myRank) {
    //     std::cout << "(" << r << ") IN:  ";
    //     for (int i=0;i<numWorkers;i++)
    //       std::cout << (recvByteCounts[i]/sizeof(Ray)) << " ";
    //     std::cout << std::endl;
    //     std::cout << "(" << r << ") OUT: ";
    //     for (int i=0;i<numWorkers;i++)
    //       std::cout << (sendByteCounts[i]/sizeof(Ray)) << " ";
    //     std::cout << std::endl;
    //     std::cout << "(" << r << ") num in queue " << numRaysInQueue << std::endl;
    //     if (r == 0)
    //       std::cout << "-------------------------------------------------------" << std::endl;
    //     fflush(0);
    //   }
    //   comm->worker.withinIsland->barrier();
    // }



    return sumAllSends;
  }
  
}
