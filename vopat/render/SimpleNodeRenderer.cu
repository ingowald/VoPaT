#include "owl/owl_device.h"
#include "vopat/render/VopatRenderer.h"
#include "owl/common/math/random.h"

namespace vopat {

  using Random = owl::common::LCG<4>;

  inline __device__ vec3f backgroundColor() { return .3f; }
  
  inline __device__ vec3f lightColor() { return 2.f*vec3f(.8f,1.f,.8f); }
  inline __device__ vec3f lightDirection()
  {
    return vec3f(1.f,.1f,.1f);
    // return (0.f,0.f,1.f);
  }





  
  struct SimpleNodeRenderer {
    struct OwnGlobals {
      box3f *rankBoxes;
      box3f  myRegion;
      float *myVoxels;
      vec3i  numVoxels;
    };
    
    struct Ray {
      struct {
        uint32_t    pixelID  : 29;
        uint32_t    dbg      :  1;
        uint32_t    crosshair:  1;
        uint32_t    isShadow :  1;
      };
      vec3f       origin;
      small_vec3f direction;
      small_vec3f throughput;
    };

    using VopatGlobals = typename VopatRenderer<SimpleNodeRenderer>::Globals;
    
    SimpleNodeRenderer(Model::SP model,
                       const std::string &baseFileName,
                       int myRank)
    {
      if (myRank < 0)
        return;
      
      // ------------------------------------------------------------------
      // upload per-rank boxes
      // ------------------------------------------------------------------
      std::vector<box3f> hostRankBoxes;
      for (auto brick : model->bricks)
        hostRankBoxes.push_back(brick->spaceRange);
      rankBoxes.upload(hostRankBoxes);
      globals.rankBoxes = rankBoxes.get();

      myBrick = model->bricks[myRank];
      const std::string fileName = Model::canonicalRankFileName(baseFileName,myRank);
      std::vector<float> loadedVoxels = myBrick->load(fileName);
      
      voxels.upload(loadedVoxels);
      globals.myVoxels  = voxels.get();
      globals.numVoxels = myBrick->numVoxels;//voxelRange.size();
      globals.myRegion  = myBrick->spaceRange;
    };

    /*! one box per rank, which rays can use to find neext rank to send to */
    CUDAArray<box3f> rankBoxes;
    OwnGlobals       globals;
    Brick::SP        myBrick;
    CUDAArray<float> voxels;

    void generatePrimaryWave(const VopatGlobals &globals);
    void traceLocally(const VopatGlobals &globals);

    static inline __device__
    int computeInitialRank(const VopatGlobals &vopat,
                           const OwnGlobals &globals,
                           Ray ray, bool dbg = false);
    
    static inline __device__
    int computeNextNode(const VopatGlobals &vopat,
                        const OwnGlobals &globals,
                        const Ray &ray,
                        const float t_exit);
    
    Model::SP model;
  };



  inline __device__
  float fixDir(float f) { return (f==0.f)?1e-6f:f; }
  
  inline __device__
  vec3f fixDir(vec3f v)
  { return {fixDir(v.x),fixDir(v.y),fixDir(v.z)}; }
  
  inline __device__
  bool boxTest(box3f box, SimpleNodeRenderer::Ray ray, float &t_min)
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
  bool boxTest(box3f box, SimpleNodeRenderer::Ray ray, float t0, float t1, float &t_min)
  {
    vec3f t_lo = (box.lower - ray.origin) * rcp(fixDir(from_half(ray.direction)));
    vec3f t_hi = (box.upper - ray.origin) * rcp(fixDir(from_half(ray.direction)));
    
    vec3f t_nr = min(t_lo,t_hi);
    vec3f t_fr = max(t_lo,t_hi);

    t0 = max(t0,reduce_max(t_nr));
    t1 = min(t1,reduce_min(t_fr));

    if (t0 >= t1) return false;
    t_min = t0;
    return true;
  }

  
  inline __device__
  void clipRay(box3f box, SimpleNodeRenderer::Ray ray, float &t_min, float &t_max)
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
  int SimpleNodeRenderer::computeNextNode(const VopatGlobals &vopat,
                                          const OwnGlobals &globals,
                                          const Ray &ray,
                                          const float t_exit)
  {
#if 1
    int closest = -1;
    float t_closest = CUDART_INF;
    for (int i=0;i<vopat.numWorkers;i++) {
      if (i == vopat.myRank) continue;

      float t_min = t_exit;
      float t_max = t_closest;
      if (!boxTest(globals.rankBoxes[i],ray,t_min,t_max,t_closest))
        continue;
      closest = i;
    }
    if (ray.dbg) printf("(%i) NEXT rank is %i\n",vopat.myRank,closest);
    return closest;
#else
      
    vec3f P = ray.origin + from_half(ray.direction) * (t_exit * (1.f+1e-5f));
    for (int i=0;i<vopat.numWorkers;i++)
      if (i != vopat.myRank && globals.rankBoxes[i].contains(P))
        return i;
    return -1;
#endif
  }

  inline __device__
  int SimpleNodeRenderer::computeInitialRank(const SimpleNodeRenderer::VopatGlobals &vopat,
                                             const SimpleNodeRenderer::OwnGlobals &globals,
                                             SimpleNodeRenderer::Ray ray,
                                             bool dbg)
  {
#if 1
    int closest = -1;
    float t_closest = CUDART_INF;
    for (int i=0;i<vopat.numWorkers;i++) {
      float t_min = 0.f;
      float t_max = t_closest;
      if (!boxTest(globals.rankBoxes[i],ray,t_min,t_max,t_closest))
        continue;
      closest = i;
    }
    if (ray.dbg) printf("(%i) INITIAL rank is %i\n",vopat.myRank,closest);
    return closest;
#else
    int closest = -1;
    float t_min = CUDART_INF;
    for (int i=0;i<vopat.numWorkers;i++) {
      if (boxTest(globals.rankBoxes[i],ray,t_min)) {
        closest = i;
      }
    }
    if (ray.dbg) printf("(%i) initial rank is %i\n",vopat.myRank,closest);
    return closest;
#endif
  }
  


  __global__ void doTraceRaysLocally(SimpleNodeRenderer::VopatGlobals vopat,
                                     SimpleNodeRenderer::OwnGlobals   globals)
  {
    int tid = threadIdx.x+blockIdx.x*blockDim.x;
    if (tid >= vopat.numRaysInQueue) return;

    SimpleNodeRenderer::Ray ray = vopat.rayQueueIn[tid];
    vec3f throughput = from_half(ray.throughput);
    vec3f org  = ray.origin;
    vec3f dir  = from_half(ray.direction);
    
    const box3f myBox = globals.rankBoxes[vopat.myRank];
    float t0, t1;
    clipRay(myBox,ray,t0,t1);

    if (ray.dbg) printf("(%i) tracing locally (%f %f)\n",vopat.myRank,t0,t1);
    
    const float dt = .001f;
    const float DENSITY = 1.1f;
    
    Random rnd(ray.pixelID,vopat.sampleID);
    int step = 0;
    for (float t = (int(t0/dt)+rnd()) * dt; t < t1; t += dt, ++step) {
      vec3f P = org + t * dir;
      vec3f relPos = (P - globals.myRegion.lower) * rcp(globals.myRegion.size());
      vec3i cellID = vec3i(relPos * vec3f(globals.numVoxels));
      if (cellID.x < 0 || cellID.x >= globals.numVoxels.x ||
          cellID.y < 0 || cellID.y >= globals.numVoxels.y ||
          cellID.z < 0 || cellID.z >= globals.numVoxels.z) continue;
      float f = globals.myVoxels[cellID.x
                                 +globals.numVoxels.x*(cellID.y
                                                       +globals.numVoxels.y*cellID.z)];
      if (ray.dbg && step == 10) printf("(%i) step %i val %f\n",
                                        vopat.myRank,step,f);
      f = f * dt * DENSITY;

      if (rnd() < f) {
#if 0
        // just absorb, done
        throughput = 0;
#else
        if (ray.isShadow) {
          throughput = 0;
          // TODO: *KILL* that ray instead of letting it go through black...
          break;
        } else {
          org = P; 
          dir = normalize(lightDirection());
          ray.origin = org;
          ray.direction = to_half(dir);
          clipRay(myBox,ray,t0,t1);
          t = rnd()*dt - dt; // subtract one dt because loop will add
                             // it back before next it
          ray.isShadow = true;
          if (ray.dbg) printf("(%i) BOUNCED (%f %f)\n",vopat.myRank,t0,t1);
          continue;
        }
#endif
      }
    }

    // throughput *= randomColor(vopat.myRank);
    ray.throughput = to_half(throughput);

    int nextNode = SimpleNodeRenderer::computeNextNode(vopat,globals,ray,t1);
    if (ray.dbg) printf("(%i) next is %i\n",vopat.myRank,nextNode);

    if (nextNode == -1) {
      // path exits volume - deposit to image
      // addToFB(&globals.accumBuffer[ray.pixelID],throughput);
      vec3f color
        = ray.isShadow
        ? lightColor()
        : backgroundColor();
      if (ray.crosshair) color = vec3f(1.f)-color;
      vopat.addPixelContribution(ray.pixelID,throughput*color);
      vopat.killRay(tid);
      // vopat.rayNextNode[tid] = -1;
    } else {
      // ray has another node to go to - add to queue
      vopat.forwardRay(tid,ray,nextNode);
      // atomicAdd(&vopat.perRankSendCounts[nextNode],1);
      // vopat.rayQueueIn[tid]  = ray;
      // vopat.rayNextNode[tid] = nextNode;
    }
  }
  
  void SimpleNodeRenderer::traceLocally(const SimpleNodeRenderer::VopatGlobals &vopat)
  {
    int blockSize = 512;
    int numBlocks = divRoundUp(vopat.numRaysInQueue,blockSize);
    if (numBlocks)
      doTraceRaysLocally<<<numBlocks,blockSize>>>
        (vopat,globals);
  }


  inline __device__ SimpleNodeRenderer::Ray
  generateRay(const SimpleNodeRenderer::VopatGlobals &globals,
              vec2i pixelID,
              vec2f pixelPos)
  {
    SimpleNodeRenderer::Ray ray;
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



  __global__
  void doGeneratePrimaryWave(SimpleNodeRenderer::VopatGlobals vopat,
                             SimpleNodeRenderer::OwnGlobals   globals)
  {
    int ix = threadIdx.x + blockIdx.x*blockDim.x;
    int iy = threadIdx.y + blockIdx.y*blockDim.y;
    if (ix >= vopat.fbSize.x) return;
    if (iy >= vopat.fbSize.y) return;

    int myRank = vopat.myRank;
    SimpleNodeRenderer::Ray ray    = generateRay(vopat,vec2i(ix,iy),vec2f(.5f));
    ray.dbg    = (vec2i(ix,iy) == vopat.fbSize/2);
    ray.crosshair = (ix == vopat.fbSize.x/2) || (iy == vopat.fbSize.y/2);
    int dest   = SimpleNodeRenderer::computeInitialRank(vopat,globals,ray);

    if (dest < 0) {
      /* "nobody" owns this pixel, set to background on rank 0 */
      if (myRank == 0) {
        vopat.accumBuffer[ray.pixelID] += backgroundColor();
      }
      return;
    }
    if (dest != myRank) {
      /* somebody else owns this pixel; we don't do anything */
      return;
    }
    int queuePos = atomicAdd(&vopat.perRankSendCounts[myRank],1);
    vopat.rayQueueIn[queuePos] = ray;
  }
  
  void SimpleNodeRenderer::generatePrimaryWave(const SimpleNodeRenderer::VopatGlobals &vopat)
  {
    vec2i blockSize(16);
    vec2i numBlocks = divRoundUp(vopat.fbSize,blockSize);
    doGeneratePrimaryWave<<<numBlocks,blockSize>>>(vopat,globals);
  }

  Renderer *createSimpleNodeRenderer(CommBackend *comm,
                                     Model::SP model,
                                     const std::string &fileNameBase,
                                     int rank)
  {
    SimpleNodeRenderer *nodeRenderer = new SimpleNodeRenderer(model,fileNameBase,rank);
    return new VopatRenderer<SimpleNodeRenderer>(comm,nodeRenderer);
  }
}
