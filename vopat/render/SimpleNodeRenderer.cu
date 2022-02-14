#include "owl/owl_device.h"
#include "vopat/render/VopatRenderer.h"

namespace vopat {

  struct SimpleNodeRenderer {
    struct OwnGlobals {
      box3f *rankBoxes;
    };
    
    struct Ray {
      struct {
        uint32_t    pixelID : 31;
        uint32_t    dbg     :  1;
        uint32_t    isShadow:  1;
      };
      vec3f       origin;
      small_vec3f direction;
      small_vec3f throughput;
    };

    using VopatGlobals = typename VopatRenderer<SimpleNodeRenderer>::Globals;
    
    SimpleNodeRenderer(Model::SP model)
    {
      // ------------------------------------------------------------------
      // upload per-rank boxes
      // ------------------------------------------------------------------

      std::vector<box3f> hostRankBoxes;
      for (auto brick : model->bricks)
        hostRankBoxes.push_back(brick->spaceRange);
      rankBoxes.upload(hostRankBoxes);
      ownGlobals.rankBoxes = rankBoxes.get();
    };

    /*! one box per rank, which rays can use to find neext rank to send to */
    CUDAArray<box3f> rankBoxes;
    OwnGlobals       ownGlobals;

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
                        const float t_exit)
    {
      vec3f P = ray.origin + from_half(ray.direction) * (t_exit * (1.f+1e-3f));
      for (int i=0;i<vopat.numWorkers;i++)
        if (i != vopat.myRank && globals.rankBoxes[i].contains(P))
          return i;
      return -1;
    }
    
    Model::SP model;
  };



  inline __device__
  float fixDir(float f) { return (f==0.f)?1e-8f:f; }
  
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
  int SimpleNodeRenderer::computeInitialRank(const SimpleNodeRenderer::VopatGlobals &vopat,
                                             const SimpleNodeRenderer::OwnGlobals &globals,
                                             SimpleNodeRenderer::Ray ray,
                                             bool dbg)
  {
    int closest = -1;
    float t_min = CUDART_INF;
    for (int i=0;i<vopat.numWorkers;i++) {
      if (boxTest(globals.rankBoxes[i],ray,t_min)) {
        closest = i;
      }
    }
    return closest;
  }
  


  __global__ void doTraceRaysLocally(SimpleNodeRenderer::VopatGlobals vopat,
                                     SimpleNodeRenderer::OwnGlobals   globals)
  {
    int tid = threadIdx.x+blockIdx.x*blockDim.x;
    if (tid >= vopat.numRaysInQueue) return;

    SimpleNodeRenderer::Ray ray = vopat.rayQueueIn[tid];
    const box3f myBox = globals.rankBoxes[vopat.myRank];
    float t0, t1;
    clipRay(myBox,ray,t0,t1);

    vec3f throughput = from_half(ray.throughput);
    throughput *= randomColor(vopat.myRank);
    ray.throughput = to_half(throughput);

    int nextNode = SimpleNodeRenderer::computeNextNode(vopat,globals,ray,t1);

    if (nextNode == -1) {
      // path exits volume - deposit to image
      // addToFB(&globals.accumBuffer[ray.pixelID],throughput);
      vopat.addPixelContribution(ray.pixelID,throughput);
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
    int blockSize = 1024;
    int numBlocks = divRoundUp(vopat.numRaysInQueue,blockSize);
    if (numBlocks)
      doTraceRaysLocally<<<numBlocks,blockSize>>>
        (vopat,ownGlobals);
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
    int dest   = SimpleNodeRenderer::computeInitialRank(vopat,globals,ray);

    if (dest < 0) {
      /* "nobody" owns this pixel, set to background on rank 0 */
      if (myRank == 0) {
        vopat.accumBuffer[ray.pixelID] += vec3f(.5f,.5f,.5f);
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
    doGeneratePrimaryWave<<<numBlocks,blockSize>>>(vopat,ownGlobals);
  }


#if 0


  


#endif

  Renderer *createSimpleNodeRenderer(CommBackend *comm, Model::SP model)
  {
    SimpleNodeRenderer *nodeRenderer = new SimpleNodeRenderer(model);
    return new VopatRenderer<SimpleNodeRenderer>(comm,nodeRenderer);
  }
}
