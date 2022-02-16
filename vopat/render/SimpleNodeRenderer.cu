#include "owl/owl_device.h"
#include "vopat/render/VopatRenderer.h"
#include "owl/common/math/random.h"

namespace vopat {

  using Random = owl::common::LCG<4>;

  inline __device__ vec3f backgroundColor() { return .3f; }
  
  inline __device__ vec3f lightColor() { return vec3f(1.f,1.f,1.f); }
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
  
  // inline __device__
  // bool boxTest(box3f box, SimpleNodeRenderer::Ray ray, float &t_min)
  // {
  //   vec3f t_lo = (box.lower - ray.origin) * rcp(fixDir(from_half(ray.direction)));
  //   vec3f t_hi = (box.upper - ray.origin) * rcp(fixDir(from_half(ray.direction)));
    
  //   vec3f t_nr = min(t_lo,t_hi);
  //   vec3f t_fr = max(t_lo,t_hi);

  //   float t0 = max(0.f,reduce_max(t_nr));
  //   float t1 = min(t_min,reduce_min(t_fr));

  //   if (t0 >= t1) return false;
  //   t_min = t0;
  //   return true;
  // }

  inline __device__
  bool boxTest(box3f box, SimpleNodeRenderer::Ray ray, float &t0, float &t1)
  {
    vec3f t_lo = (box.lower - ray.origin) * rcp(fixDir(from_half(ray.direction)));
    vec3f t_hi = (box.upper - ray.origin) * rcp(fixDir(from_half(ray.direction)));
    
    vec3f t_nr = min(t_lo,t_hi);
    vec3f t_fr = max(t_lo,t_hi);

    t0 = max(t0,reduce_max(t_nr));
    t1 = min(t1,reduce_min(t_fr));

    return (t0 < t1);
  }

  
  // inline __device__
  // void clipRay(box3f box, SimpleNodeRenderer::Ray ray, float &t_min, float &t_max)
  // {
  //   vec3f t_lo = (box.lower - ray.origin) * rcp(fixDir(from_half(ray.direction)));
  //   vec3f t_hi = (box.upper - ray.origin) * rcp(fixDir(from_half(ray.direction)));
    
  //   vec3f t_nr = min(t_lo,t_hi);
  //   vec3f t_fr = max(t_lo,t_hi);

  //   float t0 = max(0.f,reduce_max(t_nr));
  //   float t1 = reduce_min(t_fr);

  //   t_min = t0;
  //   t_max = t1;
  // }
  
  inline __device__
  int SimpleNodeRenderer::computeNextNode(const VopatGlobals &vopat,
                                          const OwnGlobals &globals,
                                          const Ray &ray,
                                          const float t_already_travelled)
  {
    int closest = -1;
    float t_closest = CUDART_INF;
    for (int i=0;i<vopat.numWorkers;i++) {
      if (i == vopat.myRank) continue;

      float t0 = t_already_travelled;
      float t1 = t_closest; 
      if (!boxTest(globals.rankBoxes[i],ray,t0,t1))
        continue;
      t_closest = t0;
      closest = i;
    }
    if (ray.dbg) printf("(%i) NEXT rank is %i\n",vopat.myRank,closest);
    return closest;
  }

  inline __device__
  int SimpleNodeRenderer::computeInitialRank(const SimpleNodeRenderer::VopatGlobals &vopat,
                                             const SimpleNodeRenderer::OwnGlobals &globals,
                                             SimpleNodeRenderer::Ray ray,
                                             bool dbg)
  {
    int closest = -1;
    float t_closest = CUDART_INF;
    for (int i=0;i<vopat.numWorkers;i++) {
      float t_min = 0.f;
      float t_max = t_closest;
      if (!boxTest(globals.rankBoxes[i],ray,t_min,t_max))
        continue;
      closest = i;
      t_closest = t_min;
    }
    if (ray.dbg) printf("(%i) INITIAL rank is %i\n",vopat.myRank,closest);
    return closest;
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
    float t0 = 0.f, t1 = CUDART_INF;
    boxTest(myBox,ray,t0,t1);

    if (ray.dbg) printf("(%i) tracing locally (%f %f)\n",vopat.myRank,t0,t1);
    
    //const float DENSITY = 1.5f;//10000.f;
    Random rnd(ray.pixelID,vopat.sampleID);
    int step = 0;
    vec3i numVoxels = globals.numVoxels;
    vec3i numCells = numVoxels - 1;

    // maximum possible voxel density
    float majorant = 1.f; // must be larger than the max voxel density
    const float dt = .01f; // relative to voxels
    float t = t0;
    while (true) {
      // Sample a distance
      t = t - (log(1.0f - rnd()) / majorant) * dt; 

      // A boundary has been hit
      if (t >= t1) break;

      // Update current position
      vec3f P = org + t * dir;
      vec3f relPos = (P - globals.myRegion.lower) * rcp(globals.myRegion.size());
      if (relPos.x < 0.f) continue;
      if (relPos.y < 0.f) continue;
      if (relPos.z < 0.f) continue;

      vec3i cellID = vec3i(relPos * vec3f(numCells));
      if (cellID.x < 0 || (cellID.x >= numCells.x) ||
          cellID.y < 0 || (cellID.y >= numCells.y) ||
          cellID.z < 0 || (cellID.z >= numCells.z)) continue;

      // Sample heterogeneous media
      float f = globals.myVoxels[cellID.x
                                +numVoxels.x*(cellID.y
                                              +numVoxels.y*cellID.z)];
      
      if (ray.dbg && (step < 2 || step == 10))
        printf("(%i) step %i rel (%f %f %f) cell (%i %i %i) val %f\n",
               vopat.myRank,
               step,
               relPos.x,
               relPos.y,
               relPos.z,
               cellID.x,
               cellID.y,
               cellID.z,
               f);
      
      // Check if a collision occurred (real particles / real + fake particles)
      if (rnd() < f / majorant) {
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
          ray.origin = org;
          ray.direction = to_half(normalize(lightDirection()));
          dir = from_half(ray.direction);
          
          t0 = 0.f;
          t1 = CUDART_INF;
          boxTest(myBox,ray,t0,t1);
          t = 0.f; // reset t to the origin
          ray.isShadow = true;
          if (ray.dbg) printf("(%i) BOUNCED t in %f (%f %f)\n",vopat.myRank,t,t0,t1);
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
      float ambient = .1f;
      vec3f albedo = vec3f(.8f, 1.f, .8f); // you'd normally sample this from the volume
      bool missed = (throughput.x == 1.f && throughput.y == 1.f && throughput.z == 1.f);
      vec3f color;
      // primary ray hitting background
      if (missed && !ray.isShadow) color = backgroundColor();
      // primary ray hitting light
      else if (missed) color = lightColor() * albedo; 
      // else, we're in shadow
      else color = lightColor() * albedo * ambient;

      if (ray.crosshair) color = vec3f(1.f)-color;
      vopat.addPixelContribution(ray.pixelID,color);
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
