#include "owl/owl_device.h"
#include "vopat/render/VopatRenderer.h"
#include "owl/common/math/random.h"

namespace vopat {

  using Random = owl::common::LCG<8>;

  inline __device__ vec3f lightColor() { return vec3f(1.f,1.f,1.f); }
  inline __device__ vec3f lightDirection()
  {
    return vec3f(1.f,.1f,.5f);
    // return (0.f,0.f,1.f);
  }

  inline __device__
  float fixDir(float f) { return (f==0.f)?1e-6f:f; }
  
  inline __device__
  vec3f fixDir(vec3f v)
  { return {fixDir(v.x),fixDir(v.y),fixDir(v.z)}; }
  

 
 



  
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
      int age;
      inline __device__ void setDirection(vec3f v) { direction = fixDir(normalize(v)); }
      inline __device__ vec3f getDirection() const {
        return direction;
        // return from_half(direction);
      }
      vec3f       origin;
      vec3f direction;
      // small_vec3f direction;
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

    static inline __device__
    vec3f backgroundColor(const Ray &ray,
                    const VopatGlobals &globals)
    {
      int iy = ray.pixelID / globals.fbSize.x;
      float t = iy / float(globals.fbSize.y);
      const vec3f c = (1.0f - t)*vec3f(1.0f, 1.0f, 1.0f) + t * vec3f(0.5f, 0.7f, 1.0f);
      return c;
    }

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
                        const float t_already_travelled,
                        bool dbg = false);
    
    Model::SP model;
  };



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
  float floor(float f) { return ::floorf(f); }
  
  inline __device__
  vec3f floor(vec3f v) { return { floor(v.x),floor(v.y),floor(v.z) }; }
  
  
  inline __device__
  bool boxTest(box3f box, SimpleNodeRenderer::Ray ray, float &t0, float &t1,
               bool dbg=false)
  {
    vec3f dir = ray.getDirection();

    if (dbg)
      printf(" ray (%f %f %f)(%f %f %f) box (%f %f %f)(%f %f %f)\n",
             ray.origin.x,
             ray.origin.y,
             ray.origin.z,
             dir.x,
             dir.y,
             dir.z,
             box.lower.x,
             box.lower.y,
             box.lower.z,
             box.upper.x,
             box.upper.y,
             box.upper.z);
    
    vec3f t_lo = (box.lower - ray.origin) * rcp(dir);
    vec3f t_hi = (box.upper - ray.origin) * rcp(dir);
    
    vec3f t_nr = min(t_lo,t_hi);
    vec3f t_fr = max(t_lo,t_hi);

    t0 = max(t0,reduce_max(t_nr));
    t1 = min(t1,reduce_min(t_fr));
    if (dbg) printf("  -> t0 %f t1 %f\n",t0,t1);
    return (t0 <= t1);
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
                                          const float t_already_travelled,
                                          bool dbg)
  {
    if (dbg) printf("finding next that's t >= %f and rank != %i\n",
                    t_already_travelled,vopat.myRank);
                    
    int closest = -1;
    float t_closest = CUDART_INF;
    for (int i=0;i<vopat.numWorkers;i++) {
      if (i == vopat.myRank) continue;

      float t0 = t_already_travelled * (1.f+1e-5f);
      float t1 = t_closest; 
      if (!boxTest(globals.rankBoxes[i],ray,t0,t1,dbg))
        continue;
      if (dbg) printf("   accepted rank %i dist %f\n",i,t0);
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
    vec3f org = ray.origin;
    vec3f dir = ray.getDirection();

    
    const box3f myBox = globals.rankBoxes[vopat.myRank];
    float t0 = 0.f, t1 = CUDART_INF;
    boxTest(myBox,ray,t0,t1);

    if (ray.dbg) printf("(%i) tracing locally (%f %f)\n",vopat.myRank,t0,t1);
    
    // Random rnd((int)ray.pixelID+vopat.myRank+ray.age,vopat.sampleID);
    Random rnd((int)ray.pixelID,vopat.sampleID+vopat.myRank*0x123456);
    int step = 0;
    vec3i numVoxels = globals.numVoxels;
    vec3i numCells = numVoxels - 1;

    // maximum possible voxel density
    const float dt = .5f; // relative to voxels
    const float DENSITY = .03f;
#if 0
    float t = t0 + dt * rnd();
    while (true) {
      if (t >= t1) break;
      ++step;
      // if (step > 1000) {
      //   printf("huge iteration ... age %i step %i t %f (%f %f)\n",
      //          ray.age,step,dt,t0,t1);
      //   break;
      // }

      // Update current position
      vec3f P = org + t * dir;
      vec3i cellID = vec3i(floor(P - globals.myRegion.lower));
      if (cellID.x < 0 || (cellID.x >= numCells.x) ||
          cellID.y < 0 || (cellID.y >= numCells.y) ||
          cellID.z < 0 || (cellID.z >= numCells.z)) {
        // if (ray.dbg)
        //   printf("SKIPPING step %i at %f P %f %f %f cell %i %i %i\n",
        //          step,t,P.x,P.y,P.z,cellID.x,cellID.y,cellID.z);

        t += dt;
        continue;
      }
      
      // Sample heterogeneous media
      float f = globals.myVoxels[cellID.x
                                +numVoxels.x*(cellID.y
                                              +numVoxels.y*cellID.z)];
#if 1
      f = max(0.f,f-0.1);
#endif
      // if (ray.dbg && step < 10)
      //   printf("step %i t %f cell %i %i %i f %f\n",
      //          step,t,cellID.x,cellID.y,cellID.z,f);
      f *= (DENSITY * dt);
      if (rnd() >= f) {
        t += dt;
        continue;
      }

      if (ray.isShadow) {
        throughput = 0;
        vopat.killRay(tid);
        return;
      } else {
        org = P; 
        ray.origin = org;
        ray.setDirection(lightDirection());
        dir = ray.getDirection();
        
        t0 = 0.f;
        t1 = CUDART_INF;
        boxTest(myBox,ray,t0,t1,ray.dbg);
        float t_old = t;
        t = dt * rnd();
        ray.isShadow = true;
        if (ray.dbg) printf("(%i) BOUNCED t at t=%f, new t %f (%f %f)\n",vopat.myRank,t_old,t,t0,t1);
        continue;
      }
    }
#else
    float majorant = 1.f ; // must be larger than the max voxel density
    float t = t0;
    while (true) {
      // Sample a distance
      t = t - (log(1.0f - rnd()) / (majorant*DENSITY)) * dt; 

      // A boundary has been hit
      if (t >= t1) break;

      // Update current position
      vec3f P = org + t * dir;
      // vec3f relPos = (P - globals.myRegion.lower) * rcp(globals.myRegion.size());
      // if (relPos.x < 0.f) continue;
      // if (relPos.y < 0.f) continue;
      // if (relPos.z < 0.f) continue;

      // vec3i cellID = vec3i(relPos * vec3f(numCells));
      vec3i cellID = vec3i(floor(P - globals.myRegion.lower));
      if (cellID.x < 0 || (cellID.x >= numCells.x) ||
          cellID.y < 0 || (cellID.y >= numCells.y) ||
          cellID.z < 0 || (cellID.z >= numCells.z)) continue;

      // Sample heterogeneous media
      float f = globals.myVoxels[cellID.x
                                +numVoxels.x*(cellID.y
                                              +numVoxels.y*cellID.z)];
#if 1
      f = max(0.f,f-0.1);
#endif
      
      // if (ray.dbg && (step < 2 || step == 10))
      //   printf("(%i) step %i rel (%f %f %f) cell (%i %i %i) val %f\n",
      //          vopat.myRank,
      //          step,
      //          relPos.x,
      //          relPos.y,
      //          relPos.z,
      //          cellID.x,
      //          cellID.y,
      //          cellID.z,
      //          f);
      
      // Check if a collision occurred (real particles / real + fake particles)
      if (rnd() < f / majorant) {
        if (ray.isShadow) {
        // throughput = 0;
        // vopat.killRay(tid);
        // return;
          throughput = 0;
          // TODO: *KILL* that ray instead of letting it go through black...
          break;
        } else {
          org = P; 
          ray.origin = org;
          ray.setDirection(lightDirection());
          dir = ray.getDirection();
          
          t0 = 0.f;
          t1 = CUDART_INF;
          boxTest(myBox,ray,t0,t1);
          t = 0.f; // reset t to the origin
          ray.isShadow = true;
          if (ray.dbg) printf("(%i) BOUNCED t in %f (%f %f)\n",vopat.myRank,t,t0,t1);
          continue;
        }
      }
    }
#endif
    
    // throughput *= randomColor(vopat.myRank);
    ray.throughput = to_half(throughput);
    ray.age++;
    int nextNode = SimpleNodeRenderer::computeNextNode(vopat,globals,ray,t1,ray.dbg);

    if (// ray.age > 10 ||
        nextNode == -1) {
      // path exits volume - deposit to image
      // addToFB(&globals.accumBuffer[ray.pixelID],throughput);
      float ambient = 0.1f;
      vec3f albedo = vec3f(.8f, 1.f, .8f); // you'd normally sample this from the volume
      bool missed = (throughput.x == 1.f && throughput.y == 1.f && throughput.z == 1.f);
      vec3f color;
      // primary ray hitting background
      if (missed && !ray.isShadow) color = SimpleNodeRenderer::backgroundColor(ray,vopat);
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
    ray.setDirection(dir);
    ray.throughput = to_half(vec3f(1.f));
    ray.age = 0;
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
    ray.dbg    = false;//(vec2i(ix,iy) == vopat.fbSize/2);
    ray.crosshair = (ix == vopat.fbSize.x/2) || (iy == vopat.fbSize.y/2);
    int dest   = SimpleNodeRenderer::computeInitialRank(vopat,globals,ray);

    if (dest < 0) {
      /* "nobody" owns this pixel, set to background on rank 0 */
      if (myRank == 0) {
        vopat.accumBuffer[ray.pixelID] += SimpleNodeRenderer::backgroundColor(ray,vopat);
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
