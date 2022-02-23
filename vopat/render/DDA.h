#pragma once

#include "owl/common/math/vec.h"
#include "owl/common/math/AffineSpace.h"

#define DDA_FAST 1

namespace dda {
  using namespace owl::common;

  inline __device__
  bool clipRay(vec3f org, vec3f rcp_dir, box3f box, float &t0, float &t1)
  {
    vec3f t_lo = (box.lower - org) * rcp_dir;
    vec3f t_up = (box.upper - org) * rcp_dir;
    vec3f t_nr = min(t_lo,t_up);
    vec3f t_fr = max(t_lo,t_up);
    t0 = max(t0,reduce_max(t_nr));
    t1 = min(t1,reduce_min(t_fr));
    return t0 < t1;
  }

#if DDA_FAST
  inline __device__ int get(vec3i v, int dim)
  {
    return dim == 0 ? v.x : (dim == 1 ? v.y : v.z);
  }
  inline __device__ float get(vec3f v, int dim)
  {
    return dim == 0 ? v.x : (dim == 1 ? v.y : v.z);
  }

  inline __device__ void set(vec3f &vec, int dim, float value)
  {
    vec.x = (dim == 0) ? value : vec.x;
    vec.y = (dim == 1) ? value : vec.y;
    vec.z = (dim == 2) ? value : vec.z;
  }
  
  inline __device__ void set(vec3i &vec, int dim, int value)
  {
    vec.x = (dim == 0) ? value : vec.x;
    vec.y = (dim == 1) ? value : vec.y;
    vec.z = (dim == 2) ? value : vec.z;
  }

  inline __device__ int smallestDim(vec3f v)
  {
    return v.x <= min(v.y,v.z) ? 0 : (v.y <= min(v.x,v.z) ? 1 : 2);
  }
#else
  inline __device__ int   get(vec3i v, int dim) { return v[dim]; }
  inline __device__ float get(vec3f v, int dim) { return v[dim]; }

  inline __device__ void set(vec3f &vec, int dim, float value) { vec[dim] = value; }
  inline __device__ void set(vec3i &vec, int dim, int   value) { vec[dim] = value; }

  inline __device__ int smallestDim(vec3f v)
  {
    return arg_min(v);
  }
#endif
  
  template<typename Lambda>
  inline __device__ void dda3(vec3f org,
                              vec3f dir,
                              float tMax,
                              vec3ui gridSize,
                              const Lambda &lambda,
                              bool dbg)
  {    
    const box3f bounds = { vec3f(0.f), vec3f(gridSize) };
    vec3f rcp_dir = rcp(dir);
    float t0 = 0.f, t1 = tMax;
    if (!clipRay(org,rcp_dir,bounds,t0,t1)) {
      // if (dbg) printf(" -> clipped %f %f\n",t0,t1);
      return;
    }
    const vec3f P = org + t0 * dir;
    vec3i idx = vec3i(P);
    vec3f res = P - vec3f(idx);
    
    if (idx.x >= gridSize.x) { idx.x = gridSize.x-1; res.x = 1.f; }
    if (idx.y >= gridSize.y) { idx.y = gridSize.y-1; res.y = 1.f; }
    if (idx.z >= gridSize.z) { idx.z = gridSize.z-1; res.z = 1.f; }
    if (idx.x < 0) { idx.x = 0; res.x = 0.f; }
    if (idx.y < 0) { idx.y = 0; res.y = 0.f; }
    if (idx.z < 0) { idx.z = 0; res.z = 0.f; }

    const vec3i stop = {
                        (dir.x < 0.f) ? -1 : (int)gridSize.x,
                        (dir.y < 0.f) ? -1 : (int)gridSize.y,
                        (dir.z < 0.f) ? -1 : (int)gridSize.z
    };
    // if (dbg) printf("# stop %i %i %i\n",stop.x,stop.y,stop.z);
    
    if (dir.x < 0.f) {
      res.x = (1.f-res.x) * - rcp_dir.x;
    } else {
      res.x = res.x / rcp_dir.x;
    }
    if (dir.y < 0.f) {
      res.y = (1.f-res.y) * - rcp_dir.y;
    } else {
      res.y = res.y / rcp_dir.y;
    }
    if (dir.z < 0.f) {
      res.z = (1.f-res.z) * - rcp_dir.z;
    } else {
      res.z = res.z / rcp_dir.z;
    }
    // if (dbg) printf("# res %i %i %i\n",res.x,res.y,res.z);

    // int step = 0;
    while (1) {
      // if (dbg) printf("# ---- step %i -> calling %i %i %i\n",
      //                 step,idx.x,idx.y,idx.z);
      const bool userWantsToGoOn = lambda(idx);
      if (userWantsToGoOn) break;

      const int dim = smallestDim(res);
      res -= reduce_min(res);
      int idx_dim = get(idx,dim);
      float rcp_dir_dim = get(rcp_dir,dim);
      idx_dim = idx_dim + ((rcp_dir_dim < 0.f) ? -1 : +1);
      if (idx_dim == get(stop,dim)) break;
      set(idx,dim,idx_dim);
      set(res,dim,abs(rcp_dir_dim));
      // ++step;
    }
  }
}

