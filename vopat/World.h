#include "vopat/Volume.h"

namespace copat {

  struct World {
    enum { MAX_DIR_LIGHTS = 2 };
    struct Globals {
  
      /*! hardcoded these, for now */
      inline __device__ float ambient()         const { return lights.ambientTerm; }
      inline __device__ float ambientEnvLight() const { return lights.ambientEnvLight; }
      inline __device__ int   numDirLights()    const { return lights.numDirectional; }
      inline __device__ vec3f lightRadiance(int which)     const {
        return lights.directional[which].rad;
      }
      inline __device__ vec3f lightDirection(int which) const {
        return lights.directional[which].dir;
      }

      /*! sample a light, return light sample in lDir/lRad, and reutrn
          pdf of this sample */
      inline __device__ float sampleLight(Random &rnd,
                                         const vec3f surfP,
                                         const vec3f surfN,
                                         vec3f &lDir,
                                         vec3f &lRad) const
      {


        float pdf = 1.f;
        const int numLights = numDirLights();
        if (numLights == 0) return 0.f;
        
        int which = int(rnd() * numLights);
        if (which < 0 || which >= numLights) which = 0;
        pdf *= 1.f / numLights;

        lDir = lights.directional[which].dir;
        lRad = lights.directional[which].rad;

#if 0
        // HACK to create a giant light in the middle...
        if (length(surfP-vec3f(256.f)) > 50.f) return 0.f;
        else lRad *= 500.f;
#endif
        
        return pdf;
      }
      
      int    islandRank;
      int    islandSize;

      struct {
        // ambient term that applies to every sample, w/o shadowing
        float ambientTerm = .0f;
        // ambient environment light, only added if paths get lost to env
        float ambientEnvLight = .2f;
        int numDirectional = 0;
        struct {
          vec3f dir = { .1f, 1.f, .1f };
          vec3f rad = { 1.f, 1.f, 1.f };
        } directional[MAX_DIR_LIGHTS];
      } lights;
    };
  };
  
};
