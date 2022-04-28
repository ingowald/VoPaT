#pragma once

typedef struct _RaflContext *rafl_context_t;

struct RaflConfig {
  int  raySizeInBytes      = 32;
  int  maxLiveRaysPerPixel = 2;
  int  numStreams          = 1;
  /* whether pixels have an alpha channel that needs to be carried
     along, too (currently not implemented) */
  bool hasAlpha            = false;
  /*! whether to convert final, added pixels to RGBA8, or leave as float4 */
  bool finalPixelsAreRGBA8 = true;
};

rafl_context_t rafl_init(MPI_Comm          comm,
                         int               gpuID,
                         const RaflConfig *config);

/*! tells the context what frame size to use */
void rafl_resize(rafl_context_t rafl,
                 int2 fbSize);

struct RaflTraceCtx {
  /*! input ray queue, where traceAndShade() can find the rays it
      needs to trace and shade */
  const    void *rayQueueIn;
  
  /*! number of rays in rayQueueIn */
  const    int   numRaysIn;

  /*! memory region where outgoing rays are to be written to */
  void          *rayQueueOut;

  /*! memory region where, for each ray in the output queue, we store
      an int specifying which other rank this rays is supposed to get
      forwarded to */
  int           *rayDestOut;

  /*! memory region where outgoing rays are to be written to */
  
  /*! pointer to an int that stores how many rays there are already in
      the out queue; can be used to atomically append rays to end of
      queue */
  volatile int  *pNumRaysOut;
  /*! local accumulatoin buffer to add pixel contributions to */
  float4 *fbAccum;
  int2    fbSize;
};

/*! returns number of rays generated into the out queue */
typedef int (*RaflRayGenFunc)(const void *userData,
                              const RaflTraceCtx *rafl);

/*! trace and shade, using the given device context to add pixel
    contributions and/or append new rays */
typedef int (*RaflTraceLocalFunc)(const void *userData,
                                  const RaflTraceCtx *rafl));

/*! traces one or more waves, using the user-supplied functions for
  generating a new wavefront and local trace/shade function. This
  function will accumulate pixels into the local frame buffer, but
  will *not* do any adding of the per-rank frame buffer (that's what
  rafl_combine() is for */
void rafl_trace(rafl_context_t     _rafl,
                RaflRayGenFunc     rayGen,
                RaflTraceLocalFunc localTraceAndShadeFunc,
                int                maxWaves);

/*! two-stage trace function - in this case one function only does the
    tracing, and is allowed to 'shelve' rays for later sahding; the
    shading func doesn't even get called until all tracing is done,
    buf *if* it gets called it'll also get all the shelved rays
    re-activated, too */
void rafl_trace2(rafl_context_t     _rafl,
                 RaflRayGenFunc     rayGen,
                 RaflTraceLocalFunc localTrace,
                 RaflTraceLocalFunc localShade,
                 int                maxWaves);

/*! combined different ranks' partial frame buffer results, and
    optionally, where to put the total final frame buffer */
void rafl_combine_ranks(rafl_context_t _rafl);

/*! describes a horizontal slice of a final frame buffer - this is the
    part of the frame buffer where rafl will put the added partial
    images. Each pixel in this frame buffer will be 'normamlized' by
    number of samples per pixel */
struct RaflPartialFB {
  float4 *pixels_f32;
  int     size_x;
  int     begin_y;
  int     end_y;
};

/*! returns the current rank's partial frame buffer into which
    rafl_combine_ranks has added all the ranks' contributions for that
    ranks' assigned range of pixels. Allows app to do its own
    modifictions to those pixels (e.g., tone mapping), and/or to do
    its own version of what rafl_combine_at_master() would otherwise
    do */
void rafl_get_partial_fb(rafl_context_t rafl,
                         RaflPartialFB *partialFB);

/*! takes the different ranks' local FBs, converts pixels to final
    master frame buffer format, and sends those to the master. */
void rafl_combine_at_master(rafl_context_t _rafl,
                            void  *masterFbPointer,
                            size_t masterFbStride);

/*! clears the local rank frame buffers */
void rafl_clear(rafl_context_t _rafl);


#ifdef __CUDA_ARCH__
namespace rafl {
  /*! helper function for (atomically) adding local pixel
      contributions */
  inline __device__ void addToPixel(const RaflTraceCtx *rafl,
                                    int2                pixelID,
                                    float4              contrib)
  {
    float4 *pixel = rafl->fbAccum + (pixelID.x + pixelID.y * rafl->fbSize.x);
    atomicAdd(&pixel->x,contrib.x);
    atomicAdd(&pixel->y,contrib.y);
    atomicAdd(&pixel->z,contrib.z);
    atomicAdd(&pixel->w,contrib.w);
  }

  /*! helper function for (atomically) adding local pixel
      contributions */
  inline __device__ void addToPixel(const RaflTraceCtx *rafl,
                                    int2                pixelID,
                                    float3              contrib)
  {
    float4 *pixel = rafl->fbAccum + (pixelID.x + pixelID.y * rafl->fbSize.x);
    atomicAdd(&pixel->x,contrib.x);
    atomicAdd(&pixel->y,contrib.y);
    atomicAdd(&pixel->z,contrib.z);
  }

  /*! helper function for (atomically) appending new rays */
  template<typename RayT>
  inline __device__ void forwardRay(const RaflTraceCtx *rafl,
                                    RayT ray,
                                    int  rankToSendTo)
  {
    int outID = atomicAdd(&rafl->pNumRaysOut,1);
    ((RayT *)rafl->rayQueueOut)[outID] = ray;
    ral->rayDestOut[outID] = rankToSendTo;
  }
}
#endif



