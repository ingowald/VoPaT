#pragma once

/*! describes a horizontal slice of a final frame buffer - this is the
    part of the frame buffer where rafl will put the added partial
    images */
struct RaflPartialFB {
  union {
    float4   *pixels_f32;
    uint32_t *pixels_ui8;
  };
  int     begin_y;
  int     end_y;
};

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
                 int fbSize_x,
                 int fbSize_y,
                 /*! pointer to a structure describing where rafl will
                     put the pixels. if null rafl will *not* allocate
                     any memory of fill in this data, and assume the
                     user will do this and pass a user-supplied
                     version of this to rafl_finish; if non-null, rafl
                     will allocate memory, fill this in, and the user
                     can use it afterwards */
                 RaflPartialFB *partialFB);

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
  volatile int  *pNumRaysInOutQueue;
  /*! local accumulatoin buffer to add pixel contributions to */
  float4 *fbPixels;
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
void rafl_combine(rafl_context_t _rafl,
                  RaflPartialFB  partialFB,
                  void          *finalFB = 0);

/*! clears the local rank frame buffers */
void rafl_clear(rafl_context_t _rafl);
