#include "Memory.hpp"
#include <cassert>

namespace Backend {
  size_t memory_total_bytes(Context* ctx) {
    switch (ctx->device) {
    case Backend::SPIKE_DEVICE_DUMMY:
      return Backend::Dummy::memory_total_bytes(ctx);
#ifdef SPIKE_WITH_CUDA
    case Backend::SPIKE_DEVICE_CUDA:
      return Backend::CUDA::memory_total_bytes(ctx);
#endif
    default:
      assert("Unsupported backend" && false);
    };
    return -1;
  }

  size_t memory_free_bytes(Context* ctx) {
    switch (ctx->device) {
    case Backend::SPIKE_DEVICE_DUMMY:
      return Backend::Dummy::memory_free_bytes(ctx);
#ifdef SPIKE_WITH_CUDA
    case Backend::SPIKE_DEVICE_CUDA:
      return Backend::CUDA::memory_free_bytes(ctx);
#endif
    default:
      assert("Unsupported backend" && false);
    };
    return 0;
  }
}
