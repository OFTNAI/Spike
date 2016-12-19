#include "Memory.hpp"
#include <cassert>

namespace Backend {
  size_t total_memory(Context* ctx) {
    switch (ctx->device) {
    case Backend::SPIKE_DEVICE_DUMMY:
      return Backend::Dummy::total_memory(ctx);
    case Backend::SPIKE_DEVICE_CUDA:
      return Backend::CUDA::total_memory(ctx);
    default:
      assert("Unsupported backend" && false);
    };
    return -1;
  }

  size_t free_memory(Context* ctx) {
    switch (ctx->device) {
    case Backend::SPIKE_DEVICE_DUMMY:
      return Backend::Dummy::free_memory(ctx);
    case Backend::SPIKE_DEVICE_CUDA:
      return Backend::CUDA::free_memory(ctx);
    default:
      assert("Unsupported backend" && false);
    };
    return 0;
  }
}
