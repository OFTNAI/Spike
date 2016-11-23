#pragma once

#include <cstddef>

#include "Spike/Backend/Context.hpp"

#include "Spike/Backend/Dummy/Memory.hpp"

namespace Backend {
  size_t total_memory(Context* ctx = _global_ctx) {
    switch (ctx->device) {
    case Backend::SPIKE_DEVICE_DUMMY:
      return Backend::Dummy::total_memory(ctx);
    // case Backend::SPIKE_DEVICE_CUDA:
    //   return Backend::CUDA::total_memory(ctx);
    default:
      assert("Unsupported backend" && false);
    };
    return -1;
  }

  size_t free_memory(Context* ctx = _global_ctx) {
    switch (ctx->device) {
    case Backend::SPIKE_DEVICE_DUMMY:
      return Backend::Dummy::free_memory(ctx);
    // case Backend::SPIKE_DEVICE_CUDA:
    //   return Backend::CUDA::free_memory(ctx);
    default:
      assert("Unsupported backend" && false);
    };
    return 0;
  }
}
