#pragma once

#include <cstddef>

#include "Spike/Backend/Context.hpp"

#include "Spike/Backend/Dummy/Memory.hpp"
#ifdef SPIKE_WITH_CUDA
#include "Spike/Backend/CUDA/Memory.hpp"
#endif

namespace Backend {
  size_t memory_total_bytes(Context* ctx = _global_ctx);

  size_t memory_free_bytes(Context* ctx = _global_ctx);
}
