#pragma once

#include <cstddef>

#include "Spike/Backend/Context.hpp"

#include "Spike/Backend/Dummy/Memory.hpp"

namespace Backend {
  size_t total_memory(Context* ctx = _global_ctx);

  size_t free_memory(Context* ctx = _global_ctx);
}
