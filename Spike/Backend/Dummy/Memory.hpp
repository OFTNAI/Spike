#pragma once

#include <cstddef>

namespace Backend {
  namespace Dummy {
    size_t total_memory(Context* ctx = _global_ctx) {
      return -1;
    }

    size_t free_memory(Context* ctx = _global_ctx) {
      return 0;
    }
  }
}
