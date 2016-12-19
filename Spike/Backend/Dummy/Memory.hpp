#pragma once

#include <cstddef>

namespace Backend {
  namespace Dummy {
    inline size_t total_memory(Context* ctx = _global_ctx) {
      return -1;
    }

    inline size_t free_memory(Context* ctx = _global_ctx) {
      return 0;
    }
  }
}
