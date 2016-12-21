#pragma once

#include <cstddef>

namespace Backend {
  namespace Dummy {
    inline size_t memory_total_bytes(Context* ctx = _global_ctx) {
      return 0;
    }

    inline size_t memory_free_bytes(Context* ctx = _global_ctx) {
      return 0;
    }
  }
}
