#include "MemoryUsage.hpp"

namespace Backend {
  size_t memory_total_bytes(Context* ctx) {
    if (ctx == nullptr)
      ctx = Backend::get_current_context();
    ::MemoryManager mm;
    mm.init_backend(ctx);
    return mm.backend_total_bytes();
  }

  size_t memory_free_bytes(Context* ctx) {
    if (ctx == nullptr)
      ctx = Backend::get_current_context();
    ::MemoryManager mm;
    mm.init_backend(ctx);
    return mm.backend_free_bytes();
  }
}

void MemoryManager::reset_state() {
  backend()->reset_state();
}

SPIKE_MAKE_INIT_BACKEND(MemoryManager);
