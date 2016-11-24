#include "Context.hpp"

Context* _global_ctx = nullptr;

namespace Backend {
  void init_global_context() {
    if (!_global_ctx)
      _global_ctx = new Context;
    // TODO:
    _global_ctx->device = SPIKE_DEVICE_DUMMY;
  }

  Context* get_current_context() {
    return _global_ctx;
  }
}
