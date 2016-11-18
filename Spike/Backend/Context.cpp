#include "Context.hpp"

namespace Backend {
  void init_global_context() {
    if (!_global_ctx)
      _global_ctx = new Context;
    // TODO:
    _global_ctx->Device = SPIKE_DEVICE_CUDA;
  }

  Context* get_current_context() {
    return _global_ctx;
  }
}
