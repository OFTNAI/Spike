#include "Context.hpp"

Context* _global_ctx = nullptr;

namespace Backend {
  void init_global_context() {
    if (!_global_ctx)
      _global_ctx = new Context;
    // TODO: Add other device types (esp CUDA)
#ifdef SPIKE_WITH_CUDA
    _global_ctx->device = SPIKE_DEVICE_CUDA;
#else
    _global_ctx->device = SPIKE_DEVICE_DUMMY;
#endif
  }

  Context* get_current_context() {
    return _global_ctx;
  }
}
