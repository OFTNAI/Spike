#include "Context.hpp"

Context* _global_ctx = nullptr;

namespace Backend {
  /* Set up default context */
  void init_global_context() {
    if (!_global_ctx)
      _global_ctx = new Context;
  }

  Context* get_current_context() {
    return _global_ctx;
  }
}
