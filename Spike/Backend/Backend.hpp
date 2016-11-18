#pragma once

#include "Context.hpp"

namespace Backend {
  class Generic {
  public:
    bool ready = false;
    Context* context = _global_ctx;
  };
} // namespace Backend

