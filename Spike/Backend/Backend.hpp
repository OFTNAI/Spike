#pragma once

#include "Context.hpp"

namespace Backend {
  class Generic {
  public:
    bool ready = false;
    Context* context;
  };
} // namespace Backend

