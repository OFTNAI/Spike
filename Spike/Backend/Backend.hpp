#pragma once

#include "Context.hpp"

namespace Backend {
  class SpikeBackendBase {
  public:
    bool ready = false;
    Context* context = _global_ctx;

    void* _frontend = nullptr;

    virtual void reset_state() = 0;
    virtual void prepare() = 0;
  };
} // namespace Backend

