#pragma once

#include "Spike/Backend/Context.hpp"

// Base class to be shared by all front-end classes.
// Exposes _backend pointer and related methods.
class SpikeBase {
public:
  void* _backend = nullptr;
  virtual void prepare_backend(Context* ctx = _global_ctx) = 0;
  inline void prepare_backend_extra() {}
  virtual void reset_state() = 0;
};
