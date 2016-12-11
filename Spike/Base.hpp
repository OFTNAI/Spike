#pragma once

#include "Spike/Backend/Context.hpp"

// Base class to be shared by front-end classes.
// Exposes _backend pointer and related methods.
class SpikeBase {
public:
  void* _backend = nullptr;

  virtual void reset_state() = 0;

  virtual void init_backend(Context* ctx = _global_ctx) = 0;

  virtual void prepare_backend_early() {}
  virtual void prepare_backend_late() {}
  void prepare_backend() {
    prepare_backend_early();
    backend()->prepare();
    prepare_backend_late();
  }
};
