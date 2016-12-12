#pragma once

#include "Spike/Backend/Backend.hpp"

// Base class to be shared by front-end classes.
// Exposes _backend pointer and related methods.
class SpikeBase {
public:
  virtual void reset_state() = 0;

  virtual void init_backend(Context* ctx = _global_ctx) = 0;
  void backend(void*) {} // Dummy _backend setter

  virtual void prepare_backend_early() {}
  virtual void prepare_backend_late() {}
};
