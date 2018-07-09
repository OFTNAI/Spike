#pragma once

// #include "Spike/Backend/Device.hpp"
#include "Spike/Backend/Macros.hpp"

#include <string>

struct DeviceParameters {
  int threads_per_block_neurons = 512;
  int threads_per_block_synapses = 512;
  int maximum_axonal_delay_in_timesteps = 0;
};

struct Context {
  DeviceParameters params;
#ifdef SPIKE_DEFAULT_BACKEND
  std::string backend = SPIKE_DEFAULT_BACKEND;
#else
  std::string backend = "Dummy";
#endif
};

extern Context* _global_ctx;

namespace Backend {
  void init_global_context();
  Context* get_current_context();
}
