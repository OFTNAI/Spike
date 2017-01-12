#pragma once

#include "Spike/Backend/Device.hpp"

#include <string>

struct DeviceParameters {
  bool high_fidelity_spike_storage = false;
  int threads_per_block_neurons = 512;
  int threads_per_block_synapses = 512;
  int maximum_axonal_delay_in_timesteps = 0;
};

class Context {
public:
  Backend::Device device;
  DeviceParameters params;
  std::string backend = "Dummy";
};

extern Context* _global_ctx;

namespace Backend {
  void init_global_context();
  Context* get_current_context();
}
