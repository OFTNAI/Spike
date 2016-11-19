// STDP Class Header
// STDP.h
//
//	Author: Nasir Ahmad
//	Date: 23/06/2016

#ifndef STDP_H
#define STDP_H

// Get Synapses & Neurons Class
#include "../Synapses/SpikingSynapses.hpp"
#include "../Neurons/SpikingNeurons.hpp"

// stdlib allows random numbers
#include <stdlib.h>
// Input Output
#include <stdio.h>
// allows maths
#include <math.h>

//CUDA #include <cuda.h>


// STDP Parameters
struct stdp_parameters_struct {
	stdp_parameters_struct() {}
};


class STDP {
public:
  // Constructor/Destructor
  STDP();
  ~STDP();

  void prepare_backend(Context* ctx) {
    printf("TODO: Synapse prepare_backend\n");
  }

  void reset_state() {
    printf("TODO: Synapse reset_state\n");
  }

  // Set STDP Parameters
  virtual void Set_STDP_Parameters(SpikingSynapses* synapses, SpikingNeurons* neurons, SpikingNeurons* input_neurons, stdp_parameters_struct* stdp_parameters);
  // Initialize STDP
  virtual void allocate_device_pointers();
  // STDP
  virtual void Run_STDP(float* d_last_spike_time_of_each_neuron, float current_time_in_seconds, float timestep);
  // Reset
  virtual void reset_STDP_activities();
};

#endif
