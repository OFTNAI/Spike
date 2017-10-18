#ifndef GeneralPoissonInputSpikingNeurons_H
#define GeneralPoissonInputSpikingNeurons_H

// #define SILENCE_IMAGE_POISSON_INPUT_SPIKING_NEURONS_SETUP

#include "PoissonInputSpikingNeurons.hpp"

#include <vector>
#include <string>

// using namespace std;

struct general_poisson_input_spiking_neuron_parameters_struct : poisson_input_spiking_neuron_parameters_struct {	
};

class GeneralPoissonInputSpikingNeurons; // forward definition

namespace Backend {
  class GeneralPoissonInputSpikingNeurons : public virtual PoissonInputSpikingNeurons {
  public:
    SPIKE_ADD_BACKEND_FACTORY(GeneralPoissonInputSpikingNeurons);
    virtual void copy_rates_to_device() = 0;
  };
}

class GeneralPoissonInputSpikingNeurons : public PoissonInputSpikingNeurons {
public:
  // Constructor/Destructor
  GeneralPoissonInputSpikingNeurons();
  ~GeneralPoissonInputSpikingNeurons() override;

  SPIKE_ADD_BACKEND_GETSET(GeneralPoissonInputSpikingNeurons, PoissonInputSpikingNeurons);
  void init_backend(Context* ctx = _global_ctx) override;
 
  
  void state_update(float current_time_in_seconds, float timestep) override;
  void add_stimulus(float* rates, int num_rates);
  void copy_rates_to_device();

  // Variable to hold stimuli
  float* stimuli_rates;
  int total_number_of_rates;

private:
  std::shared_ptr<::Backend::GeneralPoissonInputSpikingNeurons> _backend;
};

#endif
