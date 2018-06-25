#ifndef PatternedPoissonInputSpikingNeurons_H
#define PatternedPoissonInputSpikingNeurons_H

// #define SILENCE_IMAGE_POISSON_INPUT_SPIKING_NEURONS_SETUP

#include "PoissonInputSpikingNeurons.hpp"

#include <vector>
#include <string>

// using namespace std;

struct patterned_poisson_input_spiking_neuron_parameters_struct : poisson_input_spiking_neuron_parameters_struct {	
};

class PatternedPoissonInputSpikingNeurons; // forward definition

namespace Backend {
  class PatternedPoissonInputSpikingNeurons : public virtual PoissonInputSpikingNeurons {
  public:
    SPIKE_ADD_BACKEND_FACTORY(PatternedPoissonInputSpikingNeurons);
    virtual void copy_rates_to_device() = 0;
  };
}

class PatternedPoissonInputSpikingNeurons : public PoissonInputSpikingNeurons {
public:
  // Constructor/Destructor
  PatternedPoissonInputSpikingNeurons();
  ~PatternedPoissonInputSpikingNeurons() override;

  SPIKE_ADD_BACKEND_GETSET(PatternedPoissonInputSpikingNeurons, PoissonInputSpikingNeurons);
  void init_backend(Context* ctx = _global_ctx) override;
 
  void state_update(float current_time_in_seconds, float timestep) override;
  void reset_stimuli();  
  int add_stimulus(float* rates, int num_rates);
  void copy_rates_to_device();

  // Variable to hold stimuli
  float* stimuli_rates;
  int total_number_of_rates;

private:
  std::shared_ptr<::Backend::PatternedPoissonInputSpikingNeurons> _backend;
};

#endif
