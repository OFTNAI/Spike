#ifndef PoissonInputSpikingNeurons_H
#define PoissonInputSpikingNeurons_H

#include "InputSpikingNeurons.hpp"

#include "../Helpers/RandomStateManager.hpp"

struct poisson_input_spiking_neuron_parameters_struct : input_spiking_neuron_parameters_struct {
	poisson_input_spiking_neuron_parameters_struct(): rate(50.0f) { input_spiking_neuron_parameters_struct(); }

	float rate;
};

namespace Backend {
  class PoissonInputSpikingNeurons : public InputSpikingNeurons {
  public:
    virtual void reset_state() {};
  };
}

#include "Spike/Backend/Dummy/Neurons/PoissonInputSpikingNeurons.hpp"

class PoissonInputSpikingNeurons : public InputSpikingNeurons {
public:
  PoissonInputSpikingNeurons() = default;
  ~PoissonInputSpikingNeurons();

  float rate = 0;
  float * rates = NULL;
  RandomStateManager * random_state_manager = NULL;

  virtual int AddGroup(neuron_parameters_struct * group_params);
  void set_up_rates();

  // TODO:
  // virtual void update_membrane_potentials(float timestep, float current_time_in_seconds);
  virtual void init_random_state();
};

#endif
