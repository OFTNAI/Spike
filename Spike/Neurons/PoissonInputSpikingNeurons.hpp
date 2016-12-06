#ifndef PoissonInputSpikingNeurons_H
#define PoissonInputSpikingNeurons_H

#include "InputSpikingNeurons.hpp"

#include "../Helpers/RandomStateManager.hpp"

struct poisson_input_spiking_neuron_parameters_struct : input_spiking_neuron_parameters_struct {
	poisson_input_spiking_neuron_parameters_struct(): rate(50.0f) { input_spiking_neuron_parameters_struct(); }

	float rate;
};

class PoissonInputSpikingNeurons; // forward definition

namespace Backend {
  class PoissonInputSpikingNeurons : public virtual InputSpikingNeurons {
  public:
    ADD_FRONTEND_GETTER(PoissonInputSpikingNeurons);
    virtual void reset_state() {};
  };
}

#include "Spike/Backend/Dummy/Neurons/PoissonInputSpikingNeurons.hpp"
#ifdef SPIKE_WITH_CUDA
#include "Spike/Backend/CUDA/Neurons/PoissonInputSpikingNeurons.hpp"
#endif


class PoissonInputSpikingNeurons : public InputSpikingNeurons {
public:
  PoissonInputSpikingNeurons() = default;
  ~PoissonInputSpikingNeurons();

  ADD_BACKEND_GETTER(PoissonInputSpikingNeurons);
  
  float rate = 0;
  float * rates = nullptr;
  RandomStateManager * random_state_manager = nullptr;

  virtual int AddGroup(neuron_parameters_struct * group_params);
  void set_up_rates();

  // TODO:
  // virtual void update_membrane_potentials(float timestep, float current_time_in_seconds);
  virtual void init_random_state();
};

#endif
