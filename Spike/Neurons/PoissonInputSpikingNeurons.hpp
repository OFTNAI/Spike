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
    SPIKE_ADD_FRONTEND_GETTER(PoissonInputSpikingNeurons);
  };
}

#include "Spike/Backend/Dummy/Neurons/PoissonInputSpikingNeurons.hpp"
#ifdef SPIKE_WITH_CUDA
#include "Spike/Backend/CUDA/Neurons/PoissonInputSpikingNeurons.hpp"
#endif


class PoissonInputSpikingNeurons : public InputSpikingNeurons {
public:
  PoissonInputSpikingNeurons() = default;
  ~PoissonInputSpikingNeurons() override;

  void init_backend(Context* ctx) override;
  SPIKE_ADD_BACKEND_GETSET(PoissonInputSpikingNeurons, InputSpikingNeurons);
  
  float rate = 0;
  float * rates = nullptr;
  RandomStateManager * random_state_manager = nullptr;

  void prepare_backend_early();

  void set_up_rates();

  void init_random_state(bool force=false);
  
  int AddGroup(neuron_parameters_struct * group_params) override;

  void update_membrane_potentials(float timestep, float current_time_in_seconds) override;

private:
  std::shared_ptr<::Backend::PoissonInputSpikingNeurons> _backend;
};

#endif
