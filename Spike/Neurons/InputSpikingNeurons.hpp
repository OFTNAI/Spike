#ifndef InputSpikingNeurons_H
#define InputSpikingNeurons_H


class InputSpikingNeurons; // forward definition
struct input_spiking_neuron_parameters_struct;

#include "SpikingNeurons.hpp"

struct input_spiking_neuron_parameters_struct : spiking_neuron_parameters_struct {
};

namespace Backend {
  class InputSpikingNeurons : public virtual SpikingNeurons {
  public:
    SPIKE_ADD_BACKEND_FACTORY(InputSpikingNeurons);
  };
}

class InputSpikingNeurons : public SpikingNeurons {
public:
  SPIKE_ADD_BACKEND_GETSET(InputSpikingNeurons, SpikingNeurons);
  
  int current_stimulus_index = 0;
  int total_number_of_input_stimuli = 0;
  virtual void select_stimulus(int stimulus_index);
  int AddGroup(neuron_parameters_struct * group_params) override;

private:
  std::shared_ptr<::Backend::InputSpikingNeurons> _backend;
};

#endif
