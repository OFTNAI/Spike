#ifndef InputSpikingNeurons_H
#define InputSpikingNeurons_H

#include "SpikingNeurons.hpp"

struct input_spiking_neuron_parameters_struct : spiking_neuron_parameters_struct {
};

class InputSpikingNeurons; // forward definition

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
  int total_number_of_objects = 0;
  int total_number_of_transformations_per_object = 0;

  int AddGroup(neuron_parameters_struct * group_params) override;
  virtual bool stimulus_is_new_object_for_object_by_object_presentation(int stimulus_index);

private:
  std::shared_ptr<::Backend::InputSpikingNeurons> _backend;
};

#endif
