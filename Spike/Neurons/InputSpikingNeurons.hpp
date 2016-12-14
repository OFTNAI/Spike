#ifndef InputSpikingNeurons_H
#define InputSpikingNeurons_H

#include "SpikingNeurons.hpp"

struct input_spiking_neuron_parameters_struct : spiking_neuron_parameters_struct {
};

class InputSpikingNeurons; // forward definition

namespace Backend {
  class InputSpikingNeurons : public virtual SpikingNeurons {
  public:
  };
}

#include "Spike/Backend/Dummy/Neurons/InputSpikingNeurons.hpp"

class InputSpikingNeurons : public SpikingNeurons {
public:
  ADD_BACKEND_GETSET(InputSpikingNeurons, SpikingNeurons);
	
  int current_stimulus_index = 0;

  int total_number_of_input_stimuli = 0;
  int total_number_of_objects = 0;
  int total_number_of_transformations_per_object = 0;

  int AddGroup(neuron_parameters_struct * group_params) override;
  virtual bool stimulus_is_new_object_for_object_by_object_presentation(int stimulus_index);

private:
  ::Backend::InputSpikingNeurons* _backend = nullptr;
};

#endif
