#ifndef InputSpikingNeurons_H
#define InputSpikingNeurons_H

#include "SpikingNeurons.hpp"

struct input_spiking_neuron_parameters_struct : spiking_neuron_parameters_struct {
};

namespace Backend {
  class InputSpikingNeurons : public SpikingNeurons {
  public:
    virtual void reset() {};
  };
} 

class InputSpikingNeurons : public SpikingNeurons {
public:
  Backend::InputSpikingNeurons backend;
	
  int current_stimulus_index = 0;

  int total_number_of_input_stimuli = 0;
  int total_number_of_objects = 0;
  int total_number_of_transformations_per_object = 0;

  virtual int AddGroup(neuron_parameters_struct * group_params);
  virtual bool stimulus_is_new_object_for_object_by_object_presentation(int stimulus_index);
  virtual void reset();
};

#endif
