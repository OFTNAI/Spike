#include "InputSpikingNeurons.hpp"
#include <stdlib.h>
#include <algorithm>
#include "../Helpers/TerminalHelpers.hpp"

int InputSpikingNeurons::AddGroup(neuron_parameters_struct * group_params){
  int new_group_id = SpikingNeurons::AddGroup(group_params);
  return (-1*new_group_id - 1);
};

void InputSpikingNeurons::reset_state() {
  SpikingNeurons::reset_state();
  backend->reset_state();
}

bool InputSpikingNeurons::stimulus_is_new_object_for_object_by_object_presentation(int stimulus_index) {
  return true;
}


