#include "IzhikevichSpikingNeurons.hpp"
#include <stdlib.h>
#include <stdio.h>

IzhikevichSpikingNeurons::~IzhikevichSpikingNeurons() {
  if (param_a)
    free(param_a);
  if (param_b)
    free(param_b);
  if (param_d)
    free(param_d);
}


int IzhikevichSpikingNeurons::AddGroup(neuron_parameters_struct * group_params){
  int new_group_id = SpikingNeurons::AddGroup(group_params);

  izhikevich_spiking_neuron_parameters_struct * izhikevich_spiking_group_params = (izhikevich_spiking_neuron_parameters_struct*)group_params;

  param_a = (float*)realloc(param_a, (total_number_of_neurons*sizeof(float)));
  param_b = (float*)realloc(param_b, (total_number_of_neurons*sizeof(float)));
  param_d = (float*)realloc(param_d, (total_number_of_neurons*sizeof(float)));

  for (int i = total_number_of_neurons - number_of_neurons_in_new_group; i < total_number_of_neurons; i++) {
    param_a[i] = izhikevich_spiking_group_params->parama;
    param_b[i] = izhikevich_spiking_group_params->paramb;
    param_d[i] = izhikevich_spiking_group_params->paramd;
  }

  return new_group_id;
}

SPIKE_MAKE_INIT_BACKEND(IzhikevichSpikingNeurons);
