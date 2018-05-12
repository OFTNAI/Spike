#include "WeightNormSTDPPlasticity.hpp"

WeightNormSTDPPlasticity::WeightNormSTDPPlasticity(SpikingSynapses* synapses, SpikingNeurons* neurons, SpikingNeurons* input_neurons, plasticity_parameters_struct* parameters){

  plasticity_parameters = (weightnorm_stdp_plasticity_parameters_struct*) parameters;
  syns = synapses;
  neurs = neurons;
}

WeightNormSTDPPlasticity::~WeightNormSTDPPlasticity(){
  free(sum_squared_afferent_values);
  free(afferent_weight_change_updater);
}

void WeightNormSTDPPlasticity::state_update(float current_time_in_seconds, float timestep){
	backend()->weight_normalization();
}

void WeightNormSTDPPlasticity::reset_state() {
  backend()->reset_state();
}


void WeightNormSTDPPlasticity::prepare_backend_early(){
  // By making use of the neuron and synapses, I can determine which weights are contributing to the calculation to be done
  if (syns && neurs && plasticity_parameters) {
    if (post_neuron_set.size() > 0){
      // Find number of neurons which are part of this plasticity rule
      sum_squared_afferent_values = (float *) malloc(post_neuron_set.size() * sizeof(float));
      afferent_weight_change_updater = (float *) malloc(post_neuron_set.size() * sizeof(float));
      for (int neuronid = 0; neuronid < post_neuron_set.size(); neuronid++){
        sum_squared_afferent_values[neuronid] = 0.0f;
        afferent_weight_change_updater[neuronid] = 0.0f;
      }
      for (int synindex = 0; synindex < plastic_synapses.size(); synindex++){
        int postneuron = syns->postsynaptic_neuron_indices[synindex];
        sum_squared_afferent_values[post_neuron_conversion[postneuron]] += pow(syns->synaptic_efficacies_or_weights[synindex], 2.0f);
      }

      // If there is a target on incident weights to post-synaptic neuron:
      if (plasticity_parameters->settarget){
        for (int synindex = 0; synindex < plastic_synapses.size(); synindex++){
          int postneuron = syns->postsynaptic_neuron_indices[synindex];
          syns->synaptic_efficacies_or_weights[synindex] /= sqrt(sum_squared_afferent_values[post_neuron_conversion[postneuron]]);
          syns->synaptic_efficacies_or_weights[synindex] *= plasticity_parameters->target;
        }
      }
      for (int neuronindx = 0; neuronindx < post_neuron_set.size(); neuronindx++){
        sum_squared_afferent_values[neuronindx] = plasticity_parameters->target;
      }
    }
  }
}

SPIKE_MAKE_INIT_BACKEND(WeightNormSTDPPlasticity);
