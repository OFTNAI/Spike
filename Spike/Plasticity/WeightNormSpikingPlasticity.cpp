#include "WeightNormSpikingPlasticity.hpp"

WeightNormSpikingPlasticity::WeightNormSpikingPlasticity(SpikingSynapses* synapses, SpikingNeurons* neurons, SpikingNeurons* input_neurons, plasticity_parameters_struct* parameters){
	
	plasticity_parameters = (weightnorm_spiking_plasticity_parameters_struct*) parameters;
	syns = synapses;
	neurs = neurons;

}

WeightNormSpikingPlasticity::~WeightNormSpikingPlasticity(){
	free(sum_squared_afferent_values);
	free(afferent_weight_change_updater);
	free(neuron_in_plasticity_set);
}

void WeightNormSpikingPlasticity::state_update(float current_time_in_seconds, float timestep){
	backend()->weight_normalization();
}

void WeightNormSpikingPlasticity::reset_state() {
  backend()->reset_state();
}


void WeightNormSpikingPlasticity::prepare_backend_early(){
  // By making use of the neuron and synapses, I can determine which weights are contributing to the calculation to be done
  if (syns && neurs && plasticity_parameters) {
	if (plasticity_rule_id >= 0){
		sum_squared_afferent_values = (float*) malloc (neurs->total_number_of_neurons*sizeof(float));
		afferent_weight_change_updater = (float*) malloc (neurs->total_number_of_neurons*sizeof(float));
		neuron_in_plasticity_set = (bool*) malloc (neurs->total_number_of_neurons*sizeof(bool));
		// Initialize the above values to -1, Only those neurons involved in the weight normalization rule should be counted.
		for (int neuronid = 0; neuronid < neurs->total_number_of_neurons; neuronid++){
			sum_squared_afferent_values[neuronid] = 0.0f;
			afferent_weight_change_updater[neuronid] = 0.0f; 
			neuron_in_plasticity_set[neuronid] = false;
		}
		// Now for every synapse (that is a part of this stdp rule), find the post-synaptic neuron and sum the weight^2
		int num_synapses = syns->plasticity_synapse_number_per_rule[plasticity_rule_id];
		for (int synindex = 0; synindex < num_synapses; synindex++){
			int postneuron = syns->postsynaptic_neuron_indices[synindex];
			neuron_in_plasticity_set[postneuron] = true;
			sum_squared_afferent_values[postneuron] += pow(syns->synaptic_efficacies_or_weights[synindex], 2.0f);
		}
		// If there is a target total, then meet it:
		if (plasticity_parameters->settarget){
			for (int synindex = 0; synindex < num_synapses; synindex++){
				int postneuron = syns->postsynaptic_neuron_indices[synindex];
				if (postneuron >= 0){
					syns->synaptic_efficacies_or_weights[synindex] /= sqrt(sum_squared_afferent_values[postneuron]);
					syns->synaptic_efficacies_or_weights[synindex] *= plasticity_parameters->target;
				}
			}
			for (int neuronindx = 0; neuronindx < neurs->total_number_of_neurons; neuronindx++){
				sum_squared_afferent_values[neuronindx] = plasticity_parameters->target;
			}
		}
	}
  }
}

SPIKE_MAKE_INIT_BACKEND(WeightNormSpikingPlasticity);
