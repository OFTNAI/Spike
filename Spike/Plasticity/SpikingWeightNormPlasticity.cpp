#include "SpikingWeightNormPlasticity.hpp"

SpikingWeightNormPlasticity::SpikingWeightNormPlasticity(SpikingSynapses* synapses, SpikingNeurons* neurons, SpikingNeurons input_neurons, plasticity_parameters_struct* parameters){
	
	plasticity_parameters = (weightnorm_spiking_plasticity_parameters_struct*) parameters;
	syns = synapses;
	neurs = neurons;

}

~SpikingWeightNormPlasticity::SpikingWeightNormPlasticity(){
	free(total_afferent_synapse_initial);
	free(afferent_synapse_changes);
}

void SpikingWeightNormPlasticity::Run_Plasticity(float current_time_in_seconds, float timestep){
	backend()->weight_normalization();
}

void SpikingWeightNormPlasticity::reset_state() {
  backend()->reset_state();
}


void SpikingWeightNormPlasticity::prepare_backend_early(){
  // By making use of the neuron and synapses, I can determine which weights are contributing to the calculation to be done
  if (syns && neurs && plasticity_parameters) {
	total_afferent_synapse_intial = (float*) malloc (neurs->total_number_of_neurons*sizeof(float));
	afferent_synapse_changes = (float*) malloc (neurs->total_number_of_neurons*sizeof(float));
	// Initialize the above values to zero
	for (int neuronid = 0; neuronid < neurs->total_number_of_neurons; neuronid++){
		total_afferent_synapse_initial[neuronid] = 0.0f;
		afferent_synapse_changes[neuronid] = 0.0f; 
	}
	// Now for every synapse (that is a part of this stdp rule), find the post-synaptic neuron and sum the weight
	int num_synapses = syns->plasticity_synapse_number_per_rule[plasticity_rule_id];
	for (int synindex = 0; synindex < num_synapses; synindex++){
		int postneuron = syns->postsynaptic_neuron_indices[synindex];
		if (postneuron >= 0){
			total_afferent_synapse_initial[postneuron] += syns->synaptic_efficacies_or_weights[synindex];
		}
	}
	// If there is a target total, then meet it:
	if (plasticity_parameters->settarget){
		for (int synindex = 0; synindex < num_synapses; synindex++){
			int postneuron = syns->postsynaptic_neuron_indices[synindex];
			if (postneuron >= 0){
				syns->synaptic_efficacies_or_weights[synindex] /= sqrt(total_afferent_synapse_initial[postneuron]);
				syns->synaptic_efficacies_or_weights[synindex] *= plasticity_parameters->target;
			}
		}
		for (int neuronindx = 0; neuronindx < neurs->total_number_of_neurons; neuronindx++){
			total_afferent_synapse_initial[neuronindx] = plasticity_parameters->target;
		}
	}
  }
}

SPIKE_MAKE_INIT_BACKEND(SpikingWeightNormPlasticity);
