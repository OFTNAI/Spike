#include "WeightNormSpikingPlasticity.hpp"

WeightNormSpikingPlasticity::WeightNormSpikingPlasticity(SpikingSynapses* synapses, SpikingNeurons* neurons, SpikingNeurons* input_neurons, plasticity_parameters_struct* parameters){
	
	plasticity_parameters = (weightnorm_spiking_plasticity_parameters_struct*) parameters;
	syns = synapses;
	neurs = neurons;

}

WeightNormSpikingPlasticity::~WeightNormSpikingPlasticity(){
	free(total_afferent_synapse_initial);
	free(afferent_synapse_changes);
	free(neuron_in_plasticity_set);
}

void WeightNormSpikingPlasticity::Run_Plasticity(float current_time_in_seconds, float timestep){
	backend()->weight_normalization();
}

void WeightNormSpikingPlasticity::reset_state() {
  backend()->reset_state();
}


void WeightNormSpikingPlasticity::prepare_backend_early(){
  // By making use of the neuron and synapses, I can determine which weights are contributing to the calculation to be done
  if (syns && neurs && plasticity_parameters) {
	total_afferent_synapse_initial = (float*) malloc (neurs->total_number_of_neurons*sizeof(float));
	afferent_synapse_changes = (float*) malloc (neurs->total_number_of_neurons*sizeof(float));
	neuron_in_plasticity_set = (bool*) malloc (neurs->total_number_of_neurons*sizeof(bool));
	// Initialize the above values to -1, Only those neurons involved in the weight normalization rule should be counted.
	for (int neuronid = 0; neuronid < neurs->total_number_of_neurons; neuronid++){
		total_afferent_synapse_initial[neuronid] = 0.0f;
		afferent_synapse_changes[neuronid] = 0.0f; 
		neuron_in_plasticity_set[neuronid] = false;
	}
	// Now for every synapse (that is a part of this stdp rule), find the post-synaptic neuron and sum the weight
	int num_synapses = syns->plasticity_synapse_number_per_rule[plasticity_rule_id];
	for (int synindex = 0; synindex < num_synapses; synindex++){
		int postneuron = syns->postsynaptic_neuron_indices[synindex];
		neuron_in_plasticity_set[postneuron] = true;
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

SPIKE_MAKE_INIT_BACKEND(WeightNormSpikingPlasticity);
