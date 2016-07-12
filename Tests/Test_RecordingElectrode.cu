#include "catch.hpp"
#include "../Helpers/CUDAErrorCheckHelpers.h"
#include <iostream>
#include <stdio.h>
#include <fstream>

/**
		RECORDINGELECTRODES.CU Test Set
**/
#include "../RecordingElectrodes/RecordingElectrodes.h"
#include "../Neurons/SpikingNeurons.h"
#include "../Synapses/SpikingSynapses.h"
TEST_CASE("RecordingElectrode") {
	SpikingSynapses test_synapses;
	SpikingNeurons test_neurons;

	// Creating the network
	// Pre-synaptic Population
	neuron_parameters_struct neuron_params_1;
	int dim1 = 1;
	int dim2 = 100;
	neuron_params_1.group_shape[0] = dim1;
	neuron_params_1.group_shape[1] = dim2;

	int presynaptic_population = test_neurons.AddGroup(&neuron_params_1);
	
	// Post-synaptic Population
	neuron_parameters_struct neuron_params_2;
	int dim1_2 = 1;
	int dim2_2 = 100;
	neuron_params_2.group_shape[0] = dim1_2;
	neuron_params_2.group_shape[1] = dim2_2;

	int postsynaptic_population = test_neurons.AddGroup(&neuron_params_2);

	// Setting times
	float timestep = 0.01f;
	float current_time = 0.09f;

	// Setting Synapses up!
	spiking_synapse_parameters_struct synapse_params;
	synapse_params.stdp_on = true;
	synapse_params.delay_range[0] = 0.1f;
	synapse_params.delay_range[1] = 0.1f;
	// Set a fixed weight range
	synapse_params.weight_range_bottom = 0.5f;
	synapse_params.weight_range_top = 0.5f;
	// Connect
	synapse_params.connectivity_type = CONNECTIVITY_TYPE_ALL_TO_ALL;
	test_synapses.AddGroup(
		presynaptic_population,
		postsynaptic_population,
		&test_neurons,
		&test_neurons,
		timestep,
		&synapse_params);

	// Checking for spikes on every timestep, with a store of length the number of neurons. Save spikes whenever they come.
	RecordingElectrodes test_record = RecordingElectrodes(&test_neurons, "test", 1, 1, 0.0f);

	// Setting up the recording electrode
	test_record.allocate_pointers_for_spike_store();
	test_record.reset_pointers_for_spike_store();

	test_record.allocate_pointers_for_spike_count();
	test_record.reset_pointers_for_spike_count();

	SECTION("Testing Initial Weight Save"){
		// Saving the initial synaptic weights to file
		test_record.write_initial_synaptic_weights_to_file(&test_synapses);
		// Check the Results dir.
		std::ifstream weightfile;
		weightfile.open("./Results/test_NetworkWeights_Initial.bin", std::ios::binary);
		
		// Check weights
		for (int i=0; i < test_synapses.total_number_of_synapses; i++){
			float test_val;
			weightfile.read((char*)&test_val, sizeof(float));
			REQUIRE(test_val == test_synapses.synaptic_efficacies_or_weights[i]);
		}

	}
}