#include "../catch.hpp"
#include "../Helpers/CUDAErrorCheckHelpers.h"

/**
		NEURONS.CU Test Set
**/
#include "../Neurons/Neurons.h"
TEST_CASE("Neurons Class") {
	// Create an instance of the neuron class
	Neurons test_neurons;

	SECTION("Constructor begins with zero neurons/groups") {
		REQUIRE(test_neurons.total_number_of_neurons == 0);
		REQUIRE(test_neurons.total_number_of_groups == 0);
	}

	SECTION("Adding a Group"){
		neuron_parameters_struct params;
		int dim1 = 1;
		int dim2 = 10;

		params.group_shape[0] = dim1;
		params.group_shape[1] = dim2;

		int ID = test_neurons.AddGroup(&params);

		REQUIRE(ID == 0);
		REQUIRE(test_neurons.total_number_of_neurons == dim1*dim2);
		REQUIRE(test_neurons.start_neuron_indices_for_each_group[0] == 0);
		REQUIRE(test_neurons.last_neuron_indices_for_each_group[0] == (dim1*dim2 - 1));
		REQUIRE(test_neurons.group_shapes[0][0] == dim1);
		REQUIRE(test_neurons.group_shapes[0][1] == dim2);
	}
}


/**
		SPIKINGNEURONS.CU Test Set
**/
#include "../Neurons/SpikingNeurons.h"
TEST_CASE("Spiking Neurons Class") {
	// Create an instance of the neuron class
	SpikingNeurons test_neurons;

	spiking_neuron_parameters_struct params;
	int dim1 = 1;
	int dim2 = 10;
	float resting_pot = -50.0f;
	float threshold = 50.0f;

	params.group_shape[0] = dim1;
	params.group_shape[1] = dim2;
	params.resting_potential_v0 = resting_pot;
	params.threshold_for_action_potential_spike = threshold;

	test_neurons.AddGroup(&params);

	SECTION("AddGroup Setting Spiking Parameters") {
		for (int i=0; i < dim1; i++){
			for (int j=0; j < dim2; j ++){
				REQUIRE(test_neurons.after_spike_reset_membrane_potentials_c[j + i*dim2] == resting_pot);
				REQUIRE(test_neurons.thresholds_for_action_potential_spikes[j + i*dim2] == threshold);
			}
		}
	}


	// Check for Neuron Spikes Kernel
	SECTION("Low Fidelity check_for_neuron_spikes Kernel Check") {
		// Setting up the Spiking Neurons
		// No high fidelity
		test_neurons.allocate_device_pointers(0, false);
		// Selecting some indices to set to fire
		int indices[3];
		indices[0] = 1; indices[1] = 3; indices[2] = 5;
		for (int i=0; i < 3; i++){
			test_neurons.after_spike_reset_membrane_potentials_c[indices[i]] = 60.0f;
		}
		// Copying the given arrays to the GPU
		test_neurons.reset_neurons();
		// Carrying out the spike check
		float current_time = 0.9f;
		float timestep = 0.1f;
		test_neurons.check_for_neuron_spikes(current_time, timestep);
		// Copy back the correct array
		float* neuron_spike_times;
		neuron_spike_times = (float*)malloc(sizeof(float)*test_neurons.total_number_of_neurons);
		CudaSafeCall(cudaMemcpy(neuron_spike_times, test_neurons.d_last_spike_time_of_each_neuron, sizeof(float)*test_neurons.total_number_of_neurons, cudaMemcpyDeviceToHost));
		// Check the neuron spikes array
		for (int i=0; i < test_neurons.total_number_of_neurons; i++){
			if (i == indices[0] || i == indices[1] || i == indices[2]){
				REQUIRE(neuron_spike_times[i] == current_time);
			} else {
				REQUIRE(neuron_spike_times[i] == -1000.0f);
			}
		}
	}


}




