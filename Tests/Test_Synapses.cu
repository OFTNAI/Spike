#include "catch.hpp"


/**
		SYNAPSES.CU Test Set
**/
#include "../Synapses/Synapses.h"
#include "../Neurons/Neurons.h"
TEST_CASE("Synapses Class Tests") {
	Synapses test_synapses;
	Neurons test_neurons;
	float timestep = 0.1f;

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
	int dim2_2 = 250;

	neuron_params_2.group_shape[0] = dim1_2;
	neuron_params_2.group_shape[1] = dim2_2;

	int postsynaptic_population = test_neurons.AddGroup(&neuron_params_2);


	SECTION("Constructor Beginning with no synapses"){
		REQUIRE(test_synapses.total_number_of_synapses == 0);
	}

	SECTION("AddGroup All to All"){
		synapse_parameters_struct synapse_params;
		synapse_params.connectivity_type = CONNECTIVITY_TYPE_ALL_TO_ALL;
		test_synapses.AddGroup(
			presynaptic_population,
			postsynaptic_population,
			&test_neurons,
			&test_neurons,
			timestep,
			&synapse_params);

		// Check the created connections
		for(int i=0; i < test_neurons.group_shapes[presynaptic_population][1]; i++){
			for(int j=0; j < test_neurons.group_shapes[postsynaptic_population][1]; j++){
				REQUIRE(test_synapses.presynaptic_neuron_indices[j + i*test_neurons.group_shapes[postsynaptic_population][1]] == i);
				REQUIRE(test_synapses.postsynaptic_neuron_indices[j + i*test_neurons.group_shapes[postsynaptic_population][1]] == j + test_neurons.group_shapes[presynaptic_population][1]);
			}
		}
	}

	SECTION("AddGroup One to One"){
		synapse_parameters_struct synapse_params;
		synapse_params.connectivity_type = CONNECTIVITY_TYPE_ONE_TO_ONE;
		test_synapses.AddGroup(
			presynaptic_population,
			presynaptic_population,
			&test_neurons,
			&test_neurons,
			timestep,
			&synapse_params);

		// Check the created connections
		for(int i=0; i < test_neurons.group_shapes[presynaptic_population][1]; i++){
			REQUIRE(test_synapses.presynaptic_neuron_indices[i] == i);
			REQUIRE(test_synapses.presynaptic_neuron_indices[i] == i);
		}
	}

	SECTION("AddGroup Random"){
		synapse_parameters_struct synapse_params;
		synapse_params.connectivity_type = CONNECTIVITY_TYPE_RANDOM;
		synapse_params.random_connectivity_probability = 0.5f;
		test_synapses.AddGroup(
			presynaptic_population,
			postsynaptic_population,
			&test_neurons,
			&test_neurons,
			timestep,
			&synapse_params);

		// Check the created connections
		REQUIRE(std::abs(synapse_params.random_connectivity_probability - ((float)test_synapses.total_number_of_synapses / (float)(dim2*dim2_2))) < 0.05);
	}

	SECTION("AddGroup Gaussian Sample"){
		synapse_parameters_struct synapse_params;
		synapse_params.connectivity_type = CONNECTIVITY_TYPE_GAUSSIAN_SAMPLE;
		synapse_params.gaussian_synapses_per_postsynaptic_neuron = 25;
		synapse_params.gaussian_synapses_standard_deviation = 5;
		
		test_synapses.AddGroup(
			presynaptic_population,
			postsynaptic_population,
			&test_neurons,
			&test_neurons,
			timestep,
			&synapse_params);

		int average_standard_dev = 0;
		for(int i=0; i<dim2_2; i++){
			int num_connects = 0;
			int presynaptic_mean_expected = ((float)i / dim2_2)*dim2;
			int presynaptic_mean_actual = 0;
			int standard_deviation = 0;
			for(int j=0; j<test_synapses.total_number_of_synapses; j++){
				if (test_synapses.postsynaptic_neuron_indices[j] == 100 + i){
					num_connects++;
					presynaptic_mean_actual += test_synapses.presynaptic_neuron_indices[j];
					standard_deviation += pow((test_synapses.presynaptic_neuron_indices[j] - presynaptic_mean_expected), 2);
				}
			}
			// Check that the number of connections are correct
			REQUIRE(num_connects == synapse_params.gaussian_synapses_per_postsynaptic_neuron);
			// Check that the mean is within 10 neurons of what it should be
			REQUIRE(std::abs(presynaptic_mean_expected - presynaptic_mean_actual/num_connects) < 10);
			
			average_standard_dev += pow(((float)standard_deviation  / (num_connects - 1)), 0.5);
		}
		average_standard_dev /= dim2_2;
		REQUIRE(std::abs(average_standard_dev - synapse_params.gaussian_synapses_standard_deviation) < 0.25*synapse_params.gaussian_synapses_standard_deviation);

	}

	SECTION("AddGroup Single Connectivity"){
		synapse_parameters_struct synapse_params;
		synapse_params.connectivity_type = CONNECTIVITY_TYPE_SINGLE;
		synapse_params.pairwise_connect_presynaptic = 25;
		synapse_params.pairwise_connect_postsynaptic = 10;
		
		test_synapses.AddGroup(
			presynaptic_population,
			postsynaptic_population,
			&test_neurons,
			&test_neurons,
			timestep,
			&synapse_params);

		REQUIRE(test_synapses.presynaptic_neuron_indices[0] == 25);
		REQUIRE(test_synapses.postsynaptic_neuron_indices[0] == 10 + 100);
	}

	SECTION("Weight Range Test"){
		SECTION("Constant Weight Range"){
			synapse_parameters_struct synapse_params;
			synapse_params.connectivity_type = CONNECTIVITY_TYPE_ALL_TO_ALL;
			synapse_params.weight_range_bottom = 2.7f;
			synapse_params.weight_range_top = 2.7f;
			test_synapses.AddGroup(
				presynaptic_population,
				postsynaptic_population,
				&test_neurons,
				&test_neurons,
				timestep,
				&synapse_params);

			// Check the created connections
			for(int i=0; i < test_neurons.group_shapes[presynaptic_population][1]; i++){
				for(int j=0; j < test_neurons.group_shapes[postsynaptic_population][1]; j++){
					REQUIRE(test_synapses.synaptic_efficacies_or_weights[j + i*test_neurons.group_shapes[postsynaptic_population][1]] == synapse_params.weight_range_bottom);
				}
			}
		}
		SECTION("Uniform Weight Range"){
			synapse_parameters_struct synapse_params;
			synapse_params.connectivity_type = CONNECTIVITY_TYPE_ALL_TO_ALL;
			synapse_params.weight_range_bottom = 0.0f;
			synapse_params.weight_range_top = 10.0f;
			test_synapses.AddGroup(
				presynaptic_population,
				postsynaptic_population,
				&test_neurons,
				&test_neurons,
				timestep,
				&synapse_params);

			float mean_weight = 0.0f;
			// Check the created connections
			for(int i=0; i < test_neurons.group_shapes[presynaptic_population][1]; i++){
				for(int j=0; j < test_neurons.group_shapes[postsynaptic_population][1]; j++){
					mean_weight += test_synapses.synaptic_efficacies_or_weights[j + i*test_neurons.group_shapes[postsynaptic_population][1]];
					REQUIRE(test_synapses.synaptic_efficacies_or_weights[j + i*test_neurons.group_shapes[postsynaptic_population][1]] >= synapse_params.weight_range_bottom);
					REQUIRE(test_synapses.synaptic_efficacies_or_weights[j + i*test_neurons.group_shapes[postsynaptic_population][1]] <= synapse_params.weight_range_top);
				}
			}

			mean_weight /= test_neurons.group_shapes[presynaptic_population][1]*test_neurons.group_shapes[postsynaptic_population][1];
			// Check the mean is within 10% of what it should be
			REQUIRE(std::abs(mean_weight - synapse_params.weight_range_top / 2.0f) < 0.01*synapse_params.weight_range_top);
		}
	}

	SECTION("Input Neuron Indices Check"){
		synapse_parameters_struct synapse_params;
		synapse_params.connectivity_type = CONNECTIVITY_TYPE_ALL_TO_ALL;
		
		test_synapses.AddGroup(
			-1,
			postsynaptic_population,
			&test_neurons,
			&test_neurons,
			timestep,
			&synapse_params);

		// Check the created connections
		for(int i=0; i < test_neurons.group_shapes[presynaptic_population][1]; i++){
			for(int j=0; j < test_neurons.group_shapes[postsynaptic_population][1]; j++){
				REQUIRE(test_synapses.presynaptic_neuron_indices[j + i*test_neurons.group_shapes[postsynaptic_population][1]] == - 1 - i);
				REQUIRE(test_synapses.postsynaptic_neuron_indices[j + i*test_neurons.group_shapes[postsynaptic_population][1]] == j + test_neurons.group_shapes[presynaptic_population][1]);
			}
		}
	}
}