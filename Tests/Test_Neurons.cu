#include "../catch.hpp"

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