#include "../catch.hpp"

/**
		NEURONS.CU Test Set
**/
#include "../Neurons/Neurons.h"
TEST_CASE("Neurons Class") {
	// Create an instance of the neuron class
	Neurons test_neuron;

	SECTION("Constructor begins with zero neurons/groups") {
		REQUIRE(test_neuron.total_number_of_neurons == 0);
		REQUIRE(test_neuron.total_number_of_groups == 0);
	}
}