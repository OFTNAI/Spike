#ifndef Optimiser_H
#define Optimiser_H


#include <stdio.h>

#include <math.h>  

struct Optimiser_Options {

	Optimiser_Options(): model_pointer_to_be_optimised(NULL),
						synapse_bool_pointer_to_turn_on(NULL),
						use_inhibitory_neurons(false),
						number_of_non_input_layers_to_simulate(1),
						index_of_neuron_group_of_interest(0),
						initial_optimisation_parameter_min(1.0f*pow(10, -12)),
						initial_optimisation_parameter_max(1.0*pow(10, -1)),
						ideal_output_score(100.0),
						optimisation_minimum_error(1.0)
					{}

	float* model_pointer_to_be_optimised;
	bool* synapse_bool_pointer_to_turn_on;
	bool use_inhibitory_neurons;
	int number_of_non_input_layers_to_simulate;
	int index_of_neuron_group_of_interest;
	float initial_optimisation_parameter_min;
	float initial_optimisation_parameter_max;
	float ideal_output_score;
	float optimisation_minimum_error;


};

class Optimiser{
public:
	// Constructor/Destructor
	Optimiser();
	~Optimiser();


	int number_of_optimisation_stages = 0;

	float** model_pointers_to_be_optimised_for_each_optimisation_stage;
	bool** synapse_bool_pointers_to_turn_on_for_each_optimisation_stage;
	bool* use_inhibitory_neurons_for_each_optimisation_stage;
	int* number_of_non_input_layers_to_simulate_for_each_optimisation_stage;
	int* index_of_neuron_group_of_interest_for_each_optimisation_stage;
	float* initial_optimisation_parameter_min_for_each_optimisation_stage;
	float* initial_optimisation_parameter_max_for_each_optimisation_stage;
	float* ideal_output_scores_for_each_optimisation_stage;
	float* optimisation_minimum_error_for_each_optimisation_stage;

	float* final_optimal_parameter_for_each_optimisation_stage;


	void AddOptimisationStage(Optimiser_Options * optimisation_stage_options);

};


#endif
