#ifndef Optimiser_H
#define Optimiser_H

#include "../Simulator/Simulator.hpp"
#include "../Models/FourLayerVisionSpikingModel.hpp"

#include <stdio.h>
#include <math.h> 

#include <string>
#include <stdlib.h>
#include <iostream>
#include <fstream>
using namespace std;


enum SCORE_TO_USE {
		SCORE_TO_USE_average_number_of_spikes_per_neuron_group_per_second,
		SCORE_TO_USE_max_number_of_spikes_per_neuron_group_per_second,
		SCORE_TO_USE_average_number_of_spikes_per_neuron_group_per_second_excluding_silent_neurons,
		SCORE_TO_USE_running_count_of_non_silent_neurons_per_neuron_group
	};

struct Optimiser_Options {

	Optimiser_Options(): model_pointer_to_be_optimised(NULL),
						synapse_bool_pointer_to_turn_on(NULL),
						use_inhibitory_neurons(false),
						number_of_non_input_layers_to_simulate(1),
						index_of_neuron_group_of_interest(0),
						initial_optimisation_parameter_min(1.0f*pow(10, -10)),
						initial_optimisation_parameter_max(1.0*pow(10, -2)),
						ideal_output_score(100.0),
						optimisation_minimum_error(1.0),
						positive_effect_of_postive_change_in_parameter(true),
						score_to_use(SCORE_TO_USE_average_number_of_spikes_per_neuron_group_per_second)
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
	bool positive_effect_of_postive_change_in_parameter;
	SCORE_TO_USE score_to_use;

};

class Optimiser{
public:
	// Constructor/Destructor
	Optimiser(FourLayerVisionSpikingModel* four_layer_vision_spiking_model_parameter);

	FourLayerVisionSpikingModel * four_layer_vision_spiking_model = nullptr;

	int number_of_optimisation_stages = 0;

	Simulator_Options ** simulator_options_for_each_optimisation_stage = nullptr;
	float** model_pointers_to_be_optimised_for_each_optimisation_stage = nullptr;
	bool** synapse_bool_pointers_to_turn_on_for_each_optimisation_stage = nullptr;
	bool* use_inhibitory_neurons_for_each_optimisation_stage = nullptr;
	int* number_of_non_input_layers_to_simulate_for_each_optimisation_stage = nullptr;
	int* index_of_neuron_group_of_interest_for_each_optimisation_stage = nullptr;
	float* initial_optimisation_parameter_min_for_each_optimisation_stage = nullptr;
	float* initial_optimisation_parameter_max_for_each_optimisation_stage = nullptr;
	float* ideal_output_scores_for_each_optimisation_stage = nullptr;
	float* optimisation_minimum_error_for_each_optimisation_stage = nullptr;
	bool* positive_effect_of_postive_change_in_parameter_for_each_optimisation_stage = nullptr;
	int* score_to_use_for_each_optimisation_stage = nullptr;

	float* final_optimal_parameter_for_each_optimisation_stage = nullptr;
	int* final_iteration_count_for_each_optimisation_stage = nullptr;

	SpikeAnalyser * spike_analyser_from_last_optimisation_stage = nullptr;

	void write_final_optimisation_parameters_to_file(string full_output_directory);


	void AddOptimisationStage(Optimiser_Options * optimisation_stage_options, Simulator_Options * simulator_options_parameter);
	void RunOptimisation(const int start_optimisation_stage_index = 0, bool test_last_spikes_match = false);

protected:
	void setup_optimisation_stage_specific_model_parameters(int optimisation_stage_index);

};


#endif
