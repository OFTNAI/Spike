#include "Optimiser.h"

#include "../SpikeAnalyser/SpikeAnalyser.h"
#include "../Helpers/TimerWithMessages.h"
#include "../Helpers/TerminalHelpers.h"
#include "../Helpers/MemoryUsage.h"



// Constructors
Optimiser::Optimiser(FourLayerVisionSpikingModel* four_layer_vision_spiking_model_parameter) {

	four_layer_vision_spiking_model = four_layer_vision_spiking_model_parameter;

	number_of_optimisation_stages = 0;

	simulator_options_for_each_optimisation_stage = NULL;
	model_pointers_to_be_optimised_for_each_optimisation_stage = NULL;
	synapse_bool_pointers_to_turn_on_for_each_optimisation_stage = NULL;
	use_inhibitory_neurons_for_each_optimisation_stage = NULL;
	number_of_non_input_layers_to_simulate_for_each_optimisation_stage = NULL;
	index_of_neuron_group_of_interest_for_each_optimisation_stage = NULL;
	initial_optimisation_parameter_min_for_each_optimisation_stage = NULL;
	initial_optimisation_parameter_max_for_each_optimisation_stage = NULL;
	ideal_output_scores_for_each_optimisation_stage = NULL;
	optimisation_minimum_error_for_each_optimisation_stage = NULL;

	final_optimal_parameter_for_each_optimisation_stage = NULL;

}


// Destructor
Optimiser::~Optimiser(){

}


void Optimiser::AddOptimisationStage(Optimiser_Options * optimisation_stage_options, Simulator_Options * simulator_options_parameter) {

	int new_optimisation_stage = number_of_optimisation_stages;

	number_of_optimisation_stages++;

	simulator_options_for_each_optimisation_stage = (Simulator_Options**)realloc(simulator_options_for_each_optimisation_stage, number_of_optimisation_stages*sizeof(Simulator_Options*));
	model_pointers_to_be_optimised_for_each_optimisation_stage = (float**)realloc(model_pointers_to_be_optimised_for_each_optimisation_stage, number_of_optimisation_stages*sizeof(float*));
	synapse_bool_pointers_to_turn_on_for_each_optimisation_stage = (bool**)realloc(synapse_bool_pointers_to_turn_on_for_each_optimisation_stage, number_of_optimisation_stages*sizeof(bool*));
	use_inhibitory_neurons_for_each_optimisation_stage = (bool*)realloc(use_inhibitory_neurons_for_each_optimisation_stage, number_of_optimisation_stages*sizeof(bool));
	number_of_non_input_layers_to_simulate_for_each_optimisation_stage = (int*)realloc(number_of_non_input_layers_to_simulate_for_each_optimisation_stage, number_of_optimisation_stages*sizeof(int));
	index_of_neuron_group_of_interest_for_each_optimisation_stage = (int*)realloc(index_of_neuron_group_of_interest_for_each_optimisation_stage, number_of_optimisation_stages*sizeof(int));
	initial_optimisation_parameter_min_for_each_optimisation_stage = (float*)realloc(initial_optimisation_parameter_min_for_each_optimisation_stage, number_of_optimisation_stages*sizeof(float));
	initial_optimisation_parameter_max_for_each_optimisation_stage = (float*)realloc(initial_optimisation_parameter_max_for_each_optimisation_stage, number_of_optimisation_stages*sizeof(float));
	ideal_output_scores_for_each_optimisation_stage = (float*)realloc(ideal_output_scores_for_each_optimisation_stage, number_of_optimisation_stages*sizeof(float));
	optimisation_minimum_error_for_each_optimisation_stage = (float*)realloc(optimisation_minimum_error_for_each_optimisation_stage, number_of_optimisation_stages*sizeof(float));

	simulator_options_for_each_optimisation_stage[new_optimisation_stage] = simulator_options_parameter;
	model_pointers_to_be_optimised_for_each_optimisation_stage[new_optimisation_stage] = optimisation_stage_options->model_pointer_to_be_optimised;
	synapse_bool_pointers_to_turn_on_for_each_optimisation_stage[new_optimisation_stage] = optimisation_stage_options->synapse_bool_pointer_to_turn_on;
	use_inhibitory_neurons_for_each_optimisation_stage[new_optimisation_stage] = optimisation_stage_options->use_inhibitory_neurons;
	number_of_non_input_layers_to_simulate_for_each_optimisation_stage[new_optimisation_stage] = optimisation_stage_options->number_of_non_input_layers_to_simulate;
	index_of_neuron_group_of_interest_for_each_optimisation_stage[new_optimisation_stage] = optimisation_stage_options->index_of_neuron_group_of_interest;
	initial_optimisation_parameter_min_for_each_optimisation_stage[new_optimisation_stage] = optimisation_stage_options->initial_optimisation_parameter_min;
	initial_optimisation_parameter_max_for_each_optimisation_stage[new_optimisation_stage] = optimisation_stage_options->initial_optimisation_parameter_max;
	ideal_output_scores_for_each_optimisation_stage[new_optimisation_stage] = optimisation_stage_options->ideal_output_score;
	optimisation_minimum_error_for_each_optimisation_stage[new_optimisation_stage] = optimisation_stage_options->optimisation_minimum_error;

	final_optimal_parameter_for_each_optimisation_stage = (float*)realloc(final_optimal_parameter_for_each_optimisation_stage, number_of_optimisation_stages*sizeof(float));

}


void Optimiser::RunOptimisation() {

	for (int optimisation_stage = 0; optimisation_stage < number_of_optimisation_stages; optimisation_stage++) {

		float optimisation_parameter_min = initial_optimisation_parameter_min_for_each_optimisation_stage[optimisation_stage];
		float optimisation_parameter_max = initial_optimisation_parameter_max_for_each_optimisation_stage[optimisation_stage];
		float optimisation_ideal_output_score = ideal_output_scores_for_each_optimisation_stage[optimisation_stage];

		printf("optimisation_parameter_min: %f\n", optimisation_parameter_min);
		printf("optimisation_parameter_max: %f\n", optimisation_parameter_max);

		int iteration_count_for_optimisation_stage = 0;

		float previous_optimisation_output_score = -1.0;

		while (true) {
		// for (int temp_i = 0; temp_i < 3; temp_i++) {

			iteration_count_for_optimisation_stage++;

			printf("OPTIMISATION STAGE: %d\nITERATION COUNT FOR OPTIMISATION STAGE: %d\nPREVIOUS OPTIMISATION OUTPUT SCORE: %f\n", optimisation_stage, iteration_count_for_optimisation_stage, previous_optimisation_output_score);


			// print_memory_usage();

			four_layer_vision_spiking_model->set_default_parameter_values();


			for (int optimisation_stage_index = 0; optimisation_stage_index < optimisation_stage; optimisation_stage_index++) {

				*synapse_bool_pointers_to_turn_on_for_each_optimisation_stage[optimisation_stage_index] = true;


				if (optimisation_stage_index < optimisation_stage) {

					*model_pointers_to_be_optimised_for_each_optimisation_stage[optimisation_stage_index] = final_optimal_parameter_for_each_optimisation_stage[optimisation_stage_index];
				
				}

			}


			float test_optimisation_parameter_value = (optimisation_parameter_max + optimisation_parameter_min) / 2.0;
			*model_pointers_to_be_optimised_for_each_optimisation_stage[optimisation_stage] = test_optimisation_parameter_value;

			four_layer_vision_spiking_model->INHIBITORY_NEURONS_ON = use_inhibitory_neurons_for_each_optimisation_stage[optimisation_stage];
			four_layer_vision_spiking_model->number_of_non_input_layers_to_simulate = number_of_non_input_layers_to_simulate_for_each_optimisation_stage[optimisation_stage];


			four_layer_vision_spiking_model->finalise_model();
			four_layer_vision_spiking_model->copy_model_to_device();

			// CREATE SIMULATOR
			Simulator * simulator = new Simulator(four_layer_vision_spiking_model, simulator_options_for_each_optimisation_stage[optimisation_stage]);

			// RUN SIMULATION
			SpikeAnalyser * spike_analyser = new SpikeAnalyser(four_layer_vision_spiking_model->spiking_neurons, four_layer_vision_spiking_model->input_spiking_neurons);
			simulator->RunSimulation(spike_analyser);
			spike_analyser->calculate_various_neuron_spike_totals_and_averages(simulator_options_for_each_optimisation_stage[optimisation_stage]->run_simulation_general_options->presentation_time_per_stimulus_per_epoch);


			// float optimisation_output_score = spike_analyser->max_number_of_spikes_per_neuron_group_per_second[index_of_neuron_group_of_interest_for_each_optimisation_stage[optimisation_stage]];
			float optimisation_output_score = spike_analyser->average_number_of_spikes_per_neuron_group_per_second[index_of_neuron_group_of_interest_for_each_optimisation_stage[optimisation_stage]];


			printf("TEST OPTIMISATION PARAMETER VALUE: %.16f\n", test_optimisation_parameter_value);
			printf("OPTIMISATION OUTPUTSCORE: %f\n", optimisation_output_score);

			if (optimisation_output_score <= optimisation_ideal_output_score) {

				if (optimisation_ideal_output_score - optimisation_output_score < optimisation_minimum_error_for_each_optimisation_stage[optimisation_stage]) {
					final_optimal_parameter_for_each_optimisation_stage[optimisation_stage] = test_optimisation_parameter_value;
					break;
				} else {
					optimisation_parameter_min = test_optimisation_parameter_value;
				}

			} else if (optimisation_output_score >= optimisation_ideal_output_score) {

				if (optimisation_output_score - optimisation_ideal_output_score < optimisation_minimum_error_for_each_optimisation_stage[optimisation_stage]) {
					final_optimal_parameter_for_each_optimisation_stage[optimisation_stage] = test_optimisation_parameter_value;
					break;
				} else {
					optimisation_parameter_max = test_optimisation_parameter_value;
				}


			}


			print_line_of_dashes_with_blank_lines_either_side();

			delete simulator;
			delete spike_analyser;

			previous_optimisation_output_score = optimisation_output_score;


			// print_memory_usage();

		}

		printf("final_optimal_parameter_for_each_optimisation_stage[optimisation_stage]: %.12f\n", final_optimal_parameter_for_each_optimisation_stage[optimisation_stage]);
		printf("iteration_count_for_optimisation_stage: %d\n", iteration_count_for_optimisation_stage);

	}

}
