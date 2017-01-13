#include "Optimiser.hpp"

#include "../SpikeAnalyser/SpikeAnalyser.hpp"
#include "../Helpers/TimerWithMessages.hpp"
#include "../Helpers/TerminalHelpers.hpp"
#include "../Helpers/Memory.hpp"



// Constructors
Optimiser::Optimiser(FourLayerVisionSpikingModel* four_layer_vision_spiking_model_parameter) {

	four_layer_vision_spiking_model = four_layer_vision_spiking_model_parameter;

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
	positive_effect_of_postive_change_in_parameter_for_each_optimisation_stage = (bool*)realloc(positive_effect_of_postive_change_in_parameter_for_each_optimisation_stage, number_of_optimisation_stages*sizeof(bool));
	score_to_use_for_each_optimisation_stage = (int*)realloc(score_to_use_for_each_optimisation_stage, number_of_optimisation_stages*sizeof(int));

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
	positive_effect_of_postive_change_in_parameter_for_each_optimisation_stage[new_optimisation_stage] = optimisation_stage_options->positive_effect_of_postive_change_in_parameter;
	score_to_use_for_each_optimisation_stage[new_optimisation_stage] = (int)optimisation_stage_options->score_to_use;

	final_optimal_parameter_for_each_optimisation_stage = (float*)realloc(final_optimal_parameter_for_each_optimisation_stage, number_of_optimisation_stages*sizeof(float));
	final_iteration_count_for_each_optimisation_stage = (int*)realloc(final_iteration_count_for_each_optimisation_stage, number_of_optimisation_stages*sizeof(int));

}


void Optimiser::RunOptimisation(int start_optimisation_stage_index, bool test_last_spikes_match) {

	for (int optimisation_stage = start_optimisation_stage_index; optimisation_stage < number_of_optimisation_stages; optimisation_stage++) {

		float optimisation_parameter_min = initial_optimisation_parameter_min_for_each_optimisation_stage[optimisation_stage];
		float optimisation_parameter_max = initial_optimisation_parameter_max_for_each_optimisation_stage[optimisation_stage];
		float optimisation_ideal_output_score = ideal_output_scores_for_each_optimisation_stage[optimisation_stage];

		int iteration_count_for_optimisation_stage = -1;;

		while (true) {

			// printf("Backend::total_memory(): %lu\n", Backend::total_memory());

			iteration_count_for_optimisation_stage++;

			print_line_of_dashes_with_blank_lines_either_side();
			
			setup_optimisation_stage_specific_model_parameters(optimisation_stage);
		
			float test_optimisation_parameter_value = (optimisation_parameter_max + optimisation_parameter_min) / 2.0;
			*model_pointers_to_be_optimised_for_each_optimisation_stage[optimisation_stage] = test_optimisation_parameter_value;
			if (test_last_spikes_match) *model_pointers_to_be_optimised_for_each_optimisation_stage[optimisation_stage] = 0.1;

			printf("OPTIMISATION ITERATION BEGINNING... \nOptimisation Stage: %d\nIteration Count for Optimisation Stage: %d\nNew Test Optimisaton Parameter: %f\n", optimisation_stage, iteration_count_for_optimisation_stage, test_optimisation_parameter_value);

			print_line_of_dashes_with_blank_lines_either_side();

			// FINALISE MODEL + COPY TO DEVICE
			four_layer_vision_spiking_model->finalise_model();

			
			simulator_options_for_each_optimisation_stage[optimisation_stage]->run_simulation_general_options->delete_spike_analyser_on_simulator_destruction = !test_last_spikes_match;


			// CREATE SIMULATOR
			Simulator * simulator = new Simulator(four_layer_vision_spiking_model, simulator_options_for_each_optimisation_stage[optimisation_stage]);

			// RUN SIMULATION
			simulator->RunSimulation();

			// CALCULATE AVERAGES + OPTIMISATION OUTPUT SCORE
			SpikeAnalyser *new_spike_analyser = simulator->spike_analyser;
			new_spike_analyser->calculate_various_neuron_spike_totals_and_averages(simulator_options_for_each_optimisation_stage[optimisation_stage]->run_simulation_general_options->presentation_time_per_stimulus_per_epoch);

			// Temp testing

			if (test_last_spikes_match) {

				printf("TEST: Comparte spike totals to last optimisation iteration\n");

				if (iteration_count_for_optimisation_stage > 0) {
					int total_wrong_count = 0;
					for (int stimulus_index = 0; stimulus_index < four_layer_vision_spiking_model->input_spiking_neurons->total_number_of_input_stimuli; stimulus_index++) {
						for (int neuron_index = 0; neuron_index < 4096; neuron_index++) {
							if (new_spike_analyser->per_stimulus_per_neuron_spike_counts[stimulus_index][neuron_index] != spike_analyser_from_last_optimisation_stage->per_stimulus_per_neuron_spike_counts[stimulus_index][neuron_index]) {
							// if (spike_analyser_from_last_optimisation_stage->per_stimulus_per_neuron_spike_counts[stimulus_index][neuron_index]) {
								printf("new_spike_analyser->per_stimulus_per_neuron_spike_counts[%d][%d]: %d\n", stimulus_index, neuron_index, new_spike_analyser->per_stimulus_per_neuron_spike_counts[stimulus_index][neuron_index]);
								printf("spike_analyser_from_last_optimisation_stage->per_stimulus_per_neuron_spike_counts[%d][%d]: %d\n", stimulus_index, neuron_index, spike_analyser_from_last_optimisation_stage->per_stimulus_per_neuron_spike_counts[stimulus_index][neuron_index]);
								total_wrong_count++;
								printf("neuron_index: %d\n", neuron_index);
								print_message_and_exit("Test failure: Spike totals do not match");
								// if (2329 == neuron_index) {
								// 	continue;
								// } else {
								// 	print_message_and_exit("something else!!");
								// }
							}		
						}
					}
					printf("total_wrong_count: %d\n", total_wrong_count);
				}

				delete spike_analyser_from_last_optimisation_stage;
				spike_analyser_from_last_optimisation_stage = new_spike_analyser;

			}

			float optimisation_output_score = 0.0;

			int index_of_neuron_group_of_interest = index_of_neuron_group_of_interest_for_each_optimisation_stage[optimisation_stage];

			switch(score_to_use_for_each_optimisation_stage[optimisation_stage]) {

				case SCORE_TO_USE_average_number_of_spikes_per_neuron_group_per_second:
					optimisation_output_score = new_spike_analyser->average_number_of_spikes_per_neuron_group_per_second[index_of_neuron_group_of_interest];
					break;

				case SCORE_TO_USE_max_number_of_spikes_per_neuron_group_per_second:
					optimisation_output_score = new_spike_analyser->max_number_of_spikes_per_neuron_group_per_second[index_of_neuron_group_of_interest];
					break;

				case SCORE_TO_USE_average_number_of_spikes_per_neuron_group_per_second_excluding_silent_neurons:
					optimisation_output_score = new_spike_analyser->average_number_of_spikes_per_neuron_group_per_second_excluding_silent_neurons[index_of_neuron_group_of_interest];
					break;

				case SCORE_TO_USE_running_count_of_non_silent_neurons_per_neuron_group:
					// optimisation_output_score = new_spike_analyser->running_count_of_non_silent_neurons_per_neuron_group[index_of_neuron_group_of_interest];
					optimisation_output_score = (float)(new_spike_analyser->total_number_of_spikes_per_neuron_group[0])/356.0;

			}

			printf("OPTIMISATION ITERATION COMPLETED...\nTest Optimisation Parameter Value: %.16f\nOptimisation Output Score: %f\nOptimisation Ideal Output Score: %f\n", test_optimisation_parameter_value, optimisation_output_score, optimisation_ideal_output_score);


			float difference_between_ideal_score_and_output_score = optimisation_ideal_output_score - optimisation_output_score; // Supposing the function we are trying to optimise is monotonic, the sign of this value gives the direction that the optimisation must move in.

			if (fabs(difference_between_ideal_score_and_output_score) < optimisation_minimum_error_for_each_optimisation_stage[optimisation_stage]) {
			
				final_optimal_parameter_for_each_optimisation_stage[optimisation_stage] = test_optimisation_parameter_value;
				break;
			
			}

			float effect_direction_factor = positive_effect_of_postive_change_in_parameter_for_each_optimisation_stage[optimisation_stage] ? 1.0 : -1.0;

			if (effect_direction_factor * difference_between_ideal_score_and_output_score > 0) {
			
				optimisation_parameter_min = test_optimisation_parameter_value;
			
			} else {
			
				optimisation_parameter_max = test_optimisation_parameter_value;
			
			}

					
			delete simulator;

			print_line_of_dashes_with_blank_lines_either_side();
			printf("Backend::memory_free_bytes(): %lu", Backend::memory_free_bytes());
			print_line_of_dashes_with_blank_lines_either_side();

		}

		print_line_of_dashes_with_blank_lines_either_side();
		printf("Backend::memory_free_bytes(): %lu", Backend::memory_free_bytes());
		print_line_of_dashes_with_blank_lines_either_side();

		final_iteration_count_for_each_optimisation_stage[optimisation_stage] = iteration_count_for_optimisation_stage; 

		print_line_of_dashes_with_blank_lines_either_side();
		
		printf("FINAL OPTIMAL PARAMETER FOR OPTIMISATION STAGE %d: %.12f\n", optimisation_stage, final_optimal_parameter_for_each_optimisation_stage[optimisation_stage]);
		printf("TOTAL OPTIMISATION ITERATIONS FOR OPTIMISATION STAGE %d: %d\n", optimisation_stage, final_iteration_count_for_each_optimisation_stage[optimisation_stage]);

	}

	printf("\n");
	for (int optimisation_stage = 0; optimisation_stage < number_of_optimisation_stages; optimisation_stage++) {

		printf("FINAL OPTIMAL PARAMETER FOR OPTIMISATION STAGE %d: %.12f\n", optimisation_stage, final_optimal_parameter_for_each_optimisation_stage[optimisation_stage]);
		printf("TOTAL OPTIMISATION ITERATIONS FOR OPTIMISATION STAGE %d: %d\n", optimisation_stage, final_iteration_count_for_each_optimisation_stage[optimisation_stage]);

	}

		print_line_of_dashes_with_blank_lines_either_side();

}


void Optimiser::setup_optimisation_stage_specific_model_parameters(int optimisation_stage_index) {

	*synapse_bool_pointers_to_turn_on_for_each_optimisation_stage[optimisation_stage_index] = true;
	*model_pointers_to_be_optimised_for_each_optimisation_stage[optimisation_stage_index] = final_optimal_parameter_for_each_optimisation_stage[optimisation_stage_index];

	if (use_inhibitory_neurons_for_each_optimisation_stage[optimisation_stage_index]) {

		four_layer_vision_spiking_model->INHIBITORY_NEURONS_ON = use_inhibitory_neurons_for_each_optimisation_stage[optimisation_stage_index];

	}
	if (number_of_non_input_layers_to_simulate_for_each_optimisation_stage[optimisation_stage_index] > four_layer_vision_spiking_model->number_of_non_input_layers_to_simulate) {

		four_layer_vision_spiking_model->number_of_non_input_layers_to_simulate = number_of_non_input_layers_to_simulate_for_each_optimisation_stage[optimisation_stage_index];

	} 

}


void Optimiser::write_final_optimisation_parameters_to_file(string full_output_directory) {

	for (int optimisation_stage = 0; optimisation_stage < number_of_optimisation_stages; optimisation_stage++) {

		string file_IDs = full_output_directory + to_string(optimisation_stage);

		ofstream spikeidfile, spiketimesfile;
		spikeidfile.open((file_IDs + ".txt"), ios::out | ios::binary);
		// spiketimesfile.open((file_Times + ".txt"), ios::out | ios::binary);

		spikeidfile << to_string(final_optimal_parameter_for_each_optimisation_stage[optimisation_stage]) << endl;

		spikeidfile.close();


	}

}

// void Optimiser::set_final_optimised_parameters_network() {

// 	for (int optimisation_stage_index = 0; optimisation_stage_index < number_of_optimisation_stages; optimisation_stage_index++) {
		
// 		setup_optimisation_stage_specific_model_parameters(optimisation_stage_index);

// 	}

// }
