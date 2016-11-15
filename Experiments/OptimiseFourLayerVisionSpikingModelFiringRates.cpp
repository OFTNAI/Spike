#include "../Optimiser/Optimiser.h"

#include "../Simulator/Simulator.h"
#include "../Models/FourLayerVisionSpikingModel.h"

#include "../SpikeAnalyser/SpikeAnalyser.h"
#include "../Helpers/TimerWithMessages.h"
#include "../Helpers/TerminalHelpers.h"
#include "../Helpers/MemoryUsage.h"

#include "cuda_profiler_api.h"


// Use the following line to compile the binary
// make FILE='OptimiseFourLayerVisionSpikingModelFiringRates' EXPERIMENT_DIRECTORY='Experiments'  model -j22




// The function which will autorun when the executable is created
int main (int argc, char *argv[]){


	TimerWithMessages * experiment_timer = new TimerWithMessages("\n");
	
	// // Parameters for OPTIMISATION + Information Analysis
	// const float optimal_max_firing_rate = 100.0f;//set if optimizing based on maxfr //Maximum rate (spikes/sec) 87 +- 46  (*1)
	// //*1 Bair, W., & Movshon, J. A. (2004).  Adaptive Temporal Integration of Motion in Direction-Selective Neurons in Macaque Visual Cortex. The Journal of Neuroscience, 24(33), 7305遯ｶ�ｿｽ7323.
	// int number_of_bins = 5;
	// bool useThresholdForMaxFR = true;
	// float max_firing_rate = optimal_max_firing_rate*presentation_time_per_stimulus_per_epoch_test;


	const float presentation_time_per_stimulus_per_epoch = 0.3;


	// SIMULATOR OPTIONS
	Simulator_Options * simulator_options = new Simulator_Options();

	simulator_options->run_simulation_general_options->presentation_time_per_stimulus_per_epoch = presentation_time_per_stimulus_per_epoch;
	simulator_options->run_simulation_general_options->number_of_epochs = 1;
	simulator_options->run_simulation_general_options->apply_stdp_to_relevant_synapses = false;
	simulator_options->run_simulation_general_options->stimulus_presentation_order_seed = 1;

	simulator_options->recording_electrodes_options->count_neuron_spikes_recording_electrodes_bool = true;

	simulator_options->stimuli_presentation_options->presentation_format = PRESENTATION_FORMAT_OBJECT_BY_OBJECT_RESET_BETWEEN_STIMULI;
	simulator_options->stimuli_presentation_options->object_order = OBJECT_ORDER_ORIGINAL;
	simulator_options->stimuli_presentation_options->transform_order = TRANSFORM_ORDER_ORIGINAL;



	// MODEL
	FourLayerVisionSpikingModel * four_layer_vision_spiking_model = new FourLayerVisionSpikingModel();
	four_layer_vision_spiking_model->SetTimestep(0.00002);
	four_layer_vision_spiking_model->high_fidelity_spike_storage = true;
	four_layer_vision_spiking_model->inputs_directory = "MatlabGaborFilter/Inputs_1T_FR_OPT/";


	// OPTIMISATION
	float upper = 150.0;
	float lower = 100.0;
	int number_of_optimisation_stages = 7;

	Optimiser* optimiser = new Optimiser(four_layer_vision_spiking_model);

	if (number_of_optimisation_stages > 0) {

		Optimiser_Options * optimisation_stage_0_options = new Optimiser_Options();
		optimisation_stage_0_options->model_pointer_to_be_optimised = &four_layer_vision_spiking_model->LBL_biological_conductance_scaling_constant_lambda_E2E_FF[0];
		optimisation_stage_0_options->synapse_bool_pointer_to_turn_on = &four_layer_vision_spiking_model->E2E_FF_SYNAPSES_ON;
		optimisation_stage_0_options->ideal_output_score = upper;
		optimisation_stage_0_options->score_to_use = SCORE_TO_USE_average_number_of_spikes_per_neuron_group_per_second_excluding_silent_neurons;

		optimiser->AddOptimisationStage(optimisation_stage_0_options, simulator_options);

	}

	// printf("optimiser->spike_analyser_from_last_optimisation_stage->average_number_of_spikes_per_neuron_group_per_second_excluding_silent_neurons[index_of_neuron_group_of_interest_for_each_optimisation_stage[optimisation_stage]]\n");

	optimiser->RunOptimisation();

	// float temp = (int)optimiser->spike_analyser_from_last_optimisation_stage->running_count_of_non_silent_neurons_per_neuron_group[0];
	// printf("optimiser->spike_analyser_from_last_optimisation_stage->running_count_of_non_silent_neurons_per_neuron_group[0]: %f\n", temp);

	float temp = optimiser->spike_analyser_from_last_optimisation_stage->average_number_of_spikes_per_neuron_group_per_second_excluding_silent_neurons[0];

	if (number_of_optimisation_stages > 1) {

		Optimiser_Options * optimisation_stage_1_options = new Optimiser_Options();
		optimisation_stage_1_options->model_pointer_to_be_optimised = &four_layer_vision_spiking_model->LBL_biological_conductance_scaling_constant_lambda_E2I_L[0];
		optimisation_stage_1_options->synapse_bool_pointer_to_turn_on = &four_layer_vision_spiking_model->E2I_L_SYNAPSES_ON;
		optimisation_stage_1_options->use_inhibitory_neurons = true;
		optimisation_stage_1_options->index_of_neuron_group_of_interest = 1;
		optimisation_stage_1_options->ideal_output_score = upper;
		optimisation_stage_1_options->score_to_use = SCORE_TO_USE_average_number_of_spikes_per_neuron_group_per_second_excluding_silent_neurons;

		optimiser->AddOptimisationStage(optimisation_stage_1_options, simulator_options);
	
	}


	if (number_of_optimisation_stages > 2) {

		Optimiser_Options * optimisation_stage_2_options = new Optimiser_Options();
		optimisation_stage_2_options->model_pointer_to_be_optimised = &four_layer_vision_spiking_model->LBL_biological_conductance_scaling_constant_lambda_I2E_L[0];
		optimisation_stage_2_options->synapse_bool_pointer_to_turn_on = &four_layer_vision_spiking_model->I2E_L_SYNAPSES_ON;
		optimisation_stage_2_options->use_inhibitory_neurons = true;
		optimisation_stage_2_options->index_of_neuron_group_of_interest = 0;
		optimisation_stage_2_options->ideal_output_score = 0.8*temp;
		optimisation_stage_2_options->positive_effect_of_postive_change_in_parameter = true;
		// optimisation_stage_2_options->score_to_use = SCORE_TO_USE_running_count_of_non_silent_neurons_per_neuron_group;
		// optimisation_stage_2_options->optimisation_minimum_error = 10.0;

		optimiser->AddOptimisationStage(optimisation_stage_2_options, simulator_options);
	
	}

	optimiser->RunOptimisation(1);

	// if (number_of_optimisation_stages > 3) {

	// 	Optimiser_Options * optimisation_stage_3_options = new Optimiser_Options();
	// 	optimisation_stage_3_options->model_pointer_to_be_optimised = &four_layer_vision_spiking_model->LBL_biological_conductance_scaling_constant_lambda_E2E_L[0];
	// 	optimisation_stage_3_options->synapse_bool_pointer_to_turn_on = &four_layer_vision_spiking_model->E2E_L_SYNAPSES_ON;
	// 	optimisation_stage_3_options->use_inhibitory_neurons = true;
	// 	optimisation_stage_3_options->index_of_neuron_group_of_interest = 0;
	// 	optimisation_stage_3_options->ideal_output_score = upper;

	// 	optimiser->AddOptimisationStage(optimisation_stage_3_options, simulator_options);
	
	// }

	// optimiser->RunOptimisation();


	// four_layer_vision_spiking_model->set_LBL_values_for_pointer_from_layer_to_layer(four_layer_vision_spiking_model->LBL_biological_conductance_scaling_constant_lambda_E2I_L[0], four_layer_vision_spiking_model->LBL_biological_conductance_scaling_constant_lambda_E2I_L, 1, 3);
	// four_layer_vision_spiking_model->set_LBL_values_for_pointer_from_layer_to_layer(four_layer_vision_spiking_model->LBL_biological_conductance_scaling_constant_lambda_I2E_L[0], four_layer_vision_spiking_model->LBL_biological_conductance_scaling_constant_lambda_I2E_L, 1, 3);
	// four_layer_vision_spiking_model->set_LBL_values_for_pointer_from_layer_to_layer(four_layer_vision_spiking_model->LBL_biological_conductance_scaling_constant_lambda_E2E_L[0], four_layer_vision_spiking_model->LBL_biological_conductance_scaling_constant_lambda_E2E_L, 1, 3);


	// if (number_of_optimisation_stages > 4) {

	// 	Optimiser_Options * optimisation_stage_4_options = new Optimiser_Options();
	// 	optimisation_stage_4_options->model_pointer_to_be_optimised = &four_layer_vision_spiking_model->LBL_biological_conductance_scaling_constant_lambda_E2E_FF[1];
	// 	optimisation_stage_4_options->synapse_bool_pointer_to_turn_on = &four_layer_vision_spiking_model->E2E_FF_SYNAPSES_ON;
	// 	optimisation_stage_4_options->use_inhibitory_neurons = true;
	// 	optimisation_stage_4_options->number_of_non_input_layers_to_simulate = 2;
	// 	optimisation_stage_4_options->index_of_neuron_group_of_interest = 2;
	// 	optimisation_stage_4_options->ideal_output_score = upper;

	// 	optimiser->AddOptimisationStage(optimisation_stage_4_options, simulator_options);
	
	// }

	// if (number_of_optimisation_stages > 5) {

	// 	Optimiser_Options * optimisation_stage_5_options = new Optimiser_Options();
	// 	optimisation_stage_5_options->model_pointer_to_be_optimised = &four_layer_vision_spiking_model->LBL_biological_conductance_scaling_constant_lambda_E2E_FF[2];
	// 	optimisation_stage_5_options->synapse_bool_pointer_to_turn_on = &four_layer_vision_spiking_model->E2E_FF_SYNAPSES_ON;
	// 	optimisation_stage_5_options->use_inhibitory_neurons = true;
	// 	optimisation_stage_5_options->number_of_non_input_layers_to_simulate = 3;
	// 	optimisation_stage_5_options->index_of_neuron_group_of_interest = 4;
	// 	optimisation_stage_5_options->ideal_output_score = upper;

	// 	optimiser->AddOptimisationStage(optimisation_stage_5_options, simulator_options);
	
	// }

	// if (number_of_optimisation_stages > 6) {

	// 	Optimiser_Options * optimisation_stage_6_options = new Optimiser_Options();
	// 	optimisation_stage_6_options->model_pointer_to_be_optimised = &four_layer_vision_spiking_model->LBL_biological_conductance_scaling_constant_lambda_E2E_FF[3];
	// 	optimisation_stage_6_options->synapse_bool_pointer_to_turn_on = &four_layer_vision_spiking_model->E2E_FF_SYNAPSES_ON;
	// 	optimisation_stage_6_options->use_inhibitory_neurons = true;
	// 	optimisation_stage_6_options->number_of_non_input_layers_to_simulate = 4;
	// 	optimisation_stage_6_options->index_of_neuron_group_of_interest = 6;
	// 	optimisation_stage_6_options->ideal_output_score = upper;

	// 	optimiser->AddOptimisationStage(optimisation_stage_6_options, simulator_options);
	
	// }


	// // four_layer_vision_spiking_model->set_LBL_values_for_pointer_from_layer_to_layer(four_layer_vision_spiking_model->LBL_biological_conductance_scaling_constant_lambda_E2E_FF[1], four_layer_vision_spiking_model->LBL_biological_conductance_scaling_constant_lambda_E2E_FF, 2, 3);

	// optimiser->RunOptimisation(4);


	// print_line_of_dashes_with_blank_lines_either_side();

	// // FINALISE MODEL + COPY TO DEVICE
	// four_layer_vision_spiking_model->number_of_non_input_layers_to_simulate = 4;
	// four_layer_vision_spiking_model->finalise_model();
	// four_layer_vision_spiking_model->copy_model_to_device();

	// // CREATE SIMULATOR
	// Simulator * simulator = new Simulator(four_layer_vision_spiking_model, simulator_options);


	// // RUN SIMULATION
	// SpikeAnalyser * spike_analyser = new SpikeAnalyser(four_layer_vision_spiking_model->spiking_neurons, four_layer_vision_spiking_model->input_spiking_neurons);
	// simulator->RunSimulation(spike_analyser);


	// // CALCULATE AVERAGES + OPTIMISATION OUTPUT SCORE
	// spike_analyser->calculate_various_neuron_spike_totals_and_averages(simulator_options->run_simulation_general_options->presentation_time_per_stimulus_per_epoch);



	/////////// END OF EXPERIMENT ///////////
	experiment_timer->stop_timer_and_log_time_and_message("Experiment Completed.", true);

	cudaProfilerStop();

	return 0;
}