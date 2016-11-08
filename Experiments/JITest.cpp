#include "../Optimiser/Optimiser.h"

#include "../Simulator/Simulator.h"
#include "../Models/FourLayerVisionSpikingModel.h"

#include "../SpikeAnalyser/SpikeAnalyser.h"
#include "../Helpers/TimerWithMessages.h"
#include "../Helpers/TerminalHelpers.h"
#include "../Helpers/MemoryUsage.h"

#include "cuda_profiler_api.h"


// Use the following line to compile the binary
// make FILE='JITest' EXPERIMENT_DIRECTORY='Experiments'  model -j8


// The function which will autorun when the executable is created
int main (int argc, char *argv[]){

	print_line_of_dashes_with_blank_lines_either_side();

	TimerWithMessages * experiment_timer = new TimerWithMessages();

	
	// // Parameters for OPTIMISATION + Information Analysis
	// const float optimal_max_firing_rate = 100.0f;//set if optimizing based on maxfr //Maximum rate (spikes/sec) 87 +- 46  (*1)
	// //*1 Bair, W., & Movshon, J. A. (2004).  Adaptive Temporal Integration of Motion in Direction-Selective Neurons in Macaque Visual Cortex. The Journal of Neuroscience, 24(33), 7305遯ｶ�ｿｽ7323.
	// int number_of_bins = 5;
	// bool useThresholdForMaxFR = true;
	// float max_firing_rate = optimal_max_firing_rate*presentation_time_per_stimulus_per_epoch_test;


	const float presentation_time_per_stimulus_per_epoch = 0.01;


	// SIMULATOR OPTIONS
	Simulator_Options * simulator_options = new Simulator_Options();

	simulator_options->run_simulation_general_options->presentation_time_per_stimulus_per_epoch = presentation_time_per_stimulus_per_epoch;
	simulator_options->run_simulation_general_options->number_of_epochs = 1;
	simulator_options->run_simulation_general_options->apply_stdp_to_relevant_synapses = false;
	simulator_options->run_simulation_general_options->stimulus_presentation_order_seed = 1;

	simulator_options->recording_electrodes_options->count_neuron_spikes_recording_electrodes_bool = true;

	simulator_options->stimuli_presentation_options->presentation_format = PRESENTATION_FORMAT_OBJECT_BY_OBJECT_RESET_BETWEEN_OBJECTS;
	simulator_options->stimuli_presentation_options->object_order = OBJECT_ORDER_ORIGINAL;
	simulator_options->stimuli_presentation_options->transform_order = TRANSFORM_ORDER_ORIGINAL;



	// MODEL
	FourLayerVisionSpikingModel * four_layer_vision_spiking_model = new FourLayerVisionSpikingModel();
	four_layer_vision_spiking_model->SetTimestep(0.00002);
	four_layer_vision_spiking_model->high_fidelity_spike_storage = true;

	// OPTIMISATION
	float upper = 20.0;
	float lower = 10.0;
	int number_of_optimisation_stages = 5;

	Optimiser* optimiser = new Optimiser(four_layer_vision_spiking_model);

	if (number_of_optimisation_stages >= 1) {

		Optimiser_Options * optimisation_stage_1_options = new Optimiser_Options();
		optimisation_stage_1_options->model_pointer_to_be_optimised = &four_layer_vision_spiking_model->LBL_biological_conductance_scaling_constant_lambda_E2E_FF[0];
		optimisation_stage_1_options->synapse_bool_pointer_to_turn_on = &four_layer_vision_spiking_model->E2E_FF_SYNAPSES_ON;
		optimisation_stage_1_options->ideal_output_score = upper;

		optimiser->AddOptimisationStage(optimisation_stage_1_options, simulator_options);

		// Pointer Test
		// * optimiser->model_pointers_to_be_optimised_for_each_optimisation_stage[0] = 99.0;
		// * optimiser->synapse_bool_pointers_to_turn_on_for_each_optimisation_stage[0] = true;

		// printf("four_layer_vision_spiking_model->LBL_biological_conductance_scaling_constant_lambda_E2E_FF[0]: %f\n", four_layer_vision_spiking_model->LBL_biological_conductance_scaling_constant_lambda_E2E_FF[0]);
		// if (four_layer_vision_spiking_model->E2E_FF_SYNAPSES_ON) {
		// 	printf("four_layer_vision_spiking_model->E2E_FF_SYNAPSES_ON = true \n");
		// }
	
	}

	if (number_of_optimisation_stages >= 2) {

		Optimiser_Options * optimisation_stage_2_options = new Optimiser_Options();
		optimisation_stage_2_options->model_pointer_to_be_optimised = &four_layer_vision_spiking_model->LBL_biological_conductance_scaling_constant_lambda_E2I_L[0];
		optimisation_stage_2_options->synapse_bool_pointer_to_turn_on = &four_layer_vision_spiking_model->E2I_L_SYNAPSES_ON;
		optimisation_stage_2_options->use_inhibitory_neurons = true;
		optimisation_stage_2_options->index_of_neuron_group_of_interest = 1;
		optimisation_stage_2_options->ideal_output_score = upper;

		optimiser->AddOptimisationStage(optimisation_stage_2_options, simulator_options);
	
	}


	if (number_of_optimisation_stages >= 3) {

		Optimiser_Options * optimisation_stage_3_options = new Optimiser_Options();
		optimisation_stage_3_options->model_pointer_to_be_optimised = &four_layer_vision_spiking_model->LBL_biological_conductance_scaling_constant_lambda_I2E_L[0];
		optimisation_stage_3_options->synapse_bool_pointer_to_turn_on = &four_layer_vision_spiking_model->I2E_L_SYNAPSES_ON;
		optimisation_stage_3_options->use_inhibitory_neurons = true;
		optimisation_stage_3_options->index_of_neuron_group_of_interest = 0;
		optimisation_stage_3_options->ideal_output_score = lower;

		optimiser->AddOptimisationStage(optimisation_stage_3_options, simulator_options);
	
	}

	if (number_of_optimisation_stages >= 4) {

		Optimiser_Options * optimisation_stage_4_options = new Optimiser_Options();
		optimisation_stage_4_options->model_pointer_to_be_optimised = &four_layer_vision_spiking_model->LBL_biological_conductance_scaling_constant_lambda_E2E_L[0];
		optimisation_stage_4_options->synapse_bool_pointer_to_turn_on = &four_layer_vision_spiking_model->E2E_L_SYNAPSES_ON;
		optimisation_stage_4_options->use_inhibitory_neurons = true;
		optimisation_stage_4_options->index_of_neuron_group_of_interest = 0;
		optimisation_stage_4_options->ideal_output_score = upper;

		optimiser->AddOptimisationStage(optimisation_stage_4_options, simulator_options);
	
	}

	if (number_of_optimisation_stages >= 5) {

		Optimiser_Options * optimisation_stage_5_options = new Optimiser_Options();
		optimisation_stage_5_options->model_pointer_to_be_optimised = &four_layer_vision_spiking_model->LBL_biological_conductance_scaling_constant_lambda_E2E_FF[1];
		optimisation_stage_5_options->synapse_bool_pointer_to_turn_on = &four_layer_vision_spiking_model->E2E_FF_SYNAPSES_ON;
		optimisation_stage_5_options->use_inhibitory_neurons = true;
		optimisation_stage_5_options->number_of_non_input_layers_to_simulate = 2;
		optimisation_stage_5_options->index_of_neuron_group_of_interest = 2;
		optimisation_stage_5_options->ideal_output_score = upper;

		optimiser->AddOptimisationStage(optimisation_stage_5_options, simulator_options);
	
	}


	optimiser->RunOptimisation();



	print_line_of_dashes_with_blank_lines_either_side();

	// for (int optimisation_stage = 0; optimisation_stage < number_of_optimisation_stages; optimisation_stage++) {

	// 	printf("final_optimal_parameter_for_each_optimisation_stage[optimisation_stage]: %f\n", final_optimal_parameter_for_each_optimisation_stage[optimisation_stage]);

	// }

	/////////// END OF EXPERIMENT ///////////
	experiment_timer->stop_timer_and_log_time_and_message("Experiment Completed.", true);

	cudaProfilerStop();

	return 0;
}