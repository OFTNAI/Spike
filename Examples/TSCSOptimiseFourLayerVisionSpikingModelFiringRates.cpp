#include "Spike/Simulator/Simulator.hpp"
#include "Spike/Models/FourLayerVisionSpikingModel.hpp"

#include "Spike/Helpers/TimerWithMessages.hpp"
#include "Spike/Helpers/TerminalHelpers.hpp"
#include "Spike/Helpers/MemoryUsage.hpp"


#include "Spike/SpikeAnalyser/SpikeAnalyser.hpp"
#include "Spike/Optimiser/Optimiser.hpp"

// #include "Spike/Backend/CUDA/Memory.hpp"

#include "cuda_profiler_api.h"
#include <string>
#include <stdio.h>
#include <sys/stat.h>
using namespace std;


// // Use the following line to compile the binary
// //  make -C ./Spike FILE='OptimiseFourLayerVisionSpikingModelFiringRates' EXPERIMENT_DIRECTORY='../Experiments/Set2'  model -j22




// The function which will autorun when the executable is created
int main (int argc, char *argv[]){

	// int number_of_variations_of_lateral_inihibition = 3;
	// float* variations_of_lateral_inhibition[] = {0.2, 0.5, 0.8};

	// int number_of_variations_of_STDP_time_constants = 3;
	// float* variations_of_tau_C_in_ms[] = {3, 15, 75};
	// float* variations_of_tau_D_in_ms[] = {5, 25, 125};

	// // No trace vs trace
	// int number_of_variations_of_tau_g_E2E_FF = 2;
	// float* variations_of_tau_g_E2E_FF_in_ms = {2, 150};

	// // POSSIBLE OTHERS:
	// // L_EE_sd



	// for (int lateral_inhibition_index = 0; lateral_inhibition_index < number_of_variations_of_lateral_inihibition; lateral_inhibition_index++) {
		
	// 	for (int STDP_time_constant_index = 0; STDP_time_constant_index < number_of_variations_of_STDP_time_constants; STDP_time_constant_index++) {
			
	// 		for (int tau_g_E2E_FF_index = 0; tau_g_E2E_FF_index < number_of_variations_of_tau_g_E2E_FF; tau_g_E2E_FF_index++) {

				

	// 		}

	// 	}

	// }

	// std::cout << "MEMORY: " << Backend::CUDA::free_memory() << std::endl;

	TimerWithMessages * experiment_timer = new TimerWithMessages("\n");

	// const float presentation_time_per_stimulus_per_epoch = 0.15;
	// const float presentation_time_per_stimulus_per_epoch = 0.05;
	const float presentation_time_per_stimulus_per_epoch = 0.0001;

	// const float presentation_time_per_stimulus_per_epoch = 0.5;


	// SIMULATOR OPTIONS
	Simulator_Options * simulator_options = new Simulator_Options();

	simulator_options->run_simulation_general_options->presentation_time_per_stimulus_per_epoch = presentation_time_per_stimulus_per_epoch;
	simulator_options->run_simulation_general_options->number_of_epochs = 2;
	simulator_options->run_simulation_general_options->apply_stdp_to_relevant_synapses = false;
	simulator_options->run_simulation_general_options->stimulus_presentation_order_seed = 1;
	// simulator_options->run_simulation_general_options->reset_model_activities_between_epochs = false;
	// simulator_options->run_simulation_general_options->specific_epoch_to_pass_to_spike_analyser = 1;

	simulator_options->recording_electrodes_options->count_neuron_spikes_recording_electrodes_bool = true;

	simulator_options->stimuli_presentation_options->presentation_format = PRESENTATION_FORMAT_OBJECT_BY_OBJECT_NO_RESET;
	simulator_options->stimuli_presentation_options->object_order = OBJECT_ORDER_ORIGINAL;
	simulator_options->stimuli_presentation_options->transform_order = TRANSFORM_ORDER_ORIGINAL;



	// MODEL
	FourLayerVisionSpikingModel * four_layer_vision_spiking_model = new FourLayerVisionSpikingModel();
	four_layer_vision_spiking_model->SetTimestep(0.00002);
	four_layer_vision_spiking_model->high_fidelity_spike_storage = true;

	// string experiment_directory = "./Experiments/Set2/";
	string experiment_directory = "./";

	string full_inputs_directory = experiment_directory + "Inputs_9L9T/";
	string full_output_directory = experiment_directory + "Outputs/" + "take1/";

	if (mkdir((full_output_directory).c_str(),S_IRUSR | S_IWUSR | S_IXUSR | S_IRGRP | S_IWGRP | S_IXGRP | S_IROTH | S_IWOTH | S_IXOTH)==0)
		printf("\nDirectory created\n");

	four_layer_vision_spiking_model->inputs_directory = full_inputs_directory.c_str();




	float inhibition_ratio = 0.8;


	// OPTIMISATION
	float upper = 150.0;
	float lower = 100.0;
	int number_of_optimisation_stages = 7;

	Optimiser* optimiser = new Optimiser(four_layer_vision_spiking_model);

	if (number_of_optimisation_stages > 0) {

		Optimiser_Options * optimisation_stage_0_options = new Optimiser_Options();
		optimisation_stage_0_options->model_pointer_to_be_optimised = &four_layer_vision_spiking_model->LBL_biological_conductance_scaling_constant_lambda_E2E_FF[0];
		optimisation_stage_0_options->synapse_bool_pointer_to_turn_on = &four_layer_vision_spiking_model->E2E_FF_SYNAPSES_ON;
		optimisation_stage_0_options->ideal_output_score = lower;
		optimisation_stage_0_options->score_to_use = SCORE_TO_USE_average_number_of_spikes_per_neuron_group_per_second_excluding_silent_neurons;

		optimiser->AddOptimisationStage(optimisation_stage_0_options, simulator_options);

	}

	if (number_of_optimisation_stages > 1) {

		Optimiser_Options * optimisation_stage_1_options = new Optimiser_Options();
		optimisation_stage_1_options->model_pointer_to_be_optimised = &four_layer_vision_spiking_model->LBL_biological_conductance_scaling_constant_lambda_E2E_L[0];
		optimisation_stage_1_options->synapse_bool_pointer_to_turn_on = &four_layer_vision_spiking_model->E2E_L_SYNAPSES_ON;
		optimisation_stage_1_options->use_inhibitory_neurons = false;
		optimisation_stage_1_options->index_of_neuron_group_of_interest = 0;
		optimisation_stage_1_options->ideal_output_score = upper;
		// optimisation_stage_1_options->score_to_use = SCORE_TO_USE_average_number_of_spikes_per_neuron_group_per_second_excluding_silent_neurons;

		optimiser->AddOptimisationStage(optimisation_stage_1_options, simulator_options);
	
	}


	if (number_of_optimisation_stages > 2) {

		Optimiser_Options * optimisation_stage_2_options = new Optimiser_Options();
		optimisation_stage_2_options->model_pointer_to_be_optimised = &four_layer_vision_spiking_model->LBL_biological_conductance_scaling_constant_lambda_E2I_L[0];
		optimisation_stage_2_options->synapse_bool_pointer_to_turn_on = &four_layer_vision_spiking_model->E2I_L_SYNAPSES_ON;
		optimisation_stage_2_options->use_inhibitory_neurons = true;
		optimisation_stage_2_options->index_of_neuron_group_of_interest = 1;
		optimisation_stage_2_options->ideal_output_score = upper;
		optimisation_stage_2_options->score_to_use = SCORE_TO_USE_average_number_of_spikes_per_neuron_group_per_second_excluding_silent_neurons;

		optimiser->AddOptimisationStage(optimisation_stage_2_options, simulator_options);
	
	}

	optimiser->final_optimal_parameter_for_each_optimisation_stage[2] = 0.000286102295;
	optimiser->four_layer_vision_spiking_model->LBL_biological_conductance_scaling_constant_lambda_E2I_L[0] = 0.000286102295;
	optimiser->four_layer_vision_spiking_model->E2I_L_SYNAPSES_ON = true;


	optimiser->RunOptimisation(2);

	if (number_of_optimisation_stages > 3) {

		Optimiser_Options * optimisation_stage_3_options = new Optimiser_Options();
		optimisation_stage_3_options->model_pointer_to_be_optimised = &four_layer_vision_spiking_model->LBL_biological_conductance_scaling_constant_lambda_I2E_L[0];
		optimisation_stage_3_options->synapse_bool_pointer_to_turn_on = &four_layer_vision_spiking_model->I2E_L_SYNAPSES_ON;
		optimisation_stage_3_options->use_inhibitory_neurons = true;
		optimisation_stage_3_options->index_of_neuron_group_of_interest = 0;
		optimisation_stage_3_options->ideal_output_score = inhibition_ratio*upper;
		optimisation_stage_3_options->positive_effect_of_postive_change_in_parameter = false;

		optimiser->AddOptimisationStage(optimisation_stage_3_options, simulator_options);
	
	}



	four_layer_vision_spiking_model->set_LBL_values_for_pointer_from_layer_to_layer(four_layer_vision_spiking_model->LBL_biological_conductance_scaling_constant_lambda_E2I_L[0], four_layer_vision_spiking_model->LBL_biological_conductance_scaling_constant_lambda_E2I_L, 1, 3);
	four_layer_vision_spiking_model->set_LBL_values_for_pointer_from_layer_to_layer(four_layer_vision_spiking_model->LBL_biological_conductance_scaling_constant_lambda_I2E_L[0], four_layer_vision_spiking_model->LBL_biological_conductance_scaling_constant_lambda_I2E_L, 1, 3);
	four_layer_vision_spiking_model->set_LBL_values_for_pointer_from_layer_to_layer(four_layer_vision_spiking_model->LBL_biological_conductance_scaling_constant_lambda_E2E_L[0], four_layer_vision_spiking_model->LBL_biological_conductance_scaling_constant_lambda_E2E_L, 1, 3);


	if (number_of_optimisation_stages > 4) {

		Optimiser_Options * optimisation_stage_4_options = new Optimiser_Options();
		optimisation_stage_4_options->model_pointer_to_be_optimised = &four_layer_vision_spiking_model->LBL_biological_conductance_scaling_constant_lambda_E2E_FF[1];
		optimisation_stage_4_options->synapse_bool_pointer_to_turn_on = &four_layer_vision_spiking_model->E2E_FF_SYNAPSES_ON;
		optimisation_stage_4_options->use_inhibitory_neurons = true;
		optimisation_stage_4_options->number_of_non_input_layers_to_simulate = 2;
		optimisation_stage_4_options->index_of_neuron_group_of_interest = 2;
		optimisation_stage_4_options->ideal_output_score = inhibition_ratio*upper;

		optimiser->AddOptimisationStage(optimisation_stage_4_options, simulator_options);
	
	}

	if (number_of_optimisation_stages > 5) {

		Optimiser_Options * optimisation_stage_5_options = new Optimiser_Options();
		optimisation_stage_5_options->model_pointer_to_be_optimised = &four_layer_vision_spiking_model->LBL_biological_conductance_scaling_constant_lambda_E2E_FF[2];
		optimisation_stage_5_options->synapse_bool_pointer_to_turn_on = &four_layer_vision_spiking_model->E2E_FF_SYNAPSES_ON;
		optimisation_stage_5_options->use_inhibitory_neurons = true;
		optimisation_stage_5_options->number_of_non_input_layers_to_simulate = 3;
		optimisation_stage_5_options->index_of_neuron_group_of_interest = 4;
		optimisation_stage_5_options->ideal_output_score = inhibition_ratio*upper;

		optimiser->AddOptimisationStage(optimisation_stage_5_options, simulator_options);
	
	}

	if (number_of_optimisation_stages > 6) {

		Optimiser_Options * optimisation_stage_6_options = new Optimiser_Options();
		optimisation_stage_6_options->model_pointer_to_be_optimised = &four_layer_vision_spiking_model->LBL_biological_conductance_scaling_constant_lambda_E2E_FF[3];
		optimisation_stage_6_options->synapse_bool_pointer_to_turn_on = &four_layer_vision_spiking_model->E2E_FF_SYNAPSES_ON;
		optimisation_stage_6_options->use_inhibitory_neurons = true;
		optimisation_stage_6_options->number_of_non_input_layers_to_simulate = 4;
		optimisation_stage_6_options->index_of_neuron_group_of_interest = 6;
		optimisation_stage_6_options->ideal_output_score = inhibition_ratio*upper;

		optimiser->AddOptimisationStage(optimisation_stage_6_options, simulator_options);
	
	}


	optimiser->RunOptimisation();


	// // New Back Projections
	// if (number_of_optimisation_stages > 7) {

	// 	Optimiser_Options * optimisation_stage_6_options = new Optimiser_Options();
	// 	optimisation_stage_6_options->model_pointer_to_be_optimised = &four_layer_vision_spiking_model->LBL_biological_conductance_scaling_constant_lambda_E2E_FF[3];
	// 	optimisation_stage_6_options->synapse_bool_pointer_to_turn_on = &four_layer_vision_spiking_model->E2E_FF_SYNAPSES_ON;
	// 	optimisation_stage_6_options->use_inhibitory_neurons = true;
	// 	optimisation_stage_6_options->number_of_non_input_layers_to_simulate = 4;
	// 	optimisation_stage_6_options->index_of_neuron_group_of_interest = 6;
	// 	optimisation_stage_6_options->ideal_output_score = inhibition_ratio*upper;

	// 	optimiser->AddOptimisationStage(optimisation_stage_6_options, simulator_options);
	
	// }

	// optimiser->write_final_optimisation_parameters_to_file(full_output_directory);


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

	return 0;
}