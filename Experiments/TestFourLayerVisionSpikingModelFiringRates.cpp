#include "../Optimiser/Optimiser.h"

#include "../Simulator/Simulator.h"
#include "../Models/FourLayerVisionSpikingModel.h"

#include "../SpikeAnalyser/SpikeAnalyser.h"
#include "../Helpers/TimerWithMessages.h"
#include "../Helpers/TerminalHelpers.h"
#include "../Helpers/MemoryUsage.h"

#include "cuda_profiler_api.h"


// Use the following line to compile the binary
// make FILE='TestFourLayerVisionSpikingModelFiringRates' EXPERIMENT_DIRECTORY='Experiments'  model -j22


// The function which will autorun when the executable is created
int main (int argc, char *argv[]){


	TimerWithMessages * experiment_timer = new TimerWithMessages("\n");


	const float presentation_time_per_stimulus_per_epoch = 0.2;


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

	four_layer_vision_spiking_model->number_of_non_input_layers_to_simulate = 4;

	four_layer_vision_spiking_model->INHIBITORY_NEURONS_ON = true;
	four_layer_vision_spiking_model->E2E_FF_SYNAPSES_ON = true;
	four_layer_vision_spiking_model->E2I_L_SYNAPSES_ON = true;
	four_layer_vision_spiking_model->I2E_L_SYNAPSES_ON = true;
	four_layer_vision_spiking_model->E2E_L_SYNAPSES_ON = true;

	four_layer_vision_spiking_model->LBL_biological_conductance_scaling_constant_lambda_E2E_FF[0] = 0.000292968762;
	four_layer_vision_spiking_model->LBL_biological_conductance_scaling_constant_lambda_E2E_FF[1] = 0.000030517578;
	four_layer_vision_spiking_model->LBL_biological_conductance_scaling_constant_lambda_E2E_FF[2] = 0.000036621095;
	four_layer_vision_spiking_model->LBL_biological_conductance_scaling_constant_lambda_E2E_FF[3] = 0.000061035156;
	four_layer_vision_spiking_model->set_LBL_values_for_pointer_from_layer_to_layer(0.010937500745, four_layer_vision_spiking_model->LBL_biological_conductance_scaling_constant_lambda_E2I_L, 0, 3);
	four_layer_vision_spiking_model->set_LBL_values_for_pointer_from_layer_to_layer(0.050000000745, four_layer_vision_spiking_model->LBL_biological_conductance_scaling_constant_lambda_I2E_L, 0, 3);
	four_layer_vision_spiking_model->set_LBL_values_for_pointer_from_layer_to_layer(0.000292968762, four_layer_vision_spiking_model->LBL_biological_conductance_scaling_constant_lambda_E2E_L, 0, 3);


	// FINALISE MODEL + COPY TO DEVICE
	four_layer_vision_spiking_model->finalise_model();
	four_layer_vision_spiking_model->copy_model_to_device();

	// CREATE SIMULATOR
	Simulator * simulator = new Simulator(four_layer_vision_spiking_model, simulator_options);

	// RUN SIMULATION
	SpikeAnalyser * spike_analyser = new SpikeAnalyser(four_layer_vision_spiking_model->spiking_neurons, four_layer_vision_spiking_model->input_spiking_neurons);
	simulator->RunSimulation(spike_analyser);


	// CALCULATE AVERAGES + OPTIMISATION OUTPUT SCORE
	spike_analyser->calculate_various_neuron_spike_totals_and_averages(simulator_options->run_simulation_general_options->presentation_time_per_stimulus_per_epoch);



	/////////// END OF EXPERIMENT ///////////
	experiment_timer->stop_timer_and_log_time_and_message("Experiment Completed.", true);

	cudaProfilerStop();

	return 0;
}