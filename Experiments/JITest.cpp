#include "../Simulator/Simulator.h"
#include "../Models/FourLayerVisionSpikingModel.h"

#include "../SpikeAnalyser/SpikeAnalyser.h"
#include "../Helpers/TimerWithMessages.h"
#include "../Helpers/RandomStateManager.h"

// Use the following line to compile the binary
// make FILE='JITest' EXPERIMENT_DIRECTORY='Experiments'  model -j8


// The function which will autorun when the executable is created
int main (int argc, char *argv[]){
	TimerWithMessages * experiment_timer = new TimerWithMessages();

	
	// Simulator Parameters
	// float timestep = 0.00002;
	float timestep = 0.00002;
	bool high_fidelity_spike_storage = true;

	// bool simulate_network_to_test_untrained = true;
	// bool simulate_network_to_train_network = true;
	// bool simulate_network_to_test_trained = true;
	

	// Parameters for OPTIMISATION + Information Analysis
	const float optimal_max_firing_rate = 100.0f;//set if optimizing based on maxfr //Maximum rate (spikes/sec) 87 +- 46  (*1)
	//*1 Bair, W., & Movshon, J. A. (2004).  Adaptive Temporal Integration of Motion in Direction-Selective Neurons in Macaque Visual Cortex. The Journal of Neuroscience, 24(33), 7305遯ｶ�ｿｽ7323.
	int number_of_bins = 5;
	bool useThresholdForMaxFR = true;
	const float presentation_time_per_stimulus_per_epoch_test = 0.5f;
	float max_firing_rate = optimal_max_firing_rate*presentation_time_per_stimulus_per_epoch_test;


	// RANDOM STATES (SHOULDN'T BE HERE!!!)
	int random_states_threads_per_block_x = 128;
	int random_states_number_of_blocks_x = 64;
	RandomStateManager::instance()->set_up_random_states(random_states_threads_per_block_x, random_states_number_of_blocks_x, 9);


	// MODEL
	FourLayerVisionSpikingModel * four_layer_vision_spiking_model = new FourLayerVisionSpikingModel();
	four_layer_vision_spiking_model->SetTimestep(timestep);
	four_layer_vision_spiking_model->finalise_model();
	four_layer_vision_spiking_model->copy_model_to_device(high_fidelity_spike_storage);
	

	// SIMULATOR OPTIONS
	Simulator_Options * simulator_options = new Simulator_Options();
	simulator_options->run_simulation_general_options->presentation_time_per_stimulus_per_epoch = 0.2;
	simulator_options->run_simulation_general_options->number_of_epochs = 10;
	simulator_options->run_simulation_general_options->apply_stdp_to_relevant_synapses = true;
	simulator_options->run_simulation_general_options->stimulus_presentation_order_seed = 8;

	// CREATE SIMULATOR
	Simulator * simulator = new Simulator(four_layer_vision_spiking_model, simulator_options);

	Stimuli_Presentation_Struct * stimuli_presentation_params = new Stimuli_Presentation_Struct();
	stimuli_presentation_params->presentation_format = PRESENTATION_FORMAT_OBJECT_BY_OBJECT_RESET_BETWEEN_OBJECTS;
	stimuli_presentation_params->object_order = OBJECT_ORDER_ORIGINAL;
	stimuli_presentation_params->transform_order = TRANSFORM_ORDER_RANDOM;

	// RUN SIMULATION
	simulator->RunSimulation(NULL);


	/////////// END OF EXPERIMENT ///////////
	experiment_timer->stop_timer_and_log_time_and_message("TestNetworkExperiment Completed.", true);

	return 0;
}
