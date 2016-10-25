#include "../Simulator/Simulator.h"
#include "../Synapses/ConductanceSpikingSynapses.h"
#include "../STDP/STDP.h"
#include "../STDP/EvansSTDP.h"
#include "../Neurons/Neurons.h"
#include "../Neurons/SpikingNeurons.h"
#include "../Neurons/LIFSpikingNeurons.h"
#include "../Neurons/ImagePoissonInputSpikingNeurons.h"
#include "../Helpers/TerminalHelpers.h"
#include "../SpikeAnalyser/SpikeAnalyser.h"
#include "../Helpers/TimerWithMessages.h"
#include "../Helpers/RandomStateManager.h"
#include <string>
#include <fstream>
// #include "../Plotting/Plotter.h"
#include <vector>
#include "../Helpers/CUDAErrorCheckHelpers.h"

#include <iostream>
using namespace std;

#include "../Models/FourLayerVisionSpikingModel.h"
#include "../Experiments/TestNetworkExperiment.h"
#include "../Experiments/TrainNetworkExperiment.h"
#include "../Experiments/TestTrainTestExperimentSet.h"
#include "../Experiments/CollectEventsNetworkExperiment.h"


// make FILE='JITest' EXPERIMENT_DIRECTORY='Experiments'  model -j8


// The function which will autorun when the executable is created
int main (int argc, char *argv[]){
	TimerWithMessages * experiment_timer = new TimerWithMessages();

	
	const float optimal_max_firing_rate = 100.0f;//set if optimizing based on maxfr //Maximum rate (spikes/sec) 87 +- 46  (*1)
	//*1 Bair, W., & Movshon, J. A. (2004).  Adaptive Temporal Integration of Motion in Direction-Selective Neurons in Macaque Visual Cortex. The Journal of Neuroscience, 24(33), 7305遯ｶ�ｿｽ7323.

	
	// Simulator Parameters
	// float timestep = 0.00002;
	float timestep = 0.00002;
	bool high_fidelity_spike_storage = true;

	bool simulate_network_to_test_untrained = true;
	bool simulate_network_to_train_network = true;
	bool simulate_network_to_test_trained = true;
	

	bool human_readable_storage = false;
	bool plotInfoAnalysis = true;
	bool writeInformation = true;


	// Parameters for testing
	// const float presentation_time_per_stimulus_per_epoch_test = 2.0f;
	const float presentation_time_per_stimulus_per_epoch_test = 0.5f;

	bool record_test_spikes = true;
	bool save_recorded_spikes_and_states_to_file_test = true;

	// Parameters for training
	// float presentation_time_per_stimulus_per_epoch_train = 2.0f;//0.5f;
	float presentation_time_per_stimulus_per_epoch_train = 0.5f;//0.5f;

	int number_of_training_epochs = 10;

	// Parameters for Information Analysis
	int number_of_bins = 5;
	bool useThresholdForMaxFR = true;
	float max_firing_rate = optimal_max_firing_rate*presentation_time_per_stimulus_per_epoch_test;


	// init parameters
	// bool is_optimisation = false;
	bool network_is_trained = false;



	int random_states_threads_per_block_x = 128;
	int random_states_number_of_blocks_x = 64;
	RandomStateManager::instance()->set_up_random_states(random_states_threads_per_block_x, random_states_number_of_blocks_x, 9);


	FourLayerVisionSpikingModel * four_layer_vision_spiking_model = new FourLayerVisionSpikingModel();
	four_layer_vision_spiking_model->SetTimestep(timestep);
	four_layer_vision_spiking_model->finalise_model();
	four_layer_vision_spiking_model->copy_model_to_device(high_fidelity_spike_storage);



	/////////// SIMULATE NETWORK TO TEST UNTRAINED ///////////
	

	// TestNetworkExperiment * test_untrained_network_experiment = new TestNetworkExperiment();
	// test_untrained_network_experiment->four_layer_vision_spiking_model = four_layer_vision_spiking_model;
	// test_untrained_network_experiment->prepare_test_network_experiment(four_layer_vision_spiking_model, high_fidelity_spike_storage, NULL, NULL, NULL);
	// test_untrained_network_experiment->run_experiment(presentation_time_per_stimulus_per_epoch_test, record_test_spikes, save_recorded_spikes_and_states_to_file_test, human_readable_storage, network_is_trained);
	// test_untrained_network_experiment->calculate_spike_totals_averages_and_information(number_of_bins, useThresholdForMaxFR, max_firing_rate);

	// TrainNetworkExperiment * train_network_experiment = new TrainNetworkExperiment();
	// train_network_experiment->prepare_train_network_experiment(four_layer_vision_spiking_model);


	Simulator_Options * simulator_options = new Simulator_Options();
	simulator_options->run_simulation_general_options->presentation_time_per_stimulus_per_epoch = 0.2;
	simulator_options->run_simulation_general_options->number_of_epochs = 10;
	simulator_options->run_simulation_general_options->apply_stdp_to_relevant_synapses = true;
	simulator_options->run_simulation_general_options->stimulus_presentation_order_seed = 8;


	Simulator * simulator = new Simulator(four_layer_vision_spiking_model, simulator_options);



	Stimuli_Presentation_Struct * stimuli_presentation_params = new Stimuli_Presentation_Struct();
	stimuli_presentation_params->presentation_format = PRESENTATION_FORMAT_OBJECT_BY_OBJECT_RESET_BETWEEN_OBJECTS;
	stimuli_presentation_params->object_order = OBJECT_ORDER_ORIGINAL;
	stimuli_presentation_params->transform_order = TRANSFORM_ORDER_RANDOM;


	simulator->RunSimulation(stimuli_presentation_params, NULL);



	// CollectEventsNetworkExperiment * collect_events_experiment_set = new CollectEventsNetworkExperiment();
	// collect_events_experiment_set->four_layer_vision_spiking_model = four_layer_vision_spiking_model;
	// collect_events_experiment_set->prepare_experiment(four_layer_vision_spiking_model, high_fidelity_spike_storage);
	// collect_events_experiment_set->prepare_arrays_for_event_collection(test_untrained_network_experiment);
	// collect_events_experiment_set->run_experiment(presentation_time_per_stimulus_per_epoch_test, network_is_trained);


	/////////// END OF EXPERIMENT ///////////
	experiment_timer->stop_timer_and_log_time_and_message("TestNetworkExperiment Completed.", true);

	return 0;
}
//
