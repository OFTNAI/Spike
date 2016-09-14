#include "CollectEventsNetworkExperiment.h"

#include "../SpikeAnalyser/SpikeAnalyser.h"

#include "../Helpers/TerminalHelpers.h"


// CollectEventsNetworkExperiment Constructor
CollectEventsNetworkExperiment::CollectEventsNetworkExperiment() {

	spike_analyser = NULL;

	presentation_time_per_stimulus_per_epoch = 0.0;

}


// CollectEventsNetworkExperiment Destructor
CollectEventsNetworkExperiment::~CollectEventsNetworkExperiment() {
	
}

void CollectEventsNetworkExperiment::prepare_experiment(FourLayerVisionSpikingModel * four_layer_vision_spiking_model_param, bool high_fidelity_spike_storage) {

	NetworkExperiment::prepare_experiment(four_layer_vision_spiking_model, high_fidelity_spike_storage);

	/////////// SETUP RECORDING ELECTRODES ///////////
	int number_of_timesteps_per_device_spike_copy_check = 50;
	int device_spike_store_size_multiple_of_total_neurons = 52;
	float proportion_of_device_spike_store_full_before_copy = 0.2;
	simulator->setup_recording_electrodes_for_neurons(number_of_timesteps_per_device_spike_copy_check, device_spike_store_size_multiple_of_total_neurons, proportion_of_device_spike_store_full_before_copy);
	// simulator->setup_recording_electrodes_for_input_neurons(number_of_timesteps_per_device_spike_copy_check, device_spike_store_size_multiple_of_total_neurons, proportion_of_device_spike_store_full_before_copy);

	spike_analyser = new SpikeAnalyser(four_layer_vision_spiking_model->spiking_neurons, four_layer_vision_spiking_model->image_poisson_input_spiking_neurons);

}


void CollectEventsNetworkExperiment::run_experiment(float presentation_time_per_stimulus_per_epoch_param, bool isTrained) {

	presentation_time_per_stimulus_per_epoch = presentation_time_per_stimulus_per_epoch_param;

	if (experiment_prepared == false) print_message_and_exit("Please run prepare_experiment before running the experiment.");

	simulator->RunSimulationToCollectEvents(presentation_time_per_stimulus_per_epoch, isTrained);

	experiment_run = true;

}