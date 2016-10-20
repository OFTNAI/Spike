#include "NetworkExperiment.h"

#include "../SpikeAnalyser/SpikeAnalyser.h"

#include "../Helpers/TerminalHelpers.h"


// NetworkExperiment Constructor
NetworkExperiment::NetworkExperiment() {

	four_layer_vision_spiking_model = NULL;
	simulator = NULL;

	experiment_prepared = false;
	experiment_run = false;


}


// NetworkExperiment Destructor
NetworkExperiment::~NetworkExperiment() {
	
}

void NetworkExperiment::prepare_experiment(FourLayerVisionSpikingModel * four_layer_vision_spiking_model_param, bool high_fidelity_spike_storage) {

	four_layer_vision_spiking_model = four_layer_vision_spiking_model_param;

	// Create an instance of the Simulator

	simulator = new Simulator();
	simulator->SetSpikingModel(four_layer_vision_spiking_model);
	simulator->high_fidelity_spike_storage = high_fidelity_spike_storage;
	
	experiment_prepared = true;

}

void NetworkExperiment::setup_recording_electrodes_for_simulator() {
	/////////// SETUP RECORDING ELECTRODES ///////////
	int number_of_timesteps_per_device_spike_copy_check = 50;
	int device_spike_store_size_multiple_of_total_neurons = 52;
	float proportion_of_device_spike_store_full_before_copy = 0.2;
	simulator->setup_recording_electrodes_for_neurons(number_of_timesteps_per_device_spike_copy_check, device_spike_store_size_multiple_of_total_neurons, proportion_of_device_spike_store_full_before_copy);
}