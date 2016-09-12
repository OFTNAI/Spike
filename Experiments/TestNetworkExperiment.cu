#include "TestNetworkExperiment.h"

#include "../SpikeAnalyser/SpikeAnalyser.h"

#include "../Helpers/TerminalHelpers.h"


// TestNetworkExperiment Constructor
TestNetworkExperiment::TestNetworkExperiment() {

	four_layer_vision_spiking_model = NULL;
	simulator = NULL;

	experiment_prepared = false;
	experiment_run = false;

	presentation_time_per_stimulus_per_epoch = 0.0;

}


// TestNetworkExperiment Destructor
TestNetworkExperiment::~TestNetworkExperiment() {
	
}

void TestNetworkExperiment::prepare_experiment(FourLayerVisionSpikingModel * four_layer_vision_spiking_model_param, bool high_fidelity_spike_storage) {

	four_layer_vision_spiking_model = four_layer_vision_spiking_model_param;

	

	// Create an instance of the Simulator
	simulator = new Simulator();
	simulator->SetSpikingModel(four_layer_vision_spiking_model);
	simulator->high_fidelity_spike_storage = high_fidelity_spike_storage;

	four_layer_vision_spiking_model->copy_model_to_device(simulator->high_fidelity_spike_storage);

	/////////// SETUP RECORDING ELECTRODES ///////////
	int number_of_timesteps_per_device_spike_copy_check = 50;
	int device_spike_store_size_multiple_of_total_neurons = 52;
	float proportion_of_device_spike_store_full_before_copy = 0.2;
	simulator->setup_recording_electrodes_for_neurons(number_of_timesteps_per_device_spike_copy_check, device_spike_store_size_multiple_of_total_neurons, proportion_of_device_spike_store_full_before_copy);
	// simulator->setup_recording_electrodes_for_input_neurons(number_of_timesteps_per_device_spike_copy_check, device_spike_store_size_multiple_of_total_neurons, proportion_of_device_spike_store_full_before_copy);

	spike_analyser = new SpikeAnalyser(four_layer_vision_spiking_model->spiking_neurons, four_layer_vision_spiking_model->image_poisson_input_spiking_neurons);
	
	experiment_prepared = true;

}


void TestNetworkExperiment::run_experiment(float presentation_time_per_stimulus_per_epoch_param, bool record_spikes, bool save_recorded_spikes_and_states_to_file, bool human_readable_storage, bool isTrained) {

	presentation_time_per_stimulus_per_epoch = presentation_time_per_stimulus_per_epoch_param;

	if (experiment_prepared == false) print_message_and_exit("Please run prepare_experiment before running the experiment.");

	SpikeAnalyser * spike_analyser = new SpikeAnalyser(four_layer_vision_spiking_model->spiking_neurons, four_layer_vision_spiking_model->image_poisson_input_spiking_neurons);

	simulator->RunSimulationToCountNeuronSpikes(presentation_time_per_stimulus_per_epoch, record_spikes, save_recorded_spikes_and_states_to_file, spike_analyser, human_readable_storage, isTrained);


}



void TestNetworkExperiment:: calculate_spike_totals_averages_and_information(int number_of_bins, bool useThresholdForMaxFR, float max_firing_rate) {

	if (experiment_run == false) print_message_and_exit("Please run run_experiment before analysing results.");

	spike_analyser->calculate_various_neuron_spike_totals_and_averages(presentation_time_per_stimulus_per_epoch);
	
	for(int l=0; l<four_layer_vision_spiking_model->number_of_non_input_layers; l++) {
		spike_analyser->calculate_single_cell_information_scores_for_neuron_group(four_layer_vision_spiking_model->EXCITATORY_NEURONS[l], number_of_bins, useThresholdForMaxFR, max_firing_rate);
	}

}