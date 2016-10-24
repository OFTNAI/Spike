#include "TestNetworkExperiment.h"

#include "../SpikeAnalyser/SpikeAnalyser.h"

#include "../Helpers/TerminalHelpers.h"


// TestNetworkExperiment Constructor
TestNetworkExperiment::TestNetworkExperiment() {

	spike_analyser = NULL;

	presentation_time_per_stimulus_per_epoch = 0.0;

}


// TestNetworkExperiment Destructor
TestNetworkExperiment::~TestNetworkExperiment() {
	
}

void TestNetworkExperiment::prepare_test_network_experiment(FourLayerVisionSpikingModel * four_layer_vision_spiking_model_param, bool high_fidelity_spike_storage, Collect_Neuron_Spikes_Optional_Parameters * collect_neuron_spikes_optional_parameters) {

	NetworkExperiment::prepare_network_experiment(four_layer_vision_spiking_model, high_fidelity_spike_storage);

	simulator->CreateDirectoryForSimulationDataFiles("heytest");


	Simulator_Recording_Electrodes_To_Use_Struct * recording_electrodes_to_use_struct = new Recordine_Electrodes_To_Use_Struct();
	recording_electrodes_to_use_struct->collect_neuron_spikes_recording_electrodes_bool = true;

	prepare_recording_electrodes(recording_electrodes_to_use_struct, collect_neuron_spikes_optional_parameters);

	spike_analyser = new SpikeAnalyser(four_layer_vision_spiking_model->spiking_neurons, four_layer_vision_spiking_model->image_poisson_input_spiking_neurons);

}


void TestNetworkExperiment::run_experiment(float presentation_time_per_stimulus_per_epoch_param, bool record_spikes, bool save_recorded_spikes_and_states_to_file, bool human_readable_storage, bool isTrained) {

	presentation_time_per_stimulus_per_epoch = presentation_time_per_stimulus_per_epoch_param;

	if (experiment_prepared == false) print_message_and_exit("Please run prepare_experiment before running the experiment.");

	simulator->RunSimulationToCountNeuronSpikes(presentation_time_per_stimulus_per_epoch, record_spikes, save_recorded_spikes_and_states_to_file, spike_analyser, human_readable_storage, isTrained);

	experiment_run = true;

}



void TestNetworkExperiment:: calculate_spike_totals_averages_and_information(int number_of_bins, bool useThresholdForMaxFR, float max_firing_rate) {

	if (experiment_run == false) print_message_and_exit("Please run run_experiment before analysing results.");

	spike_analyser->calculate_various_neuron_spike_totals_and_averages(presentation_time_per_stimulus_per_epoch);
	
	for(int l=0; l<four_layer_vision_spiking_model->number_of_non_input_layers; l++) {
		spike_analyser->calculate_single_cell_information_scores_for_neuron_group(four_layer_vision_spiking_model->EXCITATORY_NEURONS[l], number_of_bins, useThresholdForMaxFR, max_firing_rate);
	}

}