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

	setup_recording_electrodes_for_simulator();


	// spike_analyser = new SpikeAnalyser(four_layer_vision_spiking_model->spiking_neurons, four_layer_vision_spiking_model->image_poisson_input_spiking_neurons);

}


void CollectEventsNetworkExperiment::run_experiment(float presentation_time_per_stimulus_per_epoch_param, bool isTrained) {

	presentation_time_per_stimulus_per_epoch = presentation_time_per_stimulus_per_epoch_param;

	if (experiment_prepared == false) print_message_and_exit("Please run prepare_experiment before running the experiment.");

	simulator->RunSimulationToCollectEvents(presentation_time_per_stimulus_per_epoch, isTrained);

	experiment_run = true;

}

