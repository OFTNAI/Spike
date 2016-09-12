#include "TestTrainTestExperimentSet.h"

#include "../SpikeAnalyser/SpikeAnalyser.h"

#include "../Helpers/TerminalHelpers.h"

#include "TestNetworkExperiment.h"

// TestTrainTestExperimentSet Constructor
TestTrainTestExperimentSet::TestTrainTestExperimentSet() {

}


// TestTrainTestExperimentSet Destructor
TestTrainTestExperimentSet::~TestTrainTestExperimentSet() {
	
}


void TestTrainTestExperimentSet::run_experiment_set_for_model(FourLayerVisionSpikingModel * four_layer_vision_spiking_model, float presentation_time_per_stimulus_per_epoch_test, bool record_test_spikes, bool save_recorded_test_spikes_and_states_to_file, bool human_readable_storage, bool high_fidelity_spike_storage, int number_of_bins, bool useThresholdForMaxFR, float max_firing_rate, float presentation_time_per_stimulus_per_epoch_train, int number_of_training_epochs) {

	bool network_is_trained = false;

	/////////// SIMULATE NETWORK TO TEST UNTRAINED ///////////
	TestNetworkExperiment * test_untrained_network_experiment = new TestNetworkExperiment();
	test_untrained_network_experiment->four_layer_vision_spiking_model = four_layer_vision_spiking_model;
	test_untrained_network_experiment->prepare_experiment(four_layer_vision_spiking_model, high_fidelity_spike_storage);
	test_untrained_network_experiment->run_experiment(presentation_time_per_stimulus_per_epoch_test, record_test_spikes, save_recorded_test_spikes_and_states_to_file, human_readable_storage, network_is_trained);
	test_untrained_network_experiment->calculate_spike_totals_averages_and_information(number_of_bins, useThresholdForMaxFR, max_firing_rate);


	// // /////////// SIMULATE NETWORK TRAINING ///////////
	// int stimulus_presentation_order_seed = 1;

	// Stimuli_Presentation_Struct * stimuli_presentation_params = new Stimuli_Presentation_Struct();
	// stimuli_presentation_params->presentation_format = PRESENTATION_FORMAT_OBJECT_BY_OBJECT_RESET_BETWEEN_OBJECTS;
	// stimuli_presentation_params->object_order = OBJECT_ORDER_ORIGINAL;//OBJECT_ORDER_RANDOM;
	// stimuli_presentation_params->transform_order = TRANSFORM_ORDER_RANDOM;

	// simulator->RunSimulationToTrainNetwork(presentation_time_per_stimulus_per_epoch_train, number_of_training_epochs, stimuli_presentation_params, stimulus_presentation_order_seed);

	network_is_trained = true;


	/////////// SIMULATE NETWORK TO TEST TRAINED ///////////
	TestNetworkExperiment * test_trained_network_experiment = new TestNetworkExperiment();
	test_trained_network_experiment->four_layer_vision_spiking_model = four_layer_vision_spiking_model;
	test_untrained_network_experiment->prepare_experiment(four_layer_vision_spiking_model, high_fidelity_spike_storage);
	test_trained_network_experiment->run_experiment(presentation_time_per_stimulus_per_epoch_test, record_test_spikes, save_recorded_test_spikes_and_states_to_file, human_readable_storage, network_is_trained);
	test_trained_network_experiment->calculate_spike_totals_averages_and_information(number_of_bins, useThresholdForMaxFR, max_firing_rate);


}