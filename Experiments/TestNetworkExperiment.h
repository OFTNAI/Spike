#ifndef TestNetworkExperiment_H
#define TestNetworkExperiment_H

#include "NetworkExperiment.h"

class TestNetworkExperiment : public NetworkExperiment {
public:
	// Constructor/Destructor
	TestNetworkExperiment();
	~TestNetworkExperiment();

	SpikeAnalyser * spike_analyser;

	float presentation_time_per_stimulus_per_epoch;

	void prepare_test_network_experiment(FourLayerVisionSpikingModel * four_layer_vision_spiking_model, bool high_fidelity_spike_storage, Collect_Neuron_Spikes_Optional_Parameters * collect_neuron_spikes_optional_parameters, Collect_Neuron_Spikes_Optional_Parameters * collect_input_neuron_spikes_optional_parameters, Network_State_Archive_Optional_Parameters * network_state_archive_optional_parameters);

	void run_experiment(float presentation_time_per_stimulus_per_epoch, bool record_spikes, bool save_recorded_spikes_and_states_to_file, bool human_readable_storage, bool isTrained);

	void calculate_spike_totals_averages_and_information(int number_of_bins, bool useThresholdForMaxFR, float max_firing_rate);


};

#endif