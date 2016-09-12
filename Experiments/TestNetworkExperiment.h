#ifndef TestNetworkExperiment_H
#define TestNetworkExperiment_H

#include <cuda.h>
#include <stdio.h>

#include "../Simulator/Simulator.h"
#include "../Models/FourLayerVisionSpikingModel.h"
#include "../SpikeAnalyser/SpikeAnalyser.h"


class TestNetworkExperiment {
public:
	// Constructor/Destructor
	TestNetworkExperiment();
	~TestNetworkExperiment();

	Simulator * simulator;
	FourLayerVisionSpikingModel * four_layer_vision_spiking_model;
	SpikeAnalyser * spike_analyser;

	bool experiment_prepared;
	bool experiment_run;

	float presentation_time_per_stimulus_per_epoch;

	void prepare_experiment(FourLayerVisionSpikingModel * four_layer_vision_spiking_model, bool high_fidelity_spike_storage);

	void run_experiment(float presentation_time_per_stimulus_per_epoch, bool record_spikes, bool save_recorded_spikes_and_states_to_file, bool human_readable_storage, bool isTrained);

	void calculate_spike_totals_averages_and_information(int number_of_bins, bool useThresholdForMaxFR, float max_firing_rate);


};

#endif