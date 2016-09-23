#ifndef CollectEventsNetworkExperiment_H
#define CollectEventsNetworkExperiment_H

#include <cuda.h>
#include <stdio.h>

#include "../Simulator/Simulator.h"
#include "../Models/FourLayerVisionSpikingModel.h"
#include "../SpikeAnalyser/SpikeAnalyser.h"
#include "NetworkExperiment.h"

#include "TestNetworkExperiment.h"

class CollectEventsNetworkExperiment : public NetworkExperiment {
public:
	// Constructor/Destructor
	CollectEventsNetworkExperiment();
	~CollectEventsNetworkExperiment();

	SpikeAnalyser * spike_analyser;

	float presentation_time_per_stimulus_per_epoch;

	void prepare_arrays_for_event_collection(TestNetworkExperiment * test_network_experiment);

	void prepare_experiment(FourLayerVisionSpikingModel * four_layer_vision_spiking_model_param, bool high_fidelity_spike_storage);

	void run_experiment(float presentation_time_per_stimulus_per_epoch, bool isTrained);




};

#endif