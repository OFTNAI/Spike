#ifndef NetworkExperiment_H
#define NetworkExperiment_H

#include <cuda.h>
#include <stdio.h>

#include "../Simulator/Simulator.h"
#include "../Models/FourLayerVisionSpikingModel.h"
#include "../SpikeAnalyser/SpikeAnalyser.h"


class NetworkExperiment {
public:
	// Constructor/Destructor
	NetworkExperiment();
	~NetworkExperiment();

	Simulator * simulator;
	FourLayerVisionSpikingModel * four_layer_vision_spiking_model;

	bool experiment_prepared;
	bool experiment_run;

	virtual void prepare_experiment(FourLayerVisionSpikingModel * four_layer_vision_spiking_model, bool high_fidelity_spike_storage);

	void setup_recording_electrodes_for_simulator();

};

#endif