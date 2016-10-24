// #ifndef TrainNetworkExperiment_H
// #define TrainNetworkExperiment_H

// #include <cuda.h>
// #include <stdio.h>

// #include "../Simulator/Simulator.h"
// #include "../Models/FourLayerVisionSpikingModel.h"
// #include "../SpikeAnalyser/SpikeAnalyser.h"
// #include "NetworkExperiment.h"

// class TrainNetworkExperiment : public NetworkExperiment {
// public:
// 	// Constructor/Destructor
// 	TrainNetworkExperiment();
// 	~TrainNetworkExperiment();

// 	float presentation_time_per_stimulus_per_epoch;

// 	void prepare_experiment(FourLayerVisionSpikingModel * four_layer_vision_spiking_model, bool high_fidelity_spike_storage);

// 	void run_experiment(float presentation_time_per_stimulus_per_epoch_param, int number_of_training_epochs);


// };

// #endif