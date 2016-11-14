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

// 	void prepare_test_network_experiment(FourLayerVisionSpikingModel * four_layer_vision_spiking_model);

// 	void train_model(Simulator_Run_Simulation_General_Options * simulator_run_simulation_general_options_struct);


// };

// #endif