// #include "TrainNetworkExperiment.h"

// #include "../SpikeAnalyser/SpikeAnalyser.h"

// #include "../Helpers/TerminalHelpers.h"


// // TrainNetworkExperiment Constructor
// TrainNetworkExperiment::TrainNetworkExperiment() {

// 	presentation_time_per_stimulus_per_epoch = 0.0;

// }


// // TrainNetworkExperiment Destructor
// TrainNetworkExperiment::~TrainNetworkExperiment() {
	
// }

// void TrainNetworkExperiment::prepare_train_network_experiment(FourLayerVisionSpikingModel * four_layer_vision_spiking_model_param) {

// 	NetworkExperiment::prepare_network_experiment(four_layer_vision_spiking_model);

// }


// void TrainNetworkExperiment::train_model(Simulator_Run_Simulation_General_Options * simulator_run_simulation_general_options_struct) {

// 	presentation_time_per_stimulus_per_epoch = presentation_time_per_stimulus_per_epoch_param;

// 	if (experiment_prepared == false) print_message_and_exit("Please run prepare_experiment before running the experiment.");

// 	// /////////// SIMULATE NETWORK TRAINING ///////////
// 	int stimulus_presentation_order_seed = 1;

// 	Stimuli_Presentation_Struct * stimuli_presentation_params = new Stimuli_Presentation_Struct();
// 	stimuli_presentation_params->presentation_format = PRESENTATION_FORMAT_OBJECT_BY_OBJECT_RESET_BETWEEN_OBJECTS;
// 	stimuli_presentation_params->object_order = OBJECT_ORDER_ORIGINAL;//OBJECT_ORDER_RANDOM;
// 	stimuli_presentation_params->transform_order = TRANSFORM_ORDER_RANDOM;

// 	simulator->RunSimulationToTrainNetwork(presentation_time_per_stimulus_per_epoch, number_of_training_epochs, stimuli_presentation_params, stimulus_presentation_order_seed);

// }