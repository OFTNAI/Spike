#ifndef TestTrainTestExperimentSet_H
#define TestTrainTestExperimentSet_H

#include <cuda.h>
#include <stdio.h>

#include "../Simulator/Simulator.h"
#include "../Models/FourLayerVisionSpikingModel.h"
#include "../SpikeAnalyser/SpikeAnalyser.h"


class TestTrainTestExperimentSet {
public:
	// Constructor/Destructor
	TestTrainTestExperimentSet();
	~TestTrainTestExperimentSet();

	void run_experiment_set_for_model(FourLayerVisionSpikingModel * four_layer_vision_spiking_model, float presentation_time_per_stimulus_per_epoch_test, bool record_test_spikes, bool save_recorded_test_spikes_and_states_to_file, bool human_readable_storage, bool high_fidelity_spike_storage, int number_of_bins, bool useThresholdForMaxFR, float max_firing_rate, float presentation_time_per_stimulus_per_epoch_train, int number_of_training_epochs);

};

#endif