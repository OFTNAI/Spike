#include "CollectEventsNetworkExperiment.h"

#include "../SpikeAnalyser/SpikeAnalyser.h"

#include "../Helpers/TerminalHelpers.h"

#include "../Helpers/CUDAErrorCheckHelpers.h"


// CollectEventsNetworkExperiment Constructor
CollectEventsNetworkExperiment::CollectEventsNetworkExperiment() {

	spike_analyser = NULL;

	presentation_time_per_stimulus_per_epoch = 0.0;

}


// CollectEventsNetworkExperiment Destructor
CollectEventsNetworkExperiment::~CollectEventsNetworkExperiment() {
	
}


void CollectEventsNetworkExperiment::prepare_arrays_for_event_collection(TestNetworkExperiment * test_network_experiment) {

	int number_of_stimuli = test_network_experiment->spike_analyser->input_neurons->total_number_of_input_stimuli;
	int number_of_neurons = test_network_experiment->four_layer_vision_spiking_model->spiking_neurons->total_number_of_neurons;

	int* spikes_per_neuron = (int*)malloc(sizeof(int)*number_of_neurons);
	int* temp_index_of_latest_spike_per_neuron = (int*)malloc(sizeof(int)*number_of_neurons);

	bool *** neuron_events_for_each_neuron = new bool **[number_of_neurons];
	float ** ordered_spike_times_for_each_neuron = new float *[number_of_neurons];


	int * number_of_event_bools_per_neuron = new int[number_of_neurons]; // Don't think this array is needed but doesn't really matter
	int * beginning_event_bool_indices_per_neuron = new int[number_of_neurons];
	int * beginning_spike_time_int_indices_per_neuron = new int[number_of_neurons];

	int total_number_of_event_bools = 0;
	int total_number_of_spikes = 0;


	for (int neuron_index = 0; neuron_index < number_of_neurons; neuron_index++) {

		int number_of_spikes_for_neuron = 0;
		for (int stimulus_index = 0; stimulus_index < number_of_stimuli; stimulus_index++) {
			number_of_spikes_for_neuron += test_network_experiment->spike_analyser->per_stimulus_per_neuron_spike_counts[stimulus_index][neuron_index];
		}

		spikes_per_neuron[neuron_index] = number_of_spikes_for_neuron;

		// printf("number_of_spikes_for_neuron: %d\n", number_of_spikes_for_neuron);

		int number_of_afferent_synapses_for_neuron = four_layer_vision_spiking_model->spiking_neurons->per_neuron_afferent_synapse_count[neuron_index];

		number_of_event_bools_per_neuron[neuron_index] = number_of_afferent_synapses_for_neuron * number_of_spikes_for_neuron;

		int beginning_event_bool_index_of_neuron = 0;
		int beginning_spike_time_int_index_of_neuron = 0;
		if (neuron_index > 0) {
			beginning_event_bool_index_of_neuron = beginning_event_bool_indices_per_neuron[neuron_index - 1] + number_of_event_bools_per_neuron[neuron_index - 1];
			beginning_spike_time_int_index_of_neuron = beginning_spike_time_int_indices_per_neuron[neuron_index - 1] + spikes_per_neuron[neuron_index - 1];
		}
		beginning_event_bool_indices_per_neuron[neuron_index] = beginning_event_bool_index_of_neuron;
		beginning_spike_time_int_indices_per_neuron[neuron_index] = beginning_spike_time_int_index_of_neuron;

		if (neuron_index - 1 == number_of_neurons) {
			total_number_of_event_bools = beginning_event_bool_index_of_neuron;
		}

		total_number_of_spikes += number_of_spikes_for_neuron;

		temp_index_of_latest_spike_per_neuron[neuron_index] = 0;	

	}


	bool * events_as_bools_per_neuron_and_spike_data = new bool[total_number_of_event_bools];
	float * ordered_spike_times_data = new float[total_number_of_spikes];


	// TO DO:
	// PUT SPIKE TIMES INTO ordered_spike_times_data
		// use 
			// beginning_spike_time_int_indices_per_neuron

	// COPY ANYTHING USEFUL TO DEVICE
	// - events_as_bools_per_neuron_and_spike_data
	// - ordered_spike_times_data
	// - beginning_event_bool_indices_per_neuron
	// - beginning_spike_time_int_indices_per_neuron

		
	// for (int spike_id = 0; spike_id < total_number_of_spikes; spike_id++) {
	// 	int neuron_index = test_network_experiment->simulator->recording_electrodes->h_neuron_ids_of_stored_spikes_on_host[spike_id];
	// 	printf("neuron_index: %d\n", neuron_index);

	// 	float spike_time = test_network_experiment->simulator->recording_electrodes->h_time_in_seconds_of_stored_spikes_on_host[spike_id];
	// 	printf("spike_time: %f\n", spike_time);
	// }


	for (int spike_id = 0; spike_id < total_number_of_spikes; spike_id++) {

		int neuron_index = test_network_experiment->simulator->recording_electrodes->h_neuron_ids_of_stored_spikes_on_host[spike_id];
		float spike_time = test_network_experiment->simulator->recording_electrodes->h_time_in_seconds_of_stored_spikes_on_host[spike_id];

		printf("neuron_index: %d\n", neuron_index);
		printf("beginning_spike_time_int_indices_per_neuron[neuron_index]: %d\n", beginning_spike_time_int_indices_per_neuron[neuron_index]);
		printf("temp_index_of_latest_spike_per_neuron[neuron_index]: %d\n", temp_index_of_latest_spike_per_neuron[neuron_index]);


		int data_index = beginning_spike_time_int_indices_per_neuron[neuron_index] + temp_index_of_latest_spike_per_neuron[neuron_index];

		// printf("spike_time: %f\n", spike_time);
		printf("data_index: %d\n", data_index);

		ordered_spike_times_data[data_index] = spike_time;

		temp_index_of_latest_spike_per_neuron[neuron_index]++;



	}





	// for (int neuron_index = 0; neuron_index < number_of_neurons; neuron_index++) {


	// 	neuron_events_for_each_neuron[neuron_index] = new bool *[number_of_spikes_for_neuron];


	// 	for (int neuron_spike_number = 0; neuron_spike_number < number_of_spikes_for_neuron; neuron_spike_number++) {
	// 		int number_of_afferent_synapses_for_neuron = four_layer_vision_spiking_model->spiking_neurons->per_neuron_afferent_synapse_count[neuron_index];
	// 		neuron_events_for_each_neuron[neuron_index][neuron_spike_number] = new bool[number_of_afferent_synapses_for_neuron];
	// 	}

	// 	ordered_spike_times_for_each_neuron[neuron_index] = new float[number_of_spikes_for_neuron];
	// 	temp_index_of_latest_spike_per_neuron[neuron_index] = 0;

	// }	

	// // cudaMemcpy(d_ordered_spike_times_for_each_neuron, hd_ordered_spike_times_for_each_neuron, number_of_neurons*sizeof(int*), cudaMemcpyHostToDevice);


	// int total_number_of_spikes = test_network_experiment->simulator->recording_electrodes->h_total_number_of_spikes_stored_on_host;

	// for (int spike_id = 0; spike_id < total_number_of_spikes; spike_id++) {

	// 	int neuron_index = test_network_experiment->simulator->recording_electrodes->h_neuron_ids_of_stored_spikes_on_host[spike_id];
	// 	float spike_time = test_network_experiment->simulator->recording_electrodes->h_time_in_seconds_of_stored_spikes_on_host[spike_id];

	// 	ordered_spike_times_for_each_neuron[neuron_index][temp_index_of_latest_spike_per_neuron[neuron_index]] = spike_time;

	// 	temp_index_of_latest_spike_per_neuron[neuron_index]++;

	// }



	// // cudaMemcpy(d_neuron_events_for_each_neuron, neuron_events_for_each_neuron, number_of_neurons*sizeof(bool **), cudaMemcpyHostToDevice);
	// // cudaMemcpy(d_ordered_spike_times_for_each_neuron, ordered_spike_times_for_each_neuron, number_of_neurons*sizeof(float *), cudaMemcpyHostToDevice);



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