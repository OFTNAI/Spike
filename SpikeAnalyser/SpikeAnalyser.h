#ifndef SpikeAnalyser_H
#define SpikeAnalyser_H

#include <cuda.h>
#include "../Neurons/Neurons.h"
#include "../Neurons/ImagePoissonInputSpikingNeurons.h"

class SpikeAnalyser{
public:

	// Constructor/Destructor
	SpikeAnalyser(Neurons *neurons_parameter, ImagePoissonInputSpikingNeurons *input_neurons_parameter);
	~SpikeAnalyser();

	Neurons * neurons;
	ImagePoissonInputSpikingNeurons * input_neurons;
	
	int number_of_neurons_in_group;

	int ** per_stimulus_per_neuron_spike_counts;

	float ** information_scores_for_each_object_and_neuron;
	float ** descending_information_scores_for_each_object_and_neuron;
	float * maximum_information_score_for_each_neuron;
	float * descending_maximum_information_score_for_each_neuron;

	int ** number_of_spikes_per_stimulus_per_neuron_group;
	float ** average_number_of_spikes_per_stimulus_per_neuron_group_per_second;
	int * total_number_of_spikes_per_neuron_group;
	float * average_number_of_spikes_per_neuron_group_per_second;
	int total_number_of_neuron_spikes;
	float average_number_of_neuron_spikes_per_second;

	float combined_powered_distance_from_average_score;
	float * combined_powered_distance_from_average_score_for_each_neuron_group;



	int number_of_neurons_with_maximum_information_score_in_last_neuron_group;
	float sum_of_information_scores_for_last_neuron_group;
	float maximum_information_score_count_multiplied_by_sum_of_information_scores;

	void store_spike_counts_for_stimulus_index(int stimulus_index, int * d_neuron_spike_counts_for_stimulus);
	void calculate_various_neuron_spike_totals_and_averages(float presentation_time_per_stimulus_per_epoch);
	void calculate_combined_powered_distance_from_average_score();
	void calculate_single_cell_information_scores_for_neuron_group(int neuron_group_index, int number_of_bins);

};

#endif