#include "SpikeAnalyser.h"

#include "../Helpers/CUDAErrorCheckHelpers.h"

#include <stdlib.h>

// SpikeAnalyser Constructor
SpikeAnalyser::SpikeAnalyser(Neurons * neurons_parameter, ImagePoissonSpikingNeurons * input_neurons_parameter) {
	neurons = neurons_parameter;
	input_neurons = input_neurons_parameter;
	number_of_neurons_in_group = 0;

	information_scores_for_each_object_and_neuron = NULL;
	descending_information_scores_for_each_object_and_neuron = NULL;

	per_stimulus_per_neuron_spike_counts = new int*[input_neurons->total_number_of_input_images];

	for (int stimulus_index = 0; stimulus_index < input_neurons->total_number_of_input_images; stimulus_index++) {
		per_stimulus_per_neuron_spike_counts[stimulus_index] = new int[neurons->total_number_of_neurons];
	}
	
}


// SpikeAnalyser Destructor
SpikeAnalyser::~SpikeAnalyser() {

}


void SpikeAnalyser::store_spike_counts_for_stimulus_index(int stimulus_index, int * d_neuron_spike_counts_for_stimulus) {
	
	CudaSafeCall(cudaMemcpy(per_stimulus_per_neuron_spike_counts[stimulus_index], 
									d_neuron_spike_counts_for_stimulus, 
									sizeof(float) * neurons->total_number_of_neurons, 
									cudaMemcpyDeviceToHost));

}

void SpikeAnalyser::calculate_single_cell_information_scores_for_neuron_group(int neuron_group_index, int number_of_bins) {

	int neuron_group_start_index = neurons->start_neuron_indices_for_each_group[neuron_group_index];
	int neuron_group_end_index = neurons->last_neuron_indices_for_each_group[neuron_group_index];
	number_of_neurons_in_group = neuron_group_end_index - neuron_group_start_index + 1;

	// 1. Find max number of spikes
	int max_number_of_spikes = 0;
	for (int stimulus_index = 0; stimulus_index < input_neurons->total_number_of_input_images; stimulus_index++) {
		for (int neuron_index = neuron_group_start_index; neuron_index <= neuron_group_end_index; neuron_index++) {
			int number_of_spikes = per_stimulus_per_neuron_spike_counts[stimulus_index][neuron_index];
			if (max_number_of_spikes < number_of_spikes) {
				max_number_of_spikes = number_of_spikes;
			}
		}
	}


	// 2. Calculate bin index for all spike counts + create bin counts for each neuron
	// First set up arrays
	int ** bin_indices_per_stimulus_and_per_neuron = new int*[input_neurons->total_number_of_input_images];
	for (int stimulus_index = 0; stimulus_index < input_neurons->total_number_of_input_images; stimulus_index++) {
		bin_indices_per_stimulus_and_per_neuron[stimulus_index] = new int[number_of_neurons_in_group];
	}

	int ** individual_bin_counts_for_each_neuron = new int*[number_of_bins];
	float ** probabilities_of_bin_responses_p_r = new float*[number_of_bins];
	for (int bin_index = 0; bin_index < number_of_bins; bin_index++) {
		individual_bin_counts_for_each_neuron[bin_index] = new int[number_of_neurons_in_group];
		probabilities_of_bin_responses_p_r[bin_index] = new float[number_of_neurons_in_group];
		for (int neuron_index = 0; neuron_index < number_of_neurons_in_group; neuron_index++) {
			individual_bin_counts_for_each_neuron[bin_index][neuron_index] = 0;
			probabilities_of_bin_responses_p_r[bin_index][neuron_index] = 0.0;
		}
	}


	for (int stimulus_index = 0; stimulus_index < input_neurons->total_number_of_input_images; stimulus_index++) {
		for (int neuron_index_zeroed = 0; neuron_index_zeroed < number_of_neurons_in_group; neuron_index_zeroed++) {

			int number_of_spikes = per_stimulus_per_neuron_spike_counts[stimulus_index][neuron_index_zeroed+neuron_group_start_index];
			float ratio_of_max_number_of_spikes = (float)number_of_spikes / (float)max_number_of_spikes;

			int bin_index = min(int(floor(ratio_of_max_number_of_spikes * number_of_bins)), number_of_bins - 1);
			bin_indices_per_stimulus_and_per_neuron[stimulus_index][neuron_index_zeroed] = bin_index;

			individual_bin_counts_for_each_neuron[bin_index][neuron_index_zeroed]++;
		}
	}


	// 3. Calculate probabilities_of_bin_responses_p_r
	for (int bin_index = 0; bin_index < number_of_bins; bin_index++) {
		for (int neuron_index_zeroed = 0; neuron_index_zeroed < number_of_neurons_in_group; neuron_index_zeroed++) {
			probabilities_of_bin_responses_p_r[bin_index][neuron_index_zeroed] = (float) individual_bin_counts_for_each_neuron[bin_index][neuron_index_zeroed] / (float) input_neurons->total_number_of_input_images;
			// printf("probabilities_of_bin_responses_p_r[bin_index][neuron_index_zeroed]: %f\n", probabilities_of_bin_responses_p_r[bin_index][neuron_index_zeroed]);
		}
	}

	// 4. Calculate probabilities_of_bin_responses_given_an_object_p_r_given_s
	// First set up arrays
	float *** probabilities_of_bin_responses_given_an_object_p_r_given_s = new float**[input_neurons->total_number_of_objects];
	for (int object_index = 0; object_index < input_neurons->total_number_of_objects; object_index++) {
		probabilities_of_bin_responses_given_an_object_p_r_given_s[object_index] = new float*[number_of_neurons_in_group];
		for (int neuron_index_zeroed = 0; neuron_index_zeroed < number_of_neurons_in_group; neuron_index_zeroed++) {
			probabilities_of_bin_responses_given_an_object_p_r_given_s[object_index][neuron_index_zeroed] = new float[number_of_bins];
			for (int bin_index = 0; bin_index < number_of_bins; bin_index++) {
				probabilities_of_bin_responses_given_an_object_p_r_given_s[object_index][neuron_index_zeroed][bin_index] = 0.0;
			}
		}
	}


	for (int object_index = 0; object_index < input_neurons->total_number_of_objects; object_index++) {
		for (int neuron_index_zeroed = 0; neuron_index_zeroed < number_of_neurons_in_group; neuron_index_zeroed++) {
			for (int bin_index = 0; bin_index < number_of_bins; bin_index++) {

				int bin_count_for_object = 0;
				for (int transform_index = 0; transform_index < input_neurons->total_number_of_transformations_per_object; transform_index++) {
					int stimulus_index = object_index * input_neurons->total_number_of_transformations_per_object + transform_index;
					if (bin_indices_per_stimulus_and_per_neuron[stimulus_index][neuron_index_zeroed] == bin_index) bin_count_for_object++;
				}
				probabilities_of_bin_responses_given_an_object_p_r_given_s[object_index][neuron_index_zeroed][bin_index] = (float)bin_count_for_object / (float)input_neurons->total_number_of_transformations_per_object;
				// printf("probabilities_of_bin_responses_given_an_object_p_r_given_s[object_index][neuron_index_zeroed][bin_index]: %f\n", probabilities_of_bin_responses_given_an_object_p_r_given_s[object_index][neuron_index_zeroed][bin_index]);
			}
		}
	}


	// 5. Calculate information_scores_for_each_object_and_neuron
	// First set up arrays
	information_scores_for_each_object_and_neuron = new float *[input_neurons->total_number_of_objects];
	descending_information_scores_for_each_object_and_neuron = new float *[input_neurons->total_number_of_objects];

	for (int object_index = 0; object_index < input_neurons->total_number_of_objects; object_index++) {
		information_scores_for_each_object_and_neuron[object_index] = new float[number_of_neurons_in_group];
		descending_information_scores_for_each_object_and_neuron[object_index] = new float[number_of_neurons_in_group];
		for (int neuron_index_zeroed = 0; neuron_index_zeroed < number_of_neurons_in_group; neuron_index_zeroed++) {
			information_scores_for_each_object_and_neuron[object_index][neuron_index_zeroed] = 0.0;
			descending_information_scores_for_each_object_and_neuron[object_index][neuron_index_zeroed] = 0.0;
		}
	}

	for (int object_index = 0; object_index < input_neurons->total_number_of_objects; object_index++) {
		for (int neuron_index_zeroed = 0; neuron_index_zeroed < number_of_neurons_in_group; neuron_index_zeroed++) {
			float information_sum = 0.0;
			for (int bin_index = 0; bin_index < number_of_bins; bin_index++) {
				float p_r_given_s = probabilities_of_bin_responses_given_an_object_p_r_given_s[object_index][neuron_index_zeroed][bin_index];
				float p_r = probabilities_of_bin_responses_p_r[bin_index][neuron_index_zeroed];
				if ((p_r > 0.0) && (p_r_given_s > 0.0)){
					float fraction = p_r_given_s / p_r;
					float log_of_fraction = log2(fraction);
					float new_component = p_r_given_s * log_of_fraction;
					information_sum += new_component;
				}
			}
			information_scores_for_each_object_and_neuron[object_index][neuron_index_zeroed] = information_sum;
			descending_information_scores_for_each_object_and_neuron[object_index][neuron_index_zeroed] = information_sum;
			// printf("information_sum: %f\n", information_sum);
		}
	}


	//6. Sort information scores for each object and neuron
	for (int object_index = 0; object_index < input_neurons->total_number_of_objects; object_index++) {
		std::sort(descending_information_scores_for_each_object_and_neuron[object_index], descending_information_scores_for_each_object_and_neuron[object_index] + number_of_neurons_in_group, std::greater<float>());
	}




}


