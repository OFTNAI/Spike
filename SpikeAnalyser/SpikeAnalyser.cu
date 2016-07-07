#include "SpikeAnalyser.h"

#include "../Helpers/CUDAErrorCheckHelpers.h"
#include "../Helpers/TerminalHelpers.h"
#include "../Helpers/TimerWithMessages.h"

#include <stdlib.h>

// SpikeAnalyser Constructor
SpikeAnalyser::SpikeAnalyser(Neurons * neurons_parameter, ImagePoissonInputSpikingNeurons * input_neurons_parameter) {
	neurons = neurons_parameter;
	input_neurons = input_neurons_parameter;
	number_of_neurons_in_group = 0;

	information_scores_for_each_object_and_neuron = NULL;
	descending_information_scores_for_each_object_and_neuron = NULL;

	number_of_spikes_per_stimulus_per_neuron_group = NULL;
	total_number_of_spikes_per_neuron_group = NULL;
	total_number_of_neuron_spikes = 0;

	combined_powered_distance_from_average_score_for_each_neuron_group = NULL;

	sum_of_information_scores_for_last_neuron_group = -1.0;
	number_of_neurons_with_maximum_information_score_in_last_neuron_group = -1;
	maximum_information_score_count_multiplied_by_sum_of_information_scores = -1.0;

	per_stimulus_per_neuron_spike_counts = new int*[input_neurons->total_number_of_input_stimuli];

	for (int stimulus_index = 0; stimulus_index < input_neurons->total_number_of_input_stimuli; stimulus_index++) {
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


void SpikeAnalyser::calculate_combined_powered_distance_from_average_score() {

	combined_powered_distance_from_average_score_for_each_neuron_group = new float[neurons->total_number_of_groups];

	combined_powered_distance_from_average_score = 0.0;

	for (int neuron_group_index = 0; neuron_group_index < neurons->total_number_of_groups; neuron_group_index++) {

		float optimal_average_firing_rate = 50.0f;
		float average_number_of_neuron_spikes_per_second_for_neuron_group = average_number_of_spikes_per_neuron_group_per_second[neuron_group_index];

		float neuron_group_score = 0.0;

		if (average_number_of_neuron_spikes_per_second_for_neuron_group < optimal_average_firing_rate) {
			neuron_group_score += - pow((optimal_average_firing_rate - average_number_of_neuron_spikes_per_second_for_neuron_group), 6);
		} else {
			neuron_group_score += - pow((average_number_of_neuron_spikes_per_second_for_neuron_group - optimal_average_firing_rate), 2);
		}

		combined_powered_distance_from_average_score_for_each_neuron_group[neuron_group_index] = neuron_group_score;
		combined_powered_distance_from_average_score += neuron_group_score;
	}

}

void SpikeAnalyser::calculate_various_neuron_spike_totals_and_averages(float presentation_time_per_stimulus_per_epoch) {

	TimerWithMessages * timer = new TimerWithMessages("Calculating total and per stimulus spikes per neuron group...\n");

	number_of_spikes_per_stimulus_per_neuron_group = new int *[neurons->total_number_of_groups];
	average_number_of_spikes_per_stimulus_per_neuron_group_per_second = new float *[neurons->total_number_of_groups];
	total_number_of_spikes_per_neuron_group = new int [neurons->total_number_of_groups];
	average_number_of_spikes_per_neuron_group_per_second = new float [neurons->total_number_of_groups];
	total_number_of_neuron_spikes = 0;

	for (int neuron_group_index = 0; neuron_group_index < neurons->total_number_of_groups; neuron_group_index++) {

		number_of_spikes_per_stimulus_per_neuron_group[neuron_group_index] = new int[input_neurons->total_number_of_input_stimuli];
		average_number_of_spikes_per_stimulus_per_neuron_group_per_second[neuron_group_index] = new float[input_neurons->total_number_of_input_stimuli];

		int neuron_group_start_index = neurons->start_neuron_indices_for_each_group[neuron_group_index];
		int neuron_group_end_index = neurons->last_neuron_indices_for_each_group[neuron_group_index];
		number_of_neurons_in_group = neuron_group_end_index - neuron_group_start_index + 1;
	
		total_number_of_spikes_per_neuron_group[neuron_group_index] = 0;

		for (int stimulus_index = 0; stimulus_index < input_neurons->total_number_of_input_stimuli; stimulus_index++) {

			number_of_spikes_per_stimulus_per_neuron_group[neuron_group_index][stimulus_index] = 0;

			for (int neuron_index_zeroed = 0; neuron_index_zeroed < number_of_neurons_in_group; neuron_index_zeroed++) {

				int number_of_spikes = per_stimulus_per_neuron_spike_counts[stimulus_index][neuron_index_zeroed+neuron_group_start_index];
				number_of_spikes_per_stimulus_per_neuron_group[neuron_group_index][stimulus_index] += number_of_spikes;

			}
			
			average_number_of_spikes_per_stimulus_per_neuron_group_per_second[neuron_group_index][stimulus_index] = ((float)number_of_spikes_per_stimulus_per_neuron_group[neuron_group_index][stimulus_index]/(float)number_of_neurons_in_group) / presentation_time_per_stimulus_per_epoch;
			printf("average_number_of_spikes_per_stimulus_per_neuron_group_per_second[%d][%d]: %f\n", neuron_group_index, stimulus_index, average_number_of_spikes_per_stimulus_per_neuron_group_per_second[neuron_group_index][stimulus_index]);

			total_number_of_spikes_per_neuron_group[neuron_group_index] += number_of_spikes_per_stimulus_per_neuron_group[neuron_group_index][stimulus_index];

			// printf("number_of_spikes_per_stimulus_per_neuron_group[%d][%d]: %d\n", neuron_group_index, stimulus_index, number_of_spikes_per_stimulus_per_neuron_group[neuron_group_index][stimulus_index]);
			// printf("total_number_of_spikes_per_neuron_group[neuron_group_index]: %d\n", total_number_of_spikes_per_neuron_group[neuron_group_index]);

		}

		average_number_of_spikes_per_neuron_group_per_second[neuron_group_index] = (((float)total_number_of_spikes_per_neuron_group[neuron_group_index] / (float)number_of_neurons_in_group) / presentation_time_per_stimulus_per_epoch) / (float)input_neurons->total_number_of_input_stimuli;
		printf("average_number_of_spikes_per_neuron_group_per_second[neuron_group_index]: %f\n", average_number_of_spikes_per_neuron_group_per_second[neuron_group_index]);

		total_number_of_neuron_spikes += total_number_of_spikes_per_neuron_group[neuron_group_index];

	}

	average_number_of_neuron_spikes_per_second = ((float)total_number_of_neuron_spikes / (float)neurons->total_number_of_neurons) / presentation_time_per_stimulus_per_epoch;

	// printf("total_number_of_neuron_spikes: %d\n", total_number_of_neuron_spikes);
	printf("average_number_of_neuron_spikes_per_second: %f\n", average_number_of_neuron_spikes_per_second);

	timer->stop_timer_and_log_time_and_message("Total and per stimulus spikes per neuron group calculated.", true);

}

void SpikeAnalyser::calculate_single_cell_information_scores_for_neuron_group(int neuron_group_index, int number_of_bins) {

	TimerWithMessages * single_cell_analysis_timer = new TimerWithMessages();
	printf("Calculating Single Cell Information Scores for Neuron Group: %d\n", neuron_group_index);

	int neuron_group_start_index = neurons->start_neuron_indices_for_each_group[neuron_group_index];
	int neuron_group_end_index = neurons->last_neuron_indices_for_each_group[neuron_group_index];
	number_of_neurons_in_group = neuron_group_end_index - neuron_group_start_index + 1;

	// 1. Find max number of spikes
	int max_number_of_spikes = 0;
	for (int stimulus_index = 0; stimulus_index < input_neurons->total_number_of_input_stimuli; stimulus_index++) {
		for (int neuron_index = neuron_group_start_index; neuron_index <= neuron_group_end_index; neuron_index++) {
			int number_of_spikes = per_stimulus_per_neuron_spike_counts[stimulus_index][neuron_index];
			if (max_number_of_spikes < number_of_spikes) {
				max_number_of_spikes = number_of_spikes;
			}
		}
	}

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


	if (max_number_of_spikes == 0) {
		printf("No spikes in neuron group: %d\n", neuron_group_index);
		print_line_of_dashes_with_blank_lines_either_side();
		sum_of_information_scores_for_last_neuron_group = 0;
		number_of_neurons_with_maximum_information_score_in_last_neuron_group = 0;
		maximum_information_score_count_multiplied_by_sum_of_information_scores = 0;
		return;
	}

	// 2. Calculate bin index for all spike counts + create bin counts for each neuron
	// First set up arrays
	int ** bin_indices_per_stimulus_and_per_neuron = new int*[input_neurons->total_number_of_input_stimuli];
	for (int stimulus_index = 0; stimulus_index < input_neurons->total_number_of_input_stimuli; stimulus_index++) {
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

	for (int stimulus_index = 0; stimulus_index < input_neurons->total_number_of_input_stimuli; stimulus_index++) {
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
			probabilities_of_bin_responses_p_r[bin_index][neuron_index_zeroed] = (float) individual_bin_counts_for_each_neuron[bin_index][neuron_index_zeroed] / (float) input_neurons->total_number_of_input_stimuli;
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
	sum_of_information_scores_for_last_neuron_group = 0.0;
	number_of_neurons_with_maximum_information_score_in_last_neuron_group = 0;
	float maximum_possible_information_score = log2((float)input_neurons->total_number_of_objects);
	// printf("maximum_possible_information_score: %f\n", maximum_possible_information_score);
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

			sum_of_information_scores_for_last_neuron_group += information_sum;
			if (maximum_possible_information_score <= information_sum) number_of_neurons_with_maximum_information_score_in_last_neuron_group++;
		}
	}

	maximum_information_score_count_multiplied_by_sum_of_information_scores = (float)number_of_neurons_with_maximum_information_score_in_last_neuron_group * sum_of_information_scores_for_last_neuron_group;

	

	//6. Sort information scores for each object and neuron
	for (int object_index = 0; object_index < input_neurons->total_number_of_objects; object_index++) {
		std::sort(descending_information_scores_for_each_object_and_neuron[object_index], descending_information_scores_for_each_object_and_neuron[object_index] + number_of_neurons_in_group, std::greater<float>());
	}

	single_cell_analysis_timer->stop_timer_and_log_time_and_message("Single Cell Information Scores Calulated Neuron Group.", false);
	printf("--- sum_of_information_scores_for_last_neuron_group: %f\n", sum_of_information_scores_for_last_neuron_group);
	printf("--- number_of_neurons_with_maximum_information_score_in_last_neuron_group: %d\n", number_of_neurons_with_maximum_information_score_in_last_neuron_group);
	printf("--- maximum_information_score_count_multiplied_by_sum_of_information_scores: %f\n", maximum_information_score_count_multiplied_by_sum_of_information_scores);
	print_line_of_dashes_with_blank_lines_either_side();


}


