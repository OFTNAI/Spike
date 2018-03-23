#include "SpikeAnalyser.hpp"

#include "../Helpers/TerminalHelpers.hpp"
#include "../Helpers/TimerWithMessages.hpp"

#include <stdlib.h>
#include <functional>
#include <algorithm>
#include <cmath>

// SpikeAnalyser Constructor
SpikeAnalyser::SpikeAnalyser(SpikingNeurons *neurons_parameter,
                             InputSpikingNeurons *input_neurons_parameter,
                             CountNeuronSpikesRecordingElectrodes *electrodes_parameter) {
	neurons = neurons_parameter;
	input_neurons = input_neurons_parameter;
        count_electrodes = electrodes_parameter;
        assert(count_electrodes != nullptr &&
               "You need to have some electrodes!");
//	number_of_neurons_in_single_cell_analysis_group = 0;

	information_scores_for_each_object_and_neuron = nullptr;
	descending_information_scores_for_each_object_and_neuron = nullptr;
	maximum_information_score_for_each_neuron = nullptr;

	spike_totals_and_averages_were_calculated = false;

	number_of_spikes_per_stimulus_per_neuron_group = nullptr;
	total_number_of_spikes_per_neuron_group = nullptr;
	total_number_of_neuron_spikes = 0;

	maximum_possible_information_score = 0.0;

	combined_powered_distance_from_average_score_for_each_neuron_group = nullptr;
	combined_powered_distance_from_max_score_for_each_neuron_group = nullptr;

	sum_of_information_scores_for_last_neuron_group = -1.0;
	number_of_neurons_with_maximum_information_score_in_last_neuron_group = -1;
	maximum_information_score_count_multiplied_by_sum_of_information_scores = -1.0;

	per_stimulus_per_neuron_spike_counts = new int*[input_neurons->total_number_of_input_stimuli];
	

	for (int stimulus_index = 0; stimulus_index < input_neurons->total_number_of_input_stimuli; stimulus_index++) {
		per_stimulus_per_neuron_spike_counts[stimulus_index] = new int[neurons->total_number_of_neurons];
	}
	
}


SpikeAnalyser::~SpikeAnalyser() {
  // I hacked this together very fast, so it may be incorrect! --TSCS

  delete[] per_stimulus_per_neuron_spike_counts;
  delete[] information_scores_for_each_object_and_neuron;
  delete[] descending_information_scores_for_each_object_and_neuron;
  delete[] number_of_spikes_per_stimulus_per_neuron_group;
  delete[] average_number_of_spikes_per_stimulus_per_neuron_group_per_second;
  delete[] average_number_of_spikes_per_stimulus_per_neuron_group_per_second_excluding_silent_neurons;

  delete maximum_information_score_for_each_neuron;
  delete descending_maximum_information_score_for_each_neuron;
  delete average_information_score_for_each_neuron;
  delete descending_average_information_score_for_each_neuron;
  delete total_number_of_spikes_per_neuron_group;
  delete average_number_of_spikes_per_neuron_group_per_second;
  delete average_number_of_spikes_per_neuron_group_per_second_excluding_silent_neurons;
  delete max_number_of_spikes_per_neuron_group_per_second;
  delete running_count_of_non_silent_neurons_per_neuron_group;
  delete combined_powered_distance_from_average_score_for_each_neuron_group;
  delete combined_powered_distance_from_max_score_for_each_neuron_group;
}


void SpikeAnalyser::reset_state() {
  backend()->reset_state();
}

void SpikeAnalyser::store_spike_counts_for_stimulus_index(int stimulus_index) {
  backend()->store_spike_counts_for_stimulus_index(stimulus_index);
}


void SpikeAnalyser::calculate_fitness_score(float optimal_average_firing_rate, float optimal_max_firing_rate) {
	int comparisonType = 1; //0:James, 1:aki
	float maxScore = 10; //for aki
	combined_powered_distance_from_average_score_for_each_neuron_group = new float[neurons->total_number_of_groups];
	combined_powered_distance_from_max_score_for_each_neuron_group = new float[neurons->total_number_of_groups];
			
	combined_powered_distance_from_average_score = 0.0;
	combined_powered_distance_from_max_score = 0.0;

	for (int neuron_group_index = 0; neuron_group_index < neurons->total_number_of_groups; neuron_group_index++) {

		float average_number_of_neuron_spikes_per_second_for_neuron_group = average_number_of_spikes_per_neuron_group_per_second[neuron_group_index];
		float max_number_of_neuron_spikes_per_second_for_neuron_group = max_number_of_spikes_per_neuron_group_per_second[neuron_group_index];

		float neuron_group_score_based_on_avg = 0.0;
		float neuron_group_score_based_on_max = 0.0;

		if (comparisonType==0){
			if (average_number_of_neuron_spikes_per_second_for_neuron_group < optimal_average_firing_rate) {
				neuron_group_score_based_on_avg += - pow((optimal_average_firing_rate - average_number_of_neuron_spikes_per_second_for_neuron_group), 6);
			} else {
				neuron_group_score_based_on_avg += - pow((average_number_of_neuron_spikes_per_second_for_neuron_group - optimal_average_firing_rate), 2);
			}
			
			if (max_number_of_neuron_spikes_per_second_for_neuron_group < optimal_max_firing_rate) {
				neuron_group_score_based_on_max += - pow((optimal_max_firing_rate - max_number_of_neuron_spikes_per_second_for_neuron_group), 6);
			} else {
				neuron_group_score_based_on_max += - pow((max_number_of_neuron_spikes_per_second_for_neuron_group - optimal_max_firing_rate), 2);
			}
			
		}else if(comparisonType==1){
			if (average_number_of_neuron_spikes_per_second_for_neuron_group < optimal_average_firing_rate) {
				neuron_group_score_based_on_avg = maxScore * (1- (optimal_average_firing_rate - average_number_of_neuron_spikes_per_second_for_neuron_group)/optimal_average_firing_rate);
			} else {
				neuron_group_score_based_on_avg = maxScore/(std::fabs(average_number_of_neuron_spikes_per_second_for_neuron_group - optimal_average_firing_rate)+1);
			}
//			neuron_group_score = 100.0/(abs(average_number_of_neuron_spikes_per_second_for_neuron_group - optimal_average_firing_rate)+1);

			
			if (max_number_of_neuron_spikes_per_second_for_neuron_group < optimal_max_firing_rate) {
				neuron_group_score_based_on_max = maxScore * (1- (optimal_max_firing_rate - max_number_of_neuron_spikes_per_second_for_neuron_group)/optimal_max_firing_rate);
			} else {
				neuron_group_score_based_on_max = maxScore/(std::fabs(max_number_of_neuron_spikes_per_second_for_neuron_group - optimal_max_firing_rate)+1);
			}
		}
		
		
		combined_powered_distance_from_average_score_for_each_neuron_group[neuron_group_index] = neuron_group_score_based_on_avg;
		combined_powered_distance_from_average_score += neuron_group_score_based_on_avg;
//		printf("Dakota: combined_powered_distance_from_average_score_for_each_neuron_group[%d]: %f score: %f \n", neuron_group_index, average_number_of_neuron_spikes_per_second_for_neuron_group, neuron_group_score);

		combined_powered_distance_from_max_score_for_each_neuron_group[neuron_group_index] = neuron_group_score_based_on_max;
		combined_powered_distance_from_max_score += neuron_group_score_based_on_max;
//		printf("Dakota: combined_powered_distance_from_max_score_for_each_neuron_group[%d]: %f score: %f \n", neuron_group_index, max_number_of_neuron_spikes_per_second_for_neuron_group, neuron_group_score2);

	}

}


void SpikeAnalyser::calculate_various_neuron_spike_totals_and_averages(float presentation_time_per_stimulus_per_epoch) {

	TimerWithMessages * timer = new TimerWithMessages("Calculating total and per stimulus spikes per neuron group...\n");

	number_of_spikes_per_stimulus_per_neuron_group = new int *[neurons->total_number_of_groups];
	average_number_of_spikes_per_stimulus_per_neuron_group_per_second = new float *[neurons->total_number_of_groups];
        average_number_of_spikes_per_stimulus_per_neuron_group_per_second_excluding_silent_neurons = new float *[neurons->total_number_of_groups];
	total_number_of_spikes_per_neuron_group = new int [neurons->total_number_of_groups];
	average_number_of_spikes_per_neuron_group_per_second = new float [neurons->total_number_of_groups];
        average_number_of_spikes_per_neuron_group_per_second_excluding_silent_neurons = new float [neurons->total_number_of_groups];
	max_number_of_spikes_per_neuron_group_per_second = new float [neurons->total_number_of_groups];
        running_count_of_non_silent_neurons_per_neuron_group = new int [neurons->total_number_of_groups];
	total_number_of_neuron_spikes = 0;
	

	for (int neuron_group_index = 0; neuron_group_index < neurons->total_number_of_groups; neuron_group_index++) {

		number_of_spikes_per_stimulus_per_neuron_group[neuron_group_index] = new int[input_neurons->total_number_of_input_stimuli];
		average_number_of_spikes_per_stimulus_per_neuron_group_per_second[neuron_group_index] = new float[input_neurons->total_number_of_input_stimuli];
                average_number_of_spikes_per_stimulus_per_neuron_group_per_second_excluding_silent_neurons[neuron_group_index] = new float[input_neurons->total_number_of_input_stimuli];

		int neuron_group_start_index = neurons->start_neuron_indices_for_each_group[neuron_group_index];
		int neuron_group_end_index = neurons->last_neuron_indices_for_each_group[neuron_group_index];
		int number_of_neurons_in_group = neuron_group_end_index - neuron_group_start_index + 1;
		printf("\n* neuron group %d (%d neurons) *\n", neuron_group_index,number_of_neurons_in_group);
	
		total_number_of_spikes_per_neuron_group[neuron_group_index] = 0;

                int running_count_of_non_silent_neurons_in_group = 0;

		int tmp_max_number_of_neuron_spikes = 0;
		for (int stimulus_index = 0; stimulus_index < input_neurons->total_number_of_input_stimuli; stimulus_index++) {
			int tmp_max_number_of_neuron_spikes_per_stimulus = 0;
                        int total_number_of_non_silent_neurons_in_group_for_stimulus = 0;
			number_of_spikes_per_stimulus_per_neuron_group[neuron_group_index][stimulus_index] = 0;
			for (int neuron_index_zeroed = 0; neuron_index_zeroed < number_of_neurons_in_group; neuron_index_zeroed++) {

				int number_of_spikes = per_stimulus_per_neuron_spike_counts[stimulus_index][neuron_index_zeroed+neuron_group_start_index];
				number_of_spikes_per_stimulus_per_neuron_group[neuron_group_index][stimulus_index] += number_of_spikes;

                                if (number_of_spikes != 0) {
					total_number_of_non_silent_neurons_in_group_for_stimulus++;
					running_count_of_non_silent_neurons_in_group++;
				}
				
				if (tmp_max_number_of_neuron_spikes<per_stimulus_per_neuron_spike_counts[stimulus_index][neuron_index_zeroed+neuron_group_start_index])
					tmp_max_number_of_neuron_spikes = per_stimulus_per_neuron_spike_counts[stimulus_index][neuron_index_zeroed+neuron_group_start_index];
				
				if (tmp_max_number_of_neuron_spikes_per_stimulus<per_stimulus_per_neuron_spike_counts[stimulus_index][neuron_index_zeroed+neuron_group_start_index])
					tmp_max_number_of_neuron_spikes_per_stimulus = per_stimulus_per_neuron_spike_counts[stimulus_index][neuron_index_zeroed+neuron_group_start_index];
			}
                        // printf("total_number_of_non_silent_neurons_in_group_for_stimulus: %d\n", total_number_of_non_silent_neurons_in_group_for_stimulus);

			average_number_of_spikes_per_stimulus_per_neuron_group_per_second[neuron_group_index][stimulus_index] = ((float)number_of_spikes_per_stimulus_per_neuron_group[neuron_group_index][stimulus_index]/(float)number_of_neurons_in_group) / presentation_time_per_stimulus_per_epoch;
                        average_number_of_spikes_per_stimulus_per_neuron_group_per_second_excluding_silent_neurons[neuron_group_index][stimulus_index] = (total_number_of_non_silent_neurons_in_group_for_stimulus == 0) ? 0.0 : ((float)number_of_spikes_per_stimulus_per_neuron_group[neuron_group_index][stimulus_index]/(float)total_number_of_non_silent_neurons_in_group_for_stimulus) / presentation_time_per_stimulus_per_epoch;
                        printf("\tnumber of spikes per neuron for stimulus %d -- max: %f\tavg: %f\tavg excluding silent: %f\n", stimulus_index, ((float)tmp_max_number_of_neuron_spikes_per_stimulus)/presentation_time_per_stimulus_per_epoch,average_number_of_spikes_per_stimulus_per_neuron_group_per_second[neuron_group_index][stimulus_index], average_number_of_spikes_per_stimulus_per_neuron_group_per_second_excluding_silent_neurons[neuron_group_index][stimulus_index]);
 
			total_number_of_spikes_per_neuron_group[neuron_group_index] += number_of_spikes_per_stimulus_per_neuron_group[neuron_group_index][stimulus_index];

			// printf("number_of_spikes_per_stimulus_per_neuron_group[%d][%d]: %d\n", neuron_group_index, stimulus_index, number_of_spikes_per_stimulus_per_neuron_group[neuron_group_index][stimulus_index]);
			// printf("total_number_of_spikes_per_neuron_group[neuron_group_index]: %d\n", total_number_of_spikes_per_neuron_group[neuron_group_index]);

		}
		
		max_number_of_spikes_per_neuron_group_per_second[neuron_group_index] = ((float)tmp_max_number_of_neuron_spikes)/presentation_time_per_stimulus_per_epoch;
		
		average_number_of_spikes_per_neuron_group_per_second[neuron_group_index] = (((float)total_number_of_spikes_per_neuron_group[neuron_group_index] / (float)number_of_neurons_in_group) / presentation_time_per_stimulus_per_epoch) / (float)input_neurons->total_number_of_input_stimuli;
                average_number_of_spikes_per_neuron_group_per_second_excluding_silent_neurons[neuron_group_index] = (running_count_of_non_silent_neurons_in_group == 0) ? 0.0 : (((float)total_number_of_spikes_per_neuron_group[neuron_group_index] / (float)running_count_of_non_silent_neurons_in_group) / presentation_time_per_stimulus_per_epoch);
		printf("-- summary -- number of spikes per neuron -- max: %f\tavg: %f\n", max_number_of_spikes_per_neuron_group_per_second[neuron_group_index], average_number_of_spikes_per_neuron_group_per_second[neuron_group_index]);

		total_number_of_neuron_spikes += total_number_of_spikes_per_neuron_group[neuron_group_index];

                running_count_of_non_silent_neurons_per_neuron_group[neuron_group_index] = running_count_of_non_silent_neurons_in_group;

	}

	average_number_of_neuron_spikes_per_second = ((float)total_number_of_neuron_spikes / (float)neurons->total_number_of_neurons) / presentation_time_per_stimulus_per_epoch / (float)input_neurons->total_number_of_input_stimuli;

	// printf("total_number_of_neuron_spikes: %d\n", total_number_of_neuron_spikes);
	printf("* average_number_of_neuron_spikes_per_second: %f\n", average_number_of_neuron_spikes_per_second);

	spike_totals_and_averages_were_calculated = true;
	timer->stop_timer_and_log_time_and_message("Total and per stimulus spikes per neuron group calculated.", true);

}

void SpikeAnalyser::calculate_single_cell_information_scores_for_neuron_group(int neuron_group_index, int number_of_bins, bool useThresholdForMaxFR, float max_firing_rate) {

	TimerWithMessages * single_cell_analysis_timer = new TimerWithMessages();
	float sum_of_maximum_information_score_for_each_neuron = 0;

	printf("Calculating Single Cell Information Scores for Neuron Group: %d\n", neuron_group_index);

	if (spike_totals_and_averages_were_calculated == false) print_message_and_exit("Please call calculate_various_neuron_spike_totals_and_averages before calculating single cell information scores");
	int neuron_group_start_index = neurons->start_neuron_indices_for_each_group[neuron_group_index];
	int neuron_group_end_index = neurons->last_neuron_indices_for_each_group[neuron_group_index];
	int number_of_neurons_in_single_cell_analysis_group= neuron_group_end_index - neuron_group_start_index + 1;
	number_of_neurons_in_single_cell_analysis_group_vec.push_back(number_of_neurons_in_single_cell_analysis_group);
	
	// 0. First initialise public variables. This is important to do first in case there are no spikes at step 1.
	maximum_possible_information_score = log2((float)input_neurons->total_number_of_objects);
	sum_of_information_scores_for_last_neuron_group = 0;
	
	number_of_neurons_with_maximum_information_score_in_last_neuron_group = 0;
	maximum_information_score_count_multiplied_by_sum_of_information_scores = 0;
	
	information_scores_for_each_object_and_neuron = new float *[input_neurons->total_number_of_objects];
	descending_information_scores_for_each_object_and_neuron = new float *[input_neurons->total_number_of_objects];
	
	for (int object_index = 0; object_index < input_neurons->total_number_of_objects; object_index++) {

		information_scores_for_each_object_and_neuron[object_index] = new float[number_of_neurons_in_single_cell_analysis_group];
		descending_information_scores_for_each_object_and_neuron[object_index] = new float[number_of_neurons_in_single_cell_analysis_group];
		for (int neuron_index_zeroed = 0; neuron_index_zeroed < number_of_neurons_in_single_cell_analysis_group; neuron_index_zeroed++) {
			information_scores_for_each_object_and_neuron[object_index][neuron_index_zeroed] = 0.0;
			descending_information_scores_for_each_object_and_neuron[object_index][neuron_index_zeroed] = 0.0;
		}
	}
	
	
	
	maximum_information_score_for_each_neuron = new float[number_of_neurons_in_single_cell_analysis_group];
	descending_maximum_information_score_for_each_neuron = new float[number_of_neurons_in_single_cell_analysis_group];
	average_information_score_for_each_neuron = new float[number_of_neurons_in_single_cell_analysis_group];
	descending_average_information_score_for_each_neuron = new float[number_of_neurons_in_single_cell_analysis_group];
		
	
	for (int neuron_index_zeroed = 0; neuron_index_zeroed < number_of_neurons_in_single_cell_analysis_group; neuron_index_zeroed++) {
		maximum_information_score_for_each_neuron[neuron_index_zeroed] = 0.0f;
		descending_maximum_information_score_for_each_neuron[neuron_index_zeroed] = 0.0f;
		average_information_score_for_each_neuron[neuron_index_zeroed] = 0.0f;
		descending_average_information_score_for_each_neuron[neuron_index_zeroed] = 0.0f;
	}

	// 1. Find max number of spikes
	int max_number_of_spikes = 0;
	if(useThresholdForMaxFR){
		max_number_of_spikes = max_firing_rate;
	}else{
		for (int stimulus_index = 0; stimulus_index < input_neurons->total_number_of_input_stimuli; stimulus_index++) {
			for (int neuron_index = neuron_group_start_index; neuron_index <= neuron_group_end_index; neuron_index++) {
				int number_of_spikes = per_stimulus_per_neuron_spike_counts[stimulus_index][neuron_index];
				if (max_number_of_spikes < number_of_spikes) {
					max_number_of_spikes = number_of_spikes;
				}
			}
		}
	}

	if (max_number_of_spikes == 0) {
		printf("No spikes in neuron group: %d\n", neuron_group_index);
		print_line_of_dashes_with_blank_lines_either_side();
		return;
	}

	// 2. Calculate bin index for all spike counts + create bin counts for each neuron
	// First set up arrays
	int ** bin_indices_per_stimulus_and_per_neuron = new int*[input_neurons->total_number_of_input_stimuli];
	for (int stimulus_index = 0; stimulus_index < input_neurons->total_number_of_input_stimuli; stimulus_index++) {
		bin_indices_per_stimulus_and_per_neuron[stimulus_index] = new int[number_of_neurons_in_single_cell_analysis_group];
	}
	int ** individual_bin_counts_for_each_neuron = new int*[number_of_bins];
	float ** probabilities_of_bin_responses_p_r = new float*[number_of_bins];
	for (int bin_index = 0; bin_index < number_of_bins; bin_index++) {
		individual_bin_counts_for_each_neuron[bin_index] = new int[number_of_neurons_in_single_cell_analysis_group];
		probabilities_of_bin_responses_p_r[bin_index] = new float[number_of_neurons_in_single_cell_analysis_group];
		for (int neuron_index = 0; neuron_index < number_of_neurons_in_single_cell_analysis_group; neuron_index++) {
			individual_bin_counts_for_each_neuron[bin_index][neuron_index] = 0;
			probabilities_of_bin_responses_p_r[bin_index][neuron_index] = 0.0;
		}
	}
	for (int stimulus_index = 0; stimulus_index < input_neurons->total_number_of_input_stimuli; stimulus_index++) {
		for (int neuron_index_zeroed = 0; neuron_index_zeroed < number_of_neurons_in_single_cell_analysis_group; neuron_index_zeroed++) {

			int number_of_spikes = per_stimulus_per_neuron_spike_counts[stimulus_index][neuron_index_zeroed+neuron_group_start_index];
			float ratio_of_max_number_of_spikes = (float)number_of_spikes / (float)max_number_of_spikes;

			int bin_index = std::min(int(floor(ratio_of_max_number_of_spikes * number_of_bins)), number_of_bins - 1);
			bin_indices_per_stimulus_and_per_neuron[stimulus_index][neuron_index_zeroed] = bin_index;

			individual_bin_counts_for_each_neuron[bin_index][neuron_index_zeroed]++;
		}
	}
	// 3. Calculate probabilities_of_bin_responses_p_r
	for (int bin_index = 0; bin_index < number_of_bins; bin_index++) {
		for (int neuron_index_zeroed = 0; neuron_index_zeroed < number_of_neurons_in_single_cell_analysis_group; neuron_index_zeroed++) {
			probabilities_of_bin_responses_p_r[bin_index][neuron_index_zeroed] = (float) individual_bin_counts_for_each_neuron[bin_index][neuron_index_zeroed] / (float) input_neurons->total_number_of_input_stimuli;
			// printf("probabilities_of_bin_responses_p_r[bin_index][neuron_index_zeroed]: %f\n", probabilities_of_bin_responses_p_r[bin_index][neuron_index_zeroed]);
		}
	}
	// 4. Calculate probabilities_of_bin_responses_given_an_object_p_r_given_s
	// First set up arrays
	float *** probabilities_of_bin_responses_given_an_object_p_r_given_s = new float**[input_neurons->total_number_of_objects];
	for (int object_index = 0; object_index < input_neurons->total_number_of_objects; object_index++) {
		probabilities_of_bin_responses_given_an_object_p_r_given_s[object_index] = new float*[number_of_neurons_in_single_cell_analysis_group];
		for (int neuron_index_zeroed = 0; neuron_index_zeroed < number_of_neurons_in_single_cell_analysis_group; neuron_index_zeroed++) {
			probabilities_of_bin_responses_given_an_object_p_r_given_s[object_index][neuron_index_zeroed] = new float[number_of_bins];
			for (int bin_index = 0; bin_index < number_of_bins; bin_index++) {
				probabilities_of_bin_responses_given_an_object_p_r_given_s[object_index][neuron_index_zeroed][bin_index] = 0.0;
			}
		}
	}

	for (int object_index = 0; object_index < input_neurons->total_number_of_objects; object_index++) {
		for (int neuron_index_zeroed = 0; neuron_index_zeroed < number_of_neurons_in_single_cell_analysis_group; neuron_index_zeroed++) {
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
	// printf("maximum_possible_information_score: %f\n", maximum_possible_information_score);
	for (int object_index = 0; object_index < input_neurons->total_number_of_objects; object_index++) {
		for (int neuron_index_zeroed = 0; neuron_index_zeroed < number_of_neurons_in_single_cell_analysis_group; neuron_index_zeroed++) {
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
	for (int neuron_index_zeroed = 0; neuron_index_zeroed < number_of_neurons_in_single_cell_analysis_group; neuron_index_zeroed++) {
		float maximum_score_for_current_neuron = -1.0;
		for (int object_index = 0; object_index < input_neurons->total_number_of_objects; object_index++) {
			if (information_scores_for_each_object_and_neuron[object_index][neuron_index_zeroed] > maximum_score_for_current_neuron) {
				maximum_score_for_current_neuron = information_scores_for_each_object_and_neuron[object_index][neuron_index_zeroed];
			}
		}
		maximum_information_score_for_each_neuron[neuron_index_zeroed] = maximum_score_for_current_neuron;
		descending_maximum_information_score_for_each_neuron[neuron_index_zeroed] = maximum_score_for_current_neuron;
		
		// printf("maximum_information_score_for_each_neuron[neuron_index_zeroed]: %f\n", maximum_information_score_for_each_neuron[neuron_index_zeroed]);
	}
	std::sort(descending_maximum_information_score_for_each_neuron, descending_maximum_information_score_for_each_neuron + number_of_neurons_in_single_cell_analysis_group, std::greater<float>());
	
	sum_of_maximum_information_score_for_each_neuron = 0;
	for (int neuron_index_zeroed = 0; neuron_index_zeroed < number_of_neurons_in_single_cell_analysis_group; neuron_index_zeroed++) {
		sum_of_maximum_information_score_for_each_neuron += descending_maximum_information_score_for_each_neuron[neuron_index_zeroed];
		// printf("descending_maximum_information_score_for_each_neuron[%d]: %f\n", neuron_index_zeroed, descending_maximum_information_score_for_each_neuron[neuron_index_zeroed]);
	}

	maximum_information_score_count_multiplied_by_sum_of_information_scores = (float)number_of_neurons_with_maximum_information_score_in_last_neuron_group * sum_of_information_scores_for_last_neuron_group;

	
	// printf("HERE\n");
	//6. Sort information scores for each object and neuron
	for (int object_index = 0; object_index < input_neurons->total_number_of_objects; object_index++) {
		std::sort(descending_information_scores_for_each_object_and_neuron[object_index], descending_information_scores_for_each_object_and_neuron[object_index] + number_of_neurons_in_single_cell_analysis_group, std::greater<float>());
		
		// for (int neuron_index_zeroed = 0; neuron_index_zeroed < number_of_neurons_in_single_cell_analysis_group; neuron_index_zeroed++) {
		// 	printf("%f\n", descending_maximum_information_score_for_each_neuron[neuron_index_zeroed]);
		// }

	}
	
	
	//calculate minimal number of cells that became to be tuned to a particular object
	number_of_neurons_with_maximum_information_score_in_average_in_last_neuron_group = 0;

	for (int neuron_index_zeroed = 0; neuron_index_zeroed < number_of_neurons_in_single_cell_analysis_group; neuron_index_zeroed++) {
		float sum_score_for_current_neuron = 0.0;
		for (int object_index = 0; object_index < input_neurons->total_number_of_objects; object_index++) {
			sum_score_for_current_neuron+=descending_information_scores_for_each_object_and_neuron[object_index][neuron_index_zeroed];
		}
		average_information_score_for_each_neuron[neuron_index_zeroed] = sum_score_for_current_neuron/input_neurons->total_number_of_objects;
		descending_average_information_score_for_each_neuron[neuron_index_zeroed] = sum_score_for_current_neuron/input_neurons->total_number_of_objects;
		
		if (maximum_possible_information_score <= average_information_score_for_each_neuron[neuron_index_zeroed])
			number_of_neurons_with_maximum_information_score_in_average_in_last_neuron_group++;
	}
	
	std::sort(descending_average_information_score_for_each_neuron, descending_average_information_score_for_each_neuron + number_of_neurons_in_single_cell_analysis_group, std::greater<float>());

	
	std::vector<float> maximum_information_score_for_each_neuron_tmp;
	std::vector<float> descending_maximum_information_score_for_each_neuron_tmp;
	std::vector<float> average_information_score_for_each_neuron_tmp;
	std::vector<float> descending_average_information_score_for_each_neuron_tmp;
		
	for (int neuron_index_zeroed = 0; neuron_index_zeroed < number_of_neurons_in_single_cell_analysis_group; neuron_index_zeroed++) {
		maximum_information_score_for_each_neuron_tmp.push_back(maximum_information_score_for_each_neuron[neuron_index_zeroed]);
		descending_maximum_information_score_for_each_neuron_tmp.push_back(descending_maximum_information_score_for_each_neuron[neuron_index_zeroed]);
		average_information_score_for_each_neuron_tmp.push_back(average_information_score_for_each_neuron[neuron_index_zeroed]);
		descending_average_information_score_for_each_neuron_tmp.push_back(descending_average_information_score_for_each_neuron[neuron_index_zeroed]);
	}
	maximum_information_score_for_each_neuron_vec.push_back(maximum_information_score_for_each_neuron_tmp);
	descending_maximum_information_score_for_each_neuron_vec.push_back(descending_maximum_information_score_for_each_neuron_tmp);
	average_information_score_for_each_neuron_vec.push_back(average_information_score_for_each_neuron_tmp);
	descending_average_information_score_for_each_neuron_vec.push_back(descending_average_information_score_for_each_neuron_tmp);
	
	
	
	single_cell_analysis_timer->stop_timer_and_log_time_and_message("Single Cell Information Scores Calulated Neuron Group.", false);
	printf("--- sum_of_information_scores_for_last_neuron_group: %f\n", sum_of_information_scores_for_last_neuron_group);
	printf("--- number_of_neurons_with_maximum_information_score_in_last_neuron_group: %d\n", number_of_neurons_with_maximum_information_score_in_last_neuron_group);
	printf("--- number_of_neurons_with_maximum_information_score_in_average_in_last_neuron_group: %d\n", number_of_neurons_with_maximum_information_score_in_average_in_last_neuron_group);
	printf("--- maximum_information_score_count_multiplied_by_sum_of_information_scores: %f\n", maximum_information_score_count_multiplied_by_sum_of_information_scores);
	printf("--- sum_of_maximum_information_score_for_each_neuron: %f\n", sum_of_maximum_information_score_for_each_neuron);
	print_line_of_dashes_with_blank_lines_either_side();


}

SPIKE_MAKE_INIT_BACKEND(SpikeAnalyser);
