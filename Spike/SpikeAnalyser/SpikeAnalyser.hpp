#ifndef SpikeAnalyser_H
#define SpikeAnalyser_H

#include "Spike/Base.hpp"

#include "Spike/Backend/Macros.hpp"
#include "Spike/Backend/Context.hpp"
#include "Spike/Backend/Backend.hpp"
#include "Spike/Backend/Device.hpp"

#include "Spike/Neurons/SpikingNeurons.hpp"
#include "Spike/Neurons/InputSpikingNeurons.hpp"

#include "Spike/RecordingElectrodes/CountNeuronSpikesRecordingElectrodes.hpp"

#include <vector>

class SpikeAnalyser; // forward definition

namespace Backend {
  class SpikeAnalyser : public virtual SpikeBackendBase {
  public:
    SPIKE_ADD_BACKEND_FACTORY(SpikeAnalyser);
    ~SpikeAnalyser() override = default;

    virtual void store_spike_counts_for_stimulus_index
    (int stimulus_index) = 0;
  };
}

class SpikeAnalyser : public virtual SpikeBase {
public:
  ~SpikeAnalyser() override;
  SPIKE_ADD_BACKEND_GETSET(SpikeAnalyser, SpikeBase);
  void init_backend(Context* ctx = _global_ctx) override;
  void reset_state() override;

  SpikeAnalyser(SpikingNeurons *neurons_parameter,
                InputSpikingNeurons *input_neurons_parameter,
                CountNeuronSpikesRecordingElectrodes *electrodes_parameter);

  SpikingNeurons *neurons  = nullptr;
  InputSpikingNeurons *input_neurons = nullptr;
  CountNeuronSpikesRecordingElectrodes *count_electrodes = nullptr;

  std::vector<int> number_of_neurons_in_single_cell_analysis_group_vec;
  std::vector<std::vector<float> > descending_maximum_information_score_for_each_neuron_vec;
  std::vector<std::vector<float> > maximum_information_score_for_each_neuron_vec;
  std::vector<std::vector<float> > descending_average_information_score_for_each_neuron_vec;
  std::vector<std::vector<float> > average_information_score_for_each_neuron_vec;

  int ** per_stimulus_per_neuron_spike_counts = nullptr;

  float maximum_possible_information_score;
  float optimal_average_firing_rate;
  float optimal_max_firing_rate;

  float ** information_scores_for_each_object_and_neuron = nullptr;
  float ** descending_information_scores_for_each_object_and_neuron = nullptr;
  float * maximum_information_score_for_each_neuron = nullptr;
  float * descending_maximum_information_score_for_each_neuron = nullptr;
  float * average_information_score_for_each_neuron = nullptr;
  float * descending_average_information_score_for_each_neuron = nullptr;

  int ** number_of_spikes_per_stimulus_per_neuron_group = nullptr;
  float ** average_number_of_spikes_per_stimulus_per_neuron_group_per_second = nullptr;
  float ** average_number_of_spikes_per_stimulus_per_neuron_group_per_second_excluding_silent_neurons = nullptr;
  int * total_number_of_spikes_per_neuron_group = nullptr;
  float * average_number_of_spikes_per_neuron_group_per_second = nullptr;
  float * average_number_of_spikes_per_neuron_group_per_second_excluding_silent_neurons = nullptr;
  float * max_number_of_spikes_per_neuron_group_per_second = nullptr;
  int total_number_of_neuron_spikes;
  float average_number_of_neuron_spikes_per_second;

  int * running_count_of_non_silent_neurons_per_neuron_group = nullptr;

  float combined_powered_distance_from_average_score;
  float * combined_powered_distance_from_average_score_for_each_neuron_group = nullptr;

  float combined_powered_distance_from_max_score;
  float * combined_powered_distance_from_max_score_for_each_neuron_group = nullptr;

  bool spike_totals_and_averages_were_calculated;

  int number_of_neurons_with_maximum_information_score_in_last_neuron_group;
  int number_of_neurons_with_maximum_information_score_in_average_in_last_neuron_group;
  float sum_of_information_scores_for_last_neuron_group;
  float maximum_information_score_count_multiplied_by_sum_of_information_scores;

  void store_spike_counts_for_stimulus_index(int stimulus_index);
  void calculate_various_neuron_spike_totals_and_averages(float presentation_time_per_stimulus_per_epoch);
  void calculate_fitness_score(float optimal_average_firing_rate, float optimal_max_firing_rate);
  void calculate_single_cell_information_scores_for_neuron_group(int neuron_group_index, int number_of_bins, bool useThresholdForMaxFR,float optimal_max_firing_rate);

private:
  std::shared_ptr<::Backend::SpikeAnalyser> _backend;
};

#endif
