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
    ADD_FRONTEND_GETTER(SpikeAnalyser);

    virtual void store_spike_counts_for_stimulus_index
    // (int stimulus_index, int *d_neuron_spike_counts_for_stimulus) = 0;
    (int stimulus_index) = 0;

    void reset_state() override {
      // TODO?
    }
  };
}

#include "Spike/Backend/Dummy/SpikeAnalyser/SpikeAnalyser.hpp"
#ifdef SPIKE_WITH_CUDA
#include "Spike/Backend/CUDA/SpikeAnalyser/SpikeAnalyser.hpp"
#endif

class SpikeAnalyser : public virtual SpikeBase {
public:
  ADD_BACKEND_GETTER(SpikeAnalyser);
  void prepare_backend(Context* ctx = _global_ctx);
  virtual void reset_state() {}

  SpikeAnalyser(SpikingNeurons *neurons_parameter,
                InputSpikingNeurons *input_neurons_parameter,
                CountNeuronSpikesRecordingElectrodes *electrodes_parameter);

  SpikingNeurons *neurons;
  InputSpikingNeurons *input_neurons;
  CountNeuronSpikesRecordingElectrodes *count_electrodes;

  std::vector<int> number_of_neurons_in_single_cell_analysis_group_vec;
  std::vector<std::vector<float> > descending_maximum_information_score_for_each_neuron_vec;
  std::vector<std::vector<float> > maximum_information_score_for_each_neuron_vec;
  std::vector<std::vector<float> > descending_average_information_score_for_each_neuron_vec;
  std::vector<std::vector<float> > average_information_score_for_each_neuron_vec;

  int ** per_stimulus_per_neuron_spike_counts;

  float maximum_possible_information_score;
  float optimal_average_firing_rate;
  float optimal_max_firing_rate;

  float ** information_scores_for_each_object_and_neuron;
  float ** descending_information_scores_for_each_object_and_neuron;
  float * maximum_information_score_for_each_neuron;
  float * descending_maximum_information_score_for_each_neuron;
  float * average_information_score_for_each_neuron;
  float * descending_average_information_score_for_each_neuron;

  int ** number_of_spikes_per_stimulus_per_neuron_group;
  float ** average_number_of_spikes_per_stimulus_per_neuron_group_per_second;
  int * total_number_of_spikes_per_neuron_group;
  float * average_number_of_spikes_per_neuron_group_per_second;
  float * max_number_of_spikes_per_neuron_group_per_second;
  int total_number_of_neuron_spikes;
  float average_number_of_neuron_spikes_per_second;

  float combined_powered_distance_from_average_score;
  float * combined_powered_distance_from_average_score_for_each_neuron_group;

  float combined_powered_distance_from_max_score;
  float * combined_powered_distance_from_max_score_for_each_neuron_group;

  bool spike_totals_and_averages_were_calculated;

  int number_of_neurons_with_maximum_information_score_in_last_neuron_group;
  int number_of_neurons_with_maximum_information_score_in_average_in_last_neuron_group;
  float sum_of_information_scores_for_last_neuron_group;
  float maximum_information_score_count_multiplied_by_sum_of_information_scores;

  void store_spike_counts_for_stimulus_index(int stimulus_index);
  void calculate_various_neuron_spike_totals_and_averages(float presentation_time_per_stimulus_per_epoch);
  void calculate_fitness_score(float optimal_average_firing_rate, float optimal_max_firing_rate);
  void calculate_single_cell_information_scores_for_neuron_group(int neuron_group_index, int number_of_bins, bool useThresholdForMaxFR,float optimal_max_firing_rate);

};

#endif
