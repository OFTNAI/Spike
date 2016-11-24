#ifndef GeneratorInputSpikingNeurons_H
#define GeneratorInputSpikingNeurons_H

#include "InputSpikingNeurons.hpp"

struct generator_input_spiking_neuron_parameters_struct : input_spiking_neuron_parameters_struct {
	generator_input_spiking_neuron_parameters_struct() { input_spiking_neuron_parameters_struct(); }
};

class GeneratorInputSpikingNeurons; // forward definition

namespace Backend {
  class GeneratorInputSpikingNeurons : public InputSpikingNeurons {
  public:
    ADD_FRONTEND_GETTER(GeneratorInputSpikingNeurons);
    virtual void reset_state() {};
  };
} 

#include "Spike/Backend/Dummy/Neurons/GeneratorInputSpikingNeurons.hpp"

class GeneratorInputSpikingNeurons : public InputSpikingNeurons {
public:
  // Constructor/Destructor
  GeneratorInputSpikingNeurons();
  ~GeneratorInputSpikingNeurons();

  ADD_BACKEND_GETTER(GeneratorInputSpikingNeurons);
  
  // Variables
  int length_of_longest_stimulus;

  // Host Pointers
  int* number_of_spikes_in_stimuli = nullptr;
  int** neuron_id_matrix_for_stimuli = nullptr;
  float** spike_times_matrix_for_stimuli = nullptr;

  // Don't need this as it is inherited without change:
  // virtual int AddGroup(neuron_parameters_struct * group_params);

  virtual void reset_state();

  virtual void update_membrane_potentials(float timestep, float current_time_in_seconds);

  void AddStimulus(int spikenumber, int* ids, float* spiketimes);
};

#endif
