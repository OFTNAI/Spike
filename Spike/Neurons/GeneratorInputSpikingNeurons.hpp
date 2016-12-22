#ifndef GeneratorInputSpikingNeurons_H
#define GeneratorInputSpikingNeurons_H

#include "InputSpikingNeurons.hpp"

struct generator_input_spiking_neuron_parameters_struct : input_spiking_neuron_parameters_struct {
	generator_input_spiking_neuron_parameters_struct() { input_spiking_neuron_parameters_struct(); }
};

class GeneratorInputSpikingNeurons; // forward definition

namespace Backend {
  class GeneratorInputSpikingNeurons : public virtual InputSpikingNeurons {
  public:
    SPIKE_ADD_FRONTEND_GETTER(GeneratorInputSpikingNeurons);
  };
} 

#include "Spike/Backend/Dummy/Neurons/GeneratorInputSpikingNeurons.hpp"
#ifdef SPIKE_WITH_CUDA
#include "Spike/Backend/CUDA/Neurons/GeneratorInputSpikingNeurons.hpp"
#endif

class GeneratorInputSpikingNeurons : public InputSpikingNeurons {
public:
  ~GeneratorInputSpikingNeurons() override;

  void init_backend(Context* ctx) override;
  SPIKE_ADD_BACKEND_GETSET(GeneratorInputSpikingNeurons, InputSpikingNeurons);
  
  // Variables
  int length_of_longest_stimulus;

  // Host Pointers
  int* number_of_spikes_in_stimuli = nullptr;
  int** neuron_id_matrix_for_stimuli = nullptr;
  float** spike_times_matrix_for_stimuli = nullptr;

  void update_membrane_potentials(float timestep, float current_time_in_seconds) override;

  void AddStimulus(int spikenumber, int* ids, float* spiketimes);

private:
  std::shared_ptr<::Backend::GeneratorInputSpikingNeurons> _backend;
};

#endif
