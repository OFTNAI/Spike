// -*- mode: c++ -*-
#include "Spike/Backend/CUDA/SpikeAnalyser/SpikeAnalyser.hpp"

SPIKE_EXPORT_BACKEND_TYPE(CUDA, SpikeAnalyser);

namespace Backend {
  namespace CUDA {
    void SpikeAnalyser::prepare() {
      neurons_backend = dynamic_cast<::Backend::CUDA::SpikingNeurons*>
        (frontend()->neurons->backend());
      input_neurons_backend = dynamic_cast<::Backend::CUDA::InputSpikingNeurons*>
        (frontend()->input_neurons->backend());
      count_electrodes_backend =
        dynamic_cast<::Backend::CUDA::CountNeuronSpikesRecordingElectrodes*>
        (frontend()->count_electrodes->backend());
    }

    void SpikeAnalyser::reset_state() {
    }

    void SpikeAnalyser::store_spike_counts_for_stimulus_index(int stimulus_index) {
      CudaSafeCall
        (cudaMemcpy(frontend()->per_stimulus_per_neuron_spike_counts[stimulus_index], 
                    count_electrodes_backend->per_neuron_spike_counts, 
                    sizeof(float) * frontend()->neurons->total_number_of_neurons, 
                    cudaMemcpyDeviceToHost));
    }
  }
}
