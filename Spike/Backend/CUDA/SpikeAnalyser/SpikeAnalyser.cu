// -*- mode: c++ -*-
#include "Spike/Backend/CUDA/SpikeAnalyser/SpikeAnalyser.hpp"

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

    void SpikeAnalyser::push_data_front() {
      // TODO: See comment in store_spike_counts_for_stimulus_index below.
    }

    void SpikeAnalyser::pull_data_back() {
    }

    void SpikeAnalyser::store_spike_counts_for_stimulus_index(int stimulus_index) {
      // TODO: Should this be a 'push_data_front' kind of function?
      //       Perhaps even in CountNeuronSpikesRecordingElectrodes?
      //       Then the front-end could just get the data from the
      //          electrodes instance, and wouldn't need this backend at all!..
      CudaSafeCall
        (cudaMemcpy(frontend()->per_stimulus_per_neuron_spike_counts[stimulus_index], 
                    count_electrodes_backend->per_neuron_spike_counts, 
                    sizeof(float) * frontend()->neurons->total_number_of_neurons, 
                    cudaMemcpyDeviceToHost));
    }
  }
}
