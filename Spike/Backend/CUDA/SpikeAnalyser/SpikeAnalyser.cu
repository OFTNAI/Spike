#include "Spike/Backend/CUDA/SpikeAnalyser/SpikeAnalyser.hpp"
#include "Spike/Backend/CUDA/Neurons/SpikingNeurons.hpp"

namespace Backend {
  namespace CUDA {
    SpikeAnalyser::store_spike_counts_for_stimulus_index(::SpikeAnalyser* front,
                                                         int stimulus_index) {
      ::Backend::CUDA::SpikingNeurons* neurons_backend
        = (::Backend::CUDA::SpikingNeurons*)front->neurons->backend();
      	CudaSafeCall
          (cudaMemcpy(front->per_stimulus_per_neuron_spike_counts[stimulus_index], 
                      neurons_backend->neuron_spike_counts_for_stimulus, 
                      sizeof(float) * front->neurons->total_number_of_neurons, 
                      cudaMemcpyDeviceToHost));
    }
  }
}
