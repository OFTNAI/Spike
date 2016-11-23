#include "Spike/Backend/CUDA/RecordingElectrodes/CollectNeuronSpikesRecordingElectrodes.hpp"

namespace Backend {
  namespace CUDA {
    NetworkStateArchiveRecordingElectrodes::copy_state_to_front(::NetworkStateArchiveRecordingElectrodes* front) {
      CudaSafeCall(cudaMemcpy(synapses->synaptic_efficacies_or_weights, synapses->d_synaptic_efficacies_or_weights, sizeof(float)*synapses->total_number_of_synapses, cudaMemcpyDeviceToHost));
    }
  }
}
