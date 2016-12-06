#pragma once

#include "Spike/RecordingElectrodes/CountNeuronSpikesRecordingElectrodes.hpp"
#include "RecordingElectrodes.hpp"
#include "Spike/Backend/CUDA/CUDABackend.hpp"
#include <cuda.h>
#include <vector_types.h>
#include <curand.h>
#include <curand_kernel.h>

namespace Backend {
  namespace CUDA {
    class CountNeuronSpikesRecordingElectrodes :
      protected virtual ::Backend::CUDA::RecordingElectrodes,
      public virtual ::Backend::CountNeuronSpikesRecordingElectrodes {
    public:
      ~CountNeuronSpikesRecordingElectrodes();
      MAKE_BACKEND_CONSTRUCTOR(CountNeuronSpikesRecordingElectrodes);
      using ::Backend::CountNeuronSpikesRecordingElectrodes::frontend;
      
      virtual void reset_state();
      virtual void prepare();
      void allocate_pointers_for_spike_count();

      virtual void add_spikes_to_per_neuron_spike_count
      (float current_time_in_seconds);

      int * per_neuron_spike_counts = nullptr;
    };

    __global__ void add_spikes_to_per_neuron_spike_count_kernel
    (float* d_last_spike_time_of_each_neuron,
     int* d_per_neuron_spike_counts,
     float current_time_in_seconds,
     size_t total_number_of_neurons);
  }
}
