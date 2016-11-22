#pragma once

#include "Spike/RecordingElectrodes/CollectNeuronSpikesRecordingElectrodes.hpp"
#include "RecordingElectrodes.hpp"
#include "Spike/Backend/CUDA/CUDABackend.hpp"
#include <cuda.h>
#include <vector_types.h>
#include <curand.h>
#include <curand_kernel.h>

namespace Backend {
  namespace CUDA {
    class CollectNeuronSpikesRecordingElectrodes :
      public virtual ::Backend::CUDA::RecordingElectrodesCommon,
      public ::Backend::CollectNeuronSpikesRecordingElectrodes {
    public:
      ~CollectNeuronSpikesRecordingElectrodes();

      virtual void reset_state();

      int* neuron_ids_of_stored_spikes_on_device = NULL;
      int* total_number_of_spikes_stored_on_device = NULL;
      float* time_in_seconds_of_stored_spikes_on_device = NULL;
    };

    __global__ void collect_spikes_for_timestep_kernel
    (float* d_last_spike_time_of_each_neuron,
     int* d_total_number_of_spikes_stored_on_device,
     int* d_neuron_ids_of_stored_spikes_on_device,
     float* d_time_in_seconds_of_stored_spikes_on_device,
     float current_time_in_seconds,
     size_t total_number_of_neurons);
  }
}
