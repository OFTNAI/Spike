// -*- mode: c++ -*-
#include "Spike/Backend/CUDA/Plasticity/STDPPlasticity.hpp"
#include <iostream>

// SPIKE_EXPORT_BACKEND_TYPE(CUDA, STDPPlasticity);

namespace Backend {
  namespace CUDA {
    STDPPlasticity::~STDPPlasticity() {
      CudaSafeCall(cudaFree(plastic_synapse_indices));
    }

    void STDPPlasticity::prepare() {

      neurons_backend = dynamic_cast<::Backend::CUDA::SpikingNeurons*>
        (frontend()->neurs->backend());
      synapses_backend = dynamic_cast<::Backend::CUDA::SpikingSynapses*>
        (frontend()->syns->backend());

      // Get the correct ID
      int plasticity_id = frontend()->plasticity_rule_id;
      if (plasticity_id >= 0){
        total_number_of_plastic_synapses = frontend()->plastic_synapses.size();
      } else {
        total_number_of_plastic_synapses = 0;
      }

      allocate_device_pointers();
    }

    void STDPPlasticity::allocate_device_pointers(){
      if (total_number_of_plastic_synapses > 0){
        CudaSafeCall(cudaMalloc((void **)&plastic_synapse_indices, sizeof(int)*total_number_of_plastic_synapses));
        CudaSafeCall(cudaMemcpy((void*)plastic_synapse_indices,
                                (void*)&(frontend()->plastic_synapses[0]),
                                sizeof(int)*total_number_of_plastic_synapses,
                                cudaMemcpyHostToDevice));
      }
    }

    void STDPPlasticity::reset_state() {
    }
  }
}
