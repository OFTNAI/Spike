#include "Spike/Backend/CUDA/Neurons/IzhikevichSpikingNeurons.hpp"

namespace Backend {
  namespace CUDA {

    IzhikevichSpikingNeurons::~IzhikevichSpikingNeurons() {
      CudaSafeCall(cudaFree(d_param_a));
      CudaSafeCall(cudaFree(d_param_b));
      CudaSafeCall(cudaFree(d_param_d));
      CudaSafeCall(cudaFree(d_states_u));
    }
    
    void IzhikevichSpikingNeurons::allocate_device_pointers(int maximum_axonal_delay_in_timesteps, bool high_fidelity_spike_storage) {
 	
      SpikingNeurons::allocate_device_pointers(maximum_axonal_delay_in_timesteps, high_fidelity_spike_storage);

      CudaSafeCall(cudaMalloc((void **)&d_param_a, sizeof(float)*total_number_of_neurons));
      CudaSafeCall(cudaMalloc((void **)&d_param_b, sizeof(float)*total_number_of_neurons));
      CudaSafeCall(cudaMalloc((void **)&d_param_d, sizeof(float)*total_number_of_neurons));
      CudaSafeCall(cudaMalloc((void **)&d_states_u, sizeof(float)*total_number_of_neurons));
    }

    void IzhikevichSpikingNeurons::copy_constants_to_device() {
      SpikingNeurons::copy_constants_to_device();

      CudaSafeCall(cudaMemcpy(d_param_a, param_a, sizeof(float)*total_number_of_neurons, cudaMemcpyHostToDevice));
      CudaSafeCall(cudaMemcpy(d_param_b, param_b, sizeof(float)*total_number_of_neurons, cudaMemcpyHostToDevice));
      CudaSafeCall(cudaMemcpy(d_param_d, param_d, sizeof(float)*total_number_of_neurons, cudaMemcpyHostToDevice));
    }

    void IzhikevichSpikingNeurons::check_for_neuron_spikes(float current_time_in_seconds, float timestep) {
	SpikingNeurons::check_for_neuron_spikes(current_time_in_seconds, timestep);

	reset_states_u_after_spikes_kernel<<<number_of_neuron_blocks_per_grid, threads_per_block>>>
          (d_states_u,
           d_param_d,
           d_last_spike_time_of_each_neuron,
           current_time_in_seconds,
           total_number_of_neurons);
	CudaCheckError();
    }

    void IzhikevichSpikingNeurons::update_membrane_potentials(float timestep, float current_time_in_seconds) {

      izhikevich_update_membrane_potentials_kernel<<<number_of_neuron_blocks_per_grid, threads_per_block>>>
        (d_membrane_potentials_v,
         d_states_u,
         d_param_a,
         d_param_b,
         d_current_injections,
         timestep,
         total_number_of_neurons);

      CudaCheckError();
    }

    void IzhikevichSpikingNeurons::reset() {
      CudaSafeCall(cudaMemset(d_states_u, 0.0f, sizeof(float)*total_number_of_neurons));
    }

    __global__ void reset_states_u_after_spikes_kernel(float *d_states_u,
                                                       float * d_param_d,
                                                       float* d_last_spike_time_of_each_neuron,
                                                       float current_time_in_seconds,
                                                       size_t total_number_of_neurons) {
	
      int idx = threadIdx.x + blockIdx.x * blockDim.x;
      while (idx < total_number_of_neurons) {
        if (d_last_spike_time_of_each_neuron[idx] == current_time_in_seconds) {

          d_states_u[idx] += d_param_d[idx];

        }
        idx += blockDim.x * gridDim.x;
      }
      __syncthreads();
    }


    __global__ void izhikevich_update_membrane_potentials_kernel(float *d_membrane_potentials_v,
                                                                 float *d_states_u,
                                                                 float *d_param_a,
                                                                 float *d_param_b,
                                                                 float* d_current_injections,
                                                                 float timestep,
                                                                 size_t total_number_of_neurons) {

      // We require the equation timestep in ms:
      float eqtimestep = timestep*1000.0f;
      // Get thread IDs
      int idx = threadIdx.x + blockIdx.x * blockDim.x;
      while (idx < total_number_of_neurons) {
        // Update the neuron states according to the Izhikevich equations
        float v_update = 0.04f*d_membrane_potentials_v[idx]*d_membrane_potentials_v[idx] 
          + 5.0f*d_membrane_potentials_v[idx]
          + 140 
          - d_states_u[idx]
          + d_current_injections[idx];

        d_membrane_potentials_v[idx] += eqtimestep*v_update;
        d_states_u[idx] += eqtimestep*(d_param_a[idx] * (d_param_b[idx] * d_membrane_potentials_v[idx] - 
                                                         d_states_u[idx]));

        idx += blockDim.x * gridDim.x;
      }
      __syncthreads();
    }
  }
}
