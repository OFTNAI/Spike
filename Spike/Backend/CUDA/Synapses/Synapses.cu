// -*- mode: c++ -*-
#include "Spike/Backend/CUDA/Synapses/Synapses.hpp"

// SPIKE_EXPORT_BACKEND_TYPE(CUDA, Synapses);

namespace Backend {
  namespace CUDA {
    Synapses::~Synapses() {
      CudaSafeCall(cudaFree(presynaptic_neuron_indices));
      CudaSafeCall(cudaFree(postsynaptic_neuron_indices));
      CudaSafeCall(cudaFree(temp_presynaptic_neuron_indices));
      CudaSafeCall(cudaFree(temp_postsynaptic_neuron_indices));
      CudaSafeCall(cudaFree(synaptic_efficacies_or_weights));
      CudaSafeCall(cudaFree(temp_synaptic_efficacies_or_weights));
      CudaSafeCall(cudaFree(synapse_postsynaptic_neuron_count_index));
#ifdef CRAZY_DEBUG
      std::cout << "\n!!!!!!!!!!!!!!!!!!!!---AAAAAA---!!!!!!!!!!!!!!!!!!!\n";
#endif
    }

    void Synapses::reset_state() {
    }

    void Synapses::allocate_device_pointers() {
#ifdef CRAZY_DEBUG
      // if (frontend()->total_number_of_synapses == 0)
      //   return;
      std::cout << "DEBUG:::: " << frontend()->total_number_of_synapses << "\n"
                << "     :::: " << &presynaptic_neuron_indices << "\n"
        ;
#endif
      CudaSafeCall(cudaMalloc((void **)&presynaptic_neuron_indices,
                              sizeof(int)*frontend()->total_number_of_synapses));
      CudaSafeCall(cudaMalloc((void **)&postsynaptic_neuron_indices,
                              sizeof(int)*frontend()->total_number_of_synapses));
      CudaSafeCall(cudaMalloc((void **)&synaptic_efficacies_or_weights,
                              sizeof(float)*frontend()->total_number_of_synapses));
      CudaSafeCall(cudaMalloc((void **)&synapse_postsynaptic_neuron_count_index,
                              sizeof(float)*frontend()->total_number_of_synapses));
    }


    void Synapses::copy_constants_and_initial_efficacies_to_device() {
      CudaSafeCall(cudaMemcpy(presynaptic_neuron_indices,
                              frontend()->presynaptic_neuron_indices,
                              sizeof(int)*frontend()->total_number_of_synapses,
                              cudaMemcpyHostToDevice));
      CudaSafeCall(cudaMemcpy(postsynaptic_neuron_indices,
                              frontend()->postsynaptic_neuron_indices,
                              sizeof(int)*frontend()->total_number_of_synapses,
                              cudaMemcpyHostToDevice));
      CudaSafeCall(cudaMemcpy(synaptic_efficacies_or_weights,
                              frontend()->synaptic_efficacies_or_weights,
                              sizeof(float)*frontend()->total_number_of_synapses,
                              cudaMemcpyHostToDevice));
      CudaSafeCall(cudaMemcpy(synapse_postsynaptic_neuron_count_index,
                              frontend()->synapse_postsynaptic_neuron_count_index,
                              sizeof(float)*frontend()->total_number_of_synapses,
                              cudaMemcpyHostToDevice));
    }


    void Synapses::set_threads_per_block_and_blocks_per_grid(int threads) {
      threads_per_block.x = threads;
      cudaDeviceProp deviceProp;
      int deviceID;
      cudaGetDevice(&deviceID);
      cudaGetDeviceProperties(&deviceProp, deviceID);
      int max_num_blocks_per_grid = deviceProp.multiProcessorCount*(deviceProp.maxThreadsPerMultiProcessor / threads);
      int theoretical_number = (frontend()->total_number_of_synapses + threads) / threads;
      if (theoretical_number < max_num_blocks_per_grid)
	number_of_synapse_blocks_per_grid = dim3(theoretical_number);
      else
	number_of_synapse_blocks_per_grid = dim3(max_num_blocks_per_grid);
      //printf("%d, %d\n", max_num_blocks_per_grid, theoretical_number);
    }

    void Synapses::prepare() {
      allocate_device_pointers();
      copy_constants_and_initial_efficacies_to_device();
      set_threads_per_block_and_blocks_per_grid(context->params.threads_per_block_synapses);

      random_state_manager_backend
        = dynamic_cast<::Backend::CUDA::RandomStateManager*>
        (frontend()->random_state_manager->backend());
    }

    void Synapses::set_neuron_indices_by_sampling_from_normal_distribution
    (int original_number_of_synapses,
     int total_number_of_new_synapses,
     int postsynaptic_group_id,
     int poststart, int prestart,
     int* postsynaptic_group_shape,
     int* presynaptic_group_shape,
     int number_of_new_synapses_per_postsynaptic_neuron,
     int number_of_postsynaptic_neurons_in_group,
     int max_number_of_connections_per_pair,
     float standard_deviation_sigma,
     bool presynaptic_group_is_input) {

      if (total_number_of_new_synapses > frontend()->largest_synapse_group_size || !temp_presynaptic_neuron_indices) {
        CudaSafeCall(cudaMalloc((void **)&temp_presynaptic_neuron_indices,
                                sizeof(int)*total_number_of_new_synapses));
        CudaSafeCall(cudaMalloc((void **)&temp_postsynaptic_neuron_indices,
                                sizeof(int)*total_number_of_new_synapses));
      }

      set_neuron_indices_by_sampling_from_normal_distribution_kernel<<<random_state_manager_backend->block_dimensions, random_state_manager_backend->threads_per_block>>>
        (total_number_of_new_synapses,
         postsynaptic_group_id,
         poststart, prestart,
         postsynaptic_group_shape[0],
         postsynaptic_group_shape[1],
         presynaptic_group_shape[0],
         presynaptic_group_shape[1],
         number_of_new_synapses_per_postsynaptic_neuron,
         number_of_postsynaptic_neurons_in_group,
         max_number_of_connections_per_pair,
         temp_presynaptic_neuron_indices,
         temp_postsynaptic_neuron_indices,
         temp_synaptic_efficacies_or_weights,
         standard_deviation_sigma,
         presynaptic_group_is_input,
         random_state_manager_backend->states);
      CudaCheckError();

      CudaSafeCall(cudaMemcpy(&(frontend()->presynaptic_neuron_indices)[original_number_of_synapses], temp_presynaptic_neuron_indices, sizeof(int)*total_number_of_new_synapses, cudaMemcpyDeviceToHost));
      CudaSafeCall(cudaMemcpy(&(frontend()->postsynaptic_neuron_indices)[original_number_of_synapses], temp_postsynaptic_neuron_indices, sizeof(int)*total_number_of_new_synapses, cudaMemcpyDeviceToHost));
    }
    
    __global__ void set_neuron_indices_by_sampling_from_normal_distribution_kernel
    (int total_number_of_new_synapses,
     int postsynaptic_group_id,
     int poststart, int prestart,
     int post_width, int post_height,
     int pre_width, int pre_height,
     int number_of_new_synapses_per_postsynaptic_neuron,
     int number_of_postsynaptic_neurons_in_group,
     int max_number_of_connections_per_pair,
     int * d_presynaptic_neuron_indices,
     int * d_postsynaptic_neuron_indices,
     float * d_synaptic_efficacies_or_weights,
     float standard_deviation_sigma,
     bool presynaptic_group_is_input,
     curandState_t* d_states) {

      int idx = threadIdx.x + blockIdx.x * blockDim.x;
      int t_idx = idx;
      while (idx < floor((float)total_number_of_new_synapses/max_number_of_connections_per_pair)) {

        int postsynaptic_neuron_id = (idx*max_number_of_connections_per_pair) / number_of_new_synapses_per_postsynaptic_neuron;

        int postsynaptic_x = postsynaptic_neuron_id % post_width; 
        int postsynaptic_y = floor((float)(postsynaptic_neuron_id) / post_width);
        float fractional_x = (float)postsynaptic_x / post_width;
        float fractional_y = (float)postsynaptic_y / post_height;

        int corresponding_presynaptic_centre_x = floor((float)pre_width * fractional_x); 
        int corresponding_presynaptic_centre_y = floor((float)pre_height * fractional_y);

        bool presynaptic_x_set = false;
        bool presynaptic_y_set = false;
        int presynaptic_x = -1;
        int presynaptic_y = -1; 

        while (true) {

          if (presynaptic_x_set == false) {
            float value_from_normal_distribution_for_x = curand_normal(&d_states[t_idx]);
            float scaled_value_from_normal_distribution_for_x = standard_deviation_sigma * value_from_normal_distribution_for_x;
            int rounded_scaled_value_from_normal_distribution_for_x = round(scaled_value_from_normal_distribution_for_x);
            presynaptic_x = corresponding_presynaptic_centre_x + rounded_scaled_value_from_normal_distribution_for_x;
            if ((presynaptic_x > -1) && (presynaptic_x < pre_width)) {
              presynaptic_x_set = true;
            }

          }

          if (presynaptic_y_set == false) {
			
            float value_from_normal_distribution_for_y = curand_normal(&d_states[t_idx]);
            float scaled_value_from_normal_distribution_for_y = standard_deviation_sigma * value_from_normal_distribution_for_y;
            int rounded_scaled_value_from_normal_distribution_for_y = round(scaled_value_from_normal_distribution_for_y);
            presynaptic_y = corresponding_presynaptic_centre_y + rounded_scaled_value_from_normal_distribution_for_y;
            if ((presynaptic_y > -1) && (presynaptic_y < pre_height)) {
              presynaptic_y_set = true;
            }

          }

          if (presynaptic_x_set && presynaptic_y_set) {
            //d_presynaptic_neuron_indices[idx] = CORRECTED_PRESYNAPTIC_ID(prestart + presynaptic_x + presynaptic_y*pre_width, presynaptic_group_is_input);
            for(int idx_multipleContact=0; idx_multipleContact < max_number_of_connections_per_pair; idx_multipleContact++) {
              d_presynaptic_neuron_indices[idx*max_number_of_connections_per_pair+idx_multipleContact] = CORRECTED_PRESYNAPTIC_ID(prestart + presynaptic_x + presynaptic_y*pre_width, presynaptic_group_is_input);
              d_postsynaptic_neuron_indices[idx*max_number_of_connections_per_pair+idx_multipleContact] = poststart + postsynaptic_neuron_id;
            }
            break;
          }
			

        }	
        idx += blockDim.x * gridDim.x;

      }	

      __syncthreads();

    }

  }
}

