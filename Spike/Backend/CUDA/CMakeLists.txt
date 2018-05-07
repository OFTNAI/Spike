if(BUILD_SHARED_LIBS)
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" "-Xcompiler -fPIC -arch=sm_52")
  #set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" "-dc")
  #set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" "-rdc=true")
endif()

add_subdirectory(Helpers)
add_subdirectory(Neurons)
add_subdirectory(RecordingElectrodes)
add_subdirectory(SpikeAnalyser)
add_subdirectory(Plasticity)
add_subdirectory(Synapses)


set (CUDAFiles
  ${CUDAHelpers}
  ${CUDASynapses}
  ${CUDANeurons}
  ${CUDARecordingElectrodes}
  ${CUDASpikeAnalyser}
  ${CUDAPlasticity})

function(cat IN_FILE OUT_FILE)
  file(READ ${IN_FILE} CONTENTS)
  file(APPEND ${OUT_FILE} "${CONTENTS}")
endfunction()

# Prepare a temporary file to "cat" to:
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/allcuda.cu.in "")

# Call the "cat" function for each input file
foreach(CUDAFILE ${CUDAFiles})
  cat(${CUDAFILE} ${CMAKE_CURRENT_BINARY_DIR}/allcuda.cu.in)
endforeach()

# Copy the temporary file to the final location
configure_file(${CMAKE_CURRENT_BINARY_DIR}/allcuda.cu.in allcuda.cu COPYONLY)

CUDA_ADD_LIBRARY (SpikeCUDA
  ${CMAKE_CURRENT_BINARY_DIR}/allcuda.cu
  STATIC)


