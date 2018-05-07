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

CUDA_ADD_LIBRARY (SpikeCUDA
  ${CUDAHelpers}
  ${CUDANeurons}
  ${CUDARecordingElectrodes}
  ${CUDASpikeAnalyser}
  ${CUDAPlasticity}
  ${CUDASynapses}
  STATIC)

#[[
set (CUDAFiles
  ${CUDAHelpers}
  ${CUDASynapes}
  ${CUDANeurons}
  ${CUDARecordingElectrodes}
  ${CUDASpikeAnalyser}
  ${CUDAPlasticity})

function(cat IN_FILE OUT_FILE)
  file(READ ${IN_FILE} CONTENTS)
  file(APPEND ${OUT_FILE} "${CONTENTS}")
endfunction()

# Prepare a temporary file to "cat" to:
file(WRITE allcuda.cu.in "")

# Call the "cat" function for each input file
foreach(CUDAFILE ${CUDAFiles})
  cat(${CUDAFILE} allcuda.cu.in)
endforeach()

# Copy the temporary file to the final location
configure_file(allcuda.cu.in allcuda.cu COPYONLY)

#CUDA_ADD_LIBRARY (SpikeCUDA
#  ${CMAKE_CURRENT_BINARY_DIR}/allcuda.cu
#  STATIC)

]]

