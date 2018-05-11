file(GLOB_RECURSE SPIKE_MAIN_FILES
  ${PROJECT_SOURCE_DIR}/Spike/Backend/Context.cpp
  ${PROJECT_SOURCE_DIR}/Spike/Helpers/*.cpp
  ${PROJECT_SOURCE_DIR}/Spike/Models/*.cpp
  ${PROJECT_SOURCE_DIR}/Spike/Neurons/*.cpp
  ${PROJECT_SOURCE_DIR}/Spike/Plasticity/*.cpp
  ${PROJECT_SOURCE_DIR}/Spike/RecordingElectrodes/*.cpp
  ${PROJECT_SOURCE_DIR}/Spike/Simulator/*.cpp
  ${PROJECT_SOURCE_DIR}/Spike/SpikeAnalyser/*.cpp
  ${PROJECT_SOURCE_DIR}/Spike/Synapses/*.cpp
  )
file(GLOB_RECURSE SPIKE_CUDA_FILES ${PROJECT_SOURCE_DIR}/Spike/Backend/*.cu)
file(GLOB_RECURSE SPIKE_DUMMY_FILES ${PROJECT_SOURCE_DIR}/Spike/Backend/Dummy/*.cpp)

set(WHOLE_ARCHIVE_FLAG "-Wl,--whole-archive")
set(NO_WHOLE_ARCHIVE_FLAG "-Wl,--no-whole-archive")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(SpikeDummy STATIC
  ${SPIKE_DUMMY_FILES}
)

add_library(Spike SHARED
  ${SPIKE_MAIN_FILES}
)

target_link_libraries(Spike PRIVATE ${WHOLE_ARCHIVE_FLAG}
  SpikeDummy
  ${NO_WHOLE_ARCHIVE_FLAG}
)


set(CUDA_NVCC_FLAGS "-std=c++11")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-arch=sm_60")
function(add_cuda_exe_lib name files libraries is_lib)
    set (obj_list)
    foreach(file ${files})
        get_filename_component(barename ${file} NAME)
        add_custom_command(
            OUTPUT ${barename}.o
            DEPENDS ${file} 
            COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --blue "Building NVCC Device object ${file}"
            COMMAND ${CUDA_NVCC_EXECUTABLE} ${CUDA_NVCC_FLAGS} -I${PROJECT_SOURCE_DIR} -Xcompiler '-fPIC' -dc "${file}" -o ${barename}.o
            COMMENT "Building ${file}"
            VERBATIM
        )

        LIST(APPEND obj_list ${barename}.o)
    endforeach()

    set (lib_list)
    #LIST(APPEND lib_list "-lcudadevrt")
    #LIST(APPEND lib_list "-lcudart")
    foreach(library_name ${libraries})
        LIST(APPEND lib_list "-l${library_name}")
    endforeach()

    set (flags ${CUDA_NVCC_FLAGS})
    if (is_lib)
        LIST(APPEND flags "-dlink")
        set (obj_name "${CMAKE_CURRENT_BINARY_DIR}/lib${name}")
    else()
        set (obj_name "${CMAKE_CURRENT_BINARY_DIR}/${name}")
    endif()

    add_custom_target(${name} ALL 
        COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --red "Linking CXX executable ${name}"
        COMMAND ${CUDA_NVCC_EXECUTABLE} ${flags} -Xcompiler '-fPIC' ${lib_list} -o ${obj_name}.dlink.o ${obj_list} -lcudadevrt
        #COMMAND g++ -shared -o ${obj_name}.so ${obj_list} ${obj_name}.dlink.o ${CUDART}
        COMMAND ${CUDA_NVCC_EXECUTABLE} ${CUDA_NVCC_FLAGS} ${lib_list} -lib -o ${obj_name}.a ${obj_list} ${obj_name}.dlink.o -lcudadevrt
        
        DEPENDS ${obj_list}
        COMMENT "Linking ${name}"
    )
endfunction()

function(add_cuda_exe name files libraries)
    add_cuda_exe_lib(${name} "${files}" "${libraries}" OFF)
endfunction()

function(add_cuda_lib name files libraries)
    add_cuda_exe_lib(${name} "${files}" "${libraries}" ON)
endfunction()

add_cuda_lib(SpikeCUDA
  "${SPIKE_CUDA_FILES}"
  "")

add_dependencies(Spike
  SpikeCUDA
)
target_link_libraries(Spike PRIVATE ${WHOLE_ARCHIVE_FLAG}
  ${CMAKE_CURRENT_BINARY_DIR}/libSpikeCUDA.a
  ${CUDA_LIBRARIES}
  ${NO_WHOLE_ARCHIVE_FLAG}
)

