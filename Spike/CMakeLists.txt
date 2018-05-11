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

add_library(SpikeMain SHARED
  ${SPIKE_MAIN_FILES}
)

target_link_libraries(SpikeMain PRIVATE ${WHOLE_ARCHIVE_FLAG}
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

add_executable(VogelsAbbottNet
  ${PROJECT_SOURCE_DIR}/Examples/VogelsAbbottNet.cpp
)

add_dependencies(SpikeMain
  SpikeCUDA
)
target_link_libraries(SpikeMain PRIVATE ${WHOLE_ARCHIVE_FLAG}
  ${CMAKE_CURRENT_BINARY_DIR}/libSpikeCUDA.a
  ${CUDA_LIBRARIES}
  ${NO_WHOLE_ARCHIVE_FLAG}
)

target_link_libraries(VogelsAbbottNet
  SpikeMain
)














# COMMENTED
#[[
add_custom_target(SpikeDummy ALL
  COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --red "Producing SpikeDummy"
  COMMAND ${CMAKE_CXX_COMPILER} --std=c++11 -I${PROJECT_SOURCE_DIR} -o SpikeDummy.o ${SPIKE_DUMMY_FILES} 
  COMMAND ar rvs SpikeDummy.a SpikeDummy.o
  )

add_custom_target(SpikeMain ALL
  DEPENDS SpikeDummy
  COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --red "Producing SpikeMain"
  COMMAND ${CMAKE_CXX_COMPILER} --std=c++11 -I${PROJECT_SOURCE_DIR} -c ${SPIKE_MAIN_FILES} SpikeDummy.a -o SpikeMain.o
  COMMAND ar rvs SpikeMain.a SpikeMain.o
  )



add_custom_target(Spike ALL
  DEPENDS SpikeDummy
  DEPENDS SpikeMain
  COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --red "Linking SpikeDummy and SpikeMain"
  #COMMAND g++ --std=c++11 -I${PROJECT_SOURCE_DIR}/ ${PROJECT_SOURCE_DIR}/Examples/VogelsAbbottNet.cpp libSpikeCUDA.a -L${CMAKE_CURRENT_BINARY_DIR} -lSpikeCPP -Wl,-rpath=${CMAKE_CURRENT_BINARY_DIR} -o VogelsAbbottNet -L/usr/local/cuda/lib64 -lcuda -lcudart -lcudadevrt
  COMMAND ${CMAKE_CXX_COMPILER} --std=c++11 -I${PROJECT_SOURCE_DIR}  SpikeMain.a ${PROJECT_SOURCE_DIR}/Examples/VogelsAbbottNet.cpp -o VogelsAbbottNet
  )

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" "-std=c++11")
function(add_cuda_exe_lib name files libraries is_lib)
    set (obj_list)
    foreach(file ${files})
        add_custom_command(
            OUTPUT ${file}.o
            DEPENDS ${file} 
            COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --blue "Building NVCC Device object ${file}"
            COMMAND ${CUDA_NVCC_EXECUTABLE} ${CUDA_NVCC_FLAGS} -I${PROJECT_SOURCE_DIR} -dc -Xcompiler '-fPIC' "${file}" -o ${file}.o
            COMMENT "Building ${file}"
            VERBATIM
        )

        LIST(APPEND obj_list ${file}.o)
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
        COMMAND ${CUDA_NVCC_EXECUTABLE} ${flags} ${obj_list} ${lib_list} ${CUDART} -Xcompiler '-fPIC' -o ${obj_name}.dlink.o -lcudadevrt -lcudart
        #COMMAND g++ -shared -o ${obj_name}.so ${obj_list} ${obj_name}.dlink.o ${CUDART}
        COMMAND ${CUDA_NVCC_EXECUTABLE} ${obj_list} ${obj_name}.dlink.o ${lib_list} -Xcompiler '-fPIC' -lib -o ${obj_name}.a -lcudart -lcudadevrt
        
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


#set(CMAKE_POSITION_INDEPENDENT_CODE ON)
#add_library(SpikeCPP SHARED
#  ${SPIKE_CPP_FILES}
#)

add_cuda_lib(SpikeCUDA
  "${SPIKE_CU_FILES}"
  "")

]]



#[[
if (BUILD_SHARED_LIBS)
endif()

# Please do not use globbing: list targets explicitly!
# Listing targets / files makes for much easier debugging
# and clearer collaboration!
#
# See http://stackoverflow.com/a/18538444 for more info.

add_subdirectory(Backend)
add_subdirectory(Helpers)
add_subdirectory(Models)
add_subdirectory(Neurons)
add_subdirectory(Optimiser)
add_subdirectory(Plotting)
add_subdirectory(RecordingElectrodes)
add_subdirectory(Simulator)
add_subdirectory(SpikeAnalyser)
add_subdirectory(Plasticity)
add_subdirectory(Synapses)

add_library(Spike
  $<TARGET_OBJECTS:Backend>
  $<TARGET_OBJECTS:Helpers>
  $<TARGET_OBJECTS:Models>
  $<TARGET_OBJECTS:Neurons>
  $<TARGET_OBJECTS:Optimiser>
  $<TARGET_OBJECTS:RecordingElectrodes>
  $<TARGET_OBJECTS:Simulator>
  $<TARGET_OBJECTS:SpikeAnalyser>
  $<TARGET_OBJECTS:Plasticity>
  $<TARGET_OBJECTS:Synapses>
  )

# target_compile_features(Spike PUBLIC cxx_nullptr)

set(WHOLE_ARCHIVE_FLAG "-Wl,--whole-archive")
set(NO_WHOLE_ARCHIVE_FLAG "-Wl,--no-whole-archive")

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set(WHOLE_ARCHIVE_FLAG "-Wl,-all_load")
  set(NO_WHOLE_ARCHIVE_FLAG "-Wl,-noall_load")
endif()

target_link_libraries(Spike PRIVATE ${WHOLE_ARCHIVE_FLAG}
  SpikeDummyHelpers 
  SpikeDummyNeurons
  SpikeDummyRecordingElectrodes
  SpikeDummySpikeAnalyser
  SpikeDummyPlasticity
  SpikeDummySynapses
  ${NO_WHOLE_ARCHIVE_FLAG})

if (BUILD_WITH_CUDA)
  target_link_libraries(Spike PRIVATE ${WHOLE_ARCHIVE_FLAG}
    #SpikeCUDAHelpers
    #SpikeCUDANeurons
    #SpikeCUDARecordingElectrodes
    #SpikeCUDASpikeAnalyser
    #SpikeCUDAPlasticity
    #SpikeCUDASynapses
    SpikeCUDA
    ${NO_WHOLE_ARCHIVE_FLAG})
endif()

if (ENABLE_PLOTTING)
  target_link_libraries(Spike SpikePlotting)
endif()
]]
