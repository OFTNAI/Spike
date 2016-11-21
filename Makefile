#--------------------------------------------------------------------------------
#
# Simple makefile for compiling the Spike Simulator
#
#--------------------------------------------------------------------------------

# Compiler = Nvidia version for CUDA code
CC = nvcc
# Flags
# -c = ignore the fact that there is no main
# --compiler-options -Wall = Warnings All. Give them to me.
# Wall flag is inefficient
CFLAGS = -c

# CFLAGS += -lineinfo

# Mac OS X 10.9+ uses libc++, which is an implementation of c++11 standard library. 
# We must therefore specify c++11 as standard for out of the box compilation on Linux. 
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
	CFLAGS += -D_MWAITXINTRIN_H_INCLUDED
	CFLAGS += -std=c++11
endif


# mkdir -p ${EXPERIMENT_DIRECTORY}/bin
# test -d ${EXPERIMENT_DIRECTORY}/bin || mkdir ${EXPERIMENT_DIRECTORY}/bin


# Include all of the folders from which we want to include files
# CU
SIM_FILES := $(wildcard Simulator/*.cu)
MODEL_FILES := $(wildcard Models/*.cu)
EXPERIMENT_FILES := $(wildcard Experiments/*.cu)
NEUR_FILES := $(wildcard Neurons/*.cu)
STDP_FILES := $(wildcard STDP/*.cu)
HELP_FILES := $(wildcard Helpers/*.cu)
SYNS_FILES := $(wildcard Synapses/*.cu)
REC_FILES := $(wildcard RecordingElectrodes/*.cu)
ANALY_FILES := $(wildcard SpikeAnalyser/*.cu)
OPTIM_FILES := $(wildcard Optimiser/*.cu)
# CPP
HELP_CPP_FILES := $(wildcard Helpers/*.cpp)
# PLOTTING_FILES := $(wildcard Plotting/*.cpp)
PLOTTING_FILES := $

# COMBINE LISTS
CU_FILES := $(SIM_FILES) $(NEUR_FILES) $(EXPERIMENT_FILES) $(STDP_FILES) $(HELP_FILES) $(SYNS_FILES) $(REC_FILES) $(ANALY_FILES) $(OPTIM_FILES) $(MODEL_FILES)
CPP_FILES := $(HELP_CPP_FILES) $(PLOTTING_FILES)

# Create Objects
CU_OBJ_FILES := $(addprefix obj/,$(notdir $(CU_FILES:.cu=.o)))
CPP_OBJ_FILES := $(addprefix obj/,$(notdir $(CPP_FILES:.cpp=.o)))


# Default
model: ${FILE}
directory: ${EXPERIMENT_DIRECTORY}


${FILE}: obj/${FILE}.o $(CU_OBJ_FILES) $(CPP_OBJ_FILES)
	test -d ${EXPERIMENT_DIRECTORY}/binaries || mkdir ${EXPERIMENT_DIRECTORY}/binaries
	$(CC) -lineinfo obj/${FILE}.o $(CU_OBJ_FILES) $(CPP_OBJ_FILES) -o ${EXPERIMENT_DIRECTORY}/binaries/${FILE}

# Compiling the Model file
obj/${FILE}.o: ${EXPERIMENT_DIRECTORY}/${FILE}.cpp
	$(CC) $(CFLAGS) ${EXPERIMENT_DIRECTORY}/${FILE}.cpp -o $@

# CUDA
obj/%.o: */%.cu
	$(CC) $(CFLAGS) -o $@ $<

# CPP
obj/%.o: */%.cpp
	$(CC) $(CFLAGS) -o $@ $<




# Test Files
TEST_CPP_FILES := $(wildcard Tests/*.cu)
TEST_OBJ_FILES := $(addprefix Tests/obj/,$(notdir $(TEST_CPP_FILES:.cu=.o)))

test: ${TEST_OBJ_FILES} $(CU_OBJ_FILES) $(CPP_OBJ_FILES)
	$(CC) -lineinfo ${TEST_OBJ_FILES} $(CU_OBJ_FILES) $(CPP_OBJ_FILES) -o Tests/unittests

# Test Compilation
Tests/obj/%.o: Tests/%.cu
	$(CC) $(CFLAGS) -o $@ $<

# Cleaning tests
cleantest:
	rm Tests/obj/* Tests/unittests Tests/output/*



# Removing all created files
clean:
	rm obj/*.o Tests/obj/*
