#==========================================================================
# Makefile
#==========================================================================
# @brief: A makefile the compiles and runs the cordic program
#
# @desc: 1. "make float-sw" compiles & executes the floating-point implementation as software
#        2. "make float-hw" synthesizes the program with Vivado HLS and gathers resource utilization info
#        3. "make float" does (1) and (2)
#        4. "make fixed-sw" compiles & executes the fixed-point implementation as software
#        5. "make fixed-hw" synthesizes the program with Vivado HLS and gathers resource utilization info
#        6. "make fixed" does (4) and (5)
#        7. "make clean" cleans up the directory

# Extract Vivado HLS include path
XILINX_VIVADO?=/opt/xilinx/Vivado/2019.2
XIL_HLS=source $(XILINX_VIVADO)/settings64.sh; vivado_hls
VHLS_INC=$(XILINX_VIVADO)/include
CXXFLAGS=-g -I${VHLS_INC}

.PHONY: all float fixed clean

all: float fixed

# Floating-point targets
float-sw: result/float_sw.txt
float-hw: result/float_result.csv
float: float-sw float-hw

# Fixed-point targets
fixed-sw: result/fixed_sw.txt
fixed-hw: result/fixed_result.csv
fixed: fixed-sw fixed-hw

# compile floating-point implementation
cordic_float: cordic.h cordic.cpp cordic_test.cpp
	@echo "Compiling & executing SW floating-point implementation ..."
	g++ $(CXXFLAGS) cordic.cpp cordic_test.cpp -o $@
 
# compile fixed-point implementation
cordic_fixed: cordic.h cordic.cpp cordic_test.cpp
	@echo "Compiling & executing SW fixed-point implementation ..."
	g++ $(CXXFLAGS) -DFIXED_TYPE cordic.cpp cordic_test.cpp -o $@

result/%_sw.txt: cordic_%
	mkdir -p result
	./$< | tee $@

# Make hw. Compile sw version to catch syntax errors quicker
result/%_result.csv: run_%.tcl cordic_%
	mkdir -p result
	$(XIL_HLS) -f $<
	[ -e $@ ] && echo “Output produced”

# Clean up the directory before submission
clean:
	@echo "Clean up output files"
	rm -rf result *.prj *.log *.dat cordic_float cordic_fixed
