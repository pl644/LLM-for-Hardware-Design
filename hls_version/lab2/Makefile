#==========================================================================
# Makefile
#==========================================================================
# @brief: A makefile the compiles and runs the digitrec program
#
# @desc: 1. Enter "make all" to make digitrec-sw and digitrec-hw
#        2. Enter "make digitrec-sw" to compile & execute the digitrec program
#        3. Enter "make digitrec-hw" to synthesize digitrec using run.tcl
#        4. Enter "make clean" to clean up the directory (before submission)

# Extract Vivado HLS include path
XILINX_VIVADO?=/opt/xilinx/Vivado/2019.2
XIL_HLS=source $(XILINX_VIVADO)/settings64.sh; vivado_hls
VHLS_INC=$(XILINX_VIVADO)/include
# Specify compilation flags
CFLAGS=-g -I${VHLS_INC}

K_VALUES=3
# TIP: You can add multiple values of K: just add a space between values like so:
# K_VALUES=1 2 3 4 5

SW_RESULTS=$(patsubst %,result/digitrec_%-nn_sw_result.txt,$(K_VALUES))

.PHONY: all clean digitrec-sw digitrec-hw

# Compile & run digitrec implementation
all: digitrec-sw digitrec-hw

digitrec_%-nn_tb: digitrec.cpp digitrec_test.cpp
	g++ ${CFLAGS} -DK_CONST=$* $^ -o $@

result/digitrec_%-nn_sw_result.txt: digitrec_%-nn_tb
	mkdir -p result
	./$< | tee $@

digitrec-sw: $(SW_RESULTS)

result/knn_result.csv: run.tcl digitrec.cpp digitrec_test.cpp
	$(XIL_HLS) -f $<

digitrec-hw: result/knn_result.csv

# Clean up the directory before submission
clean:
	@echo "Clean up output files"
	rm -rf digitrec_*-nn_tb vivado_hls.log *nn.prj result out.dat *~
