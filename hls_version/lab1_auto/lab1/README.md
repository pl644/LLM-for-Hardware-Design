# ECE 6775 FA'24 Lab1: CORDIC design
- Vivado HLS 2019.2

## File tree structure
- Source files (input to Vivado HLS)

```shell
├── Makefile                    # Makefile. Run `make all|fixed-sw|fixed-hw|fixed|float-sw|float-hw|float|clean`
├── cordic.cpp                  # C++ source the one you need to adjust
├── cordicbase.cpp                  # C++ source the one give you the baseline
├── cordic.h                    # C++ header
├── cordic_test.cpp             # Testbench
├── run_fixed.tcl               # Tcl script to synthesize fixed-point design
├── run_float.tcl               # Tcl script to synthesize floating-point design
├── run_opt.tcl                 # Tcl script to synthesize optimized fixed-point design
└── script
    └── collect_result.tcl      # Tcl script to parse synthesis reports
```

- Output files by Vivado HLS
```shell
# Example project directory (TOT_W=32, INT_W=16, STEP=20)
$ tree fixed_32_16_20.prj/ -L 2

fixed_32_16_20.prj/
├── solution1
│   ├── csim
│   ├── impl
│   ├── sim
│   └── syn
└── vivado_hls.app
```
