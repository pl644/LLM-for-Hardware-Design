# ECE 6775 FA'24 Lab2: Digit Recognition System (Part 1)
- Vivado HLS 2019.2

## File tree structure
- Source files (input to Vivado HLS)

```shell
├── data
│   ├── testing_set.dat        # Testing dataset
│   ├── training_set_0.dat     # Training dataset for digit 0
│   ├── training_set_1.dat
│   ├── training_set_2.dat
│   ├── training_set_3.dat
│   ├── training_set_4.dat
│   ├── training_set_5.dat
│   ├── training_set_6.dat
│   ├── training_set_7.dat
│   ├── training_set_8.dat
│   └── training_set_9.dat
├── digitrec.cpp	       # C++ source
├── digitrec.h                 # C++ header
├── digitrec_test.cpp          # Testbench
├── Makefile                   # Makefile. run `make all|digitrec-sw|digitrec-hw|clean`
├── run.tcl		       # Tcl script to synthesize design
├── script
│   └── collect_result.tcl     # Tcl script to parse synthesis reports
├── training_data.h            # C++ header that combines training data in data/
└── typedefs.h		       # C++ header that defines data types used in design
```

- Output files by Vivado HLS
```shell
# Example project directory (K_CONST=3)
$ tree 3-nn.prj/ -L 2

3-nn.prj/
├── solution1
│   ├── csim
│   ├── solution1.aps
│   ├── solution1.directive
│   └── solution1.log
└── vivado_hls.app

```
