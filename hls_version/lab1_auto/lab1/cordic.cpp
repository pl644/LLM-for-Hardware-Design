//=========================================================================
// cordic.cpp
//=========================================================================
// @brief : A CORDIC implementation of sine and cosine functions.

#include "cordic.h"
#include <math.h>
#include <iostream>

//-----------------------------------
// cordic function
//-----------------------------------
// @param[in]  : theta - input angle
// @param[out] : s - sine output
// @param[out] : c - cosine output
void cordic(theta_type theta, cos_sin_type &s, cos_sin_type &c) {
  // Initialize variables
  cos_sin_type acc_s = 0;
  cos_sin_type acc_c = 1;
  theta_type z = theta;
  
  // Implement CORDIC iteration
  for (int step = 0; step < NUM_ITER; step++) {
    if (z < 0) {
      // Clockwise rotation
      acc_s += acc_c * cordic_ctab[step];
      acc_c -= acc_s * cordic_ctab[step];
      z += cordic_ctab[step];
    } else {
      // Counterclockwise rotation
      acc_s -= acc_c * cordic_ctab[step];
      acc_c += acc_s * cordic_ctab[step];
      z -= cordic_ctab[step];
    }
  }
  
  // Assign the computed values to output
  s = acc_s;
  c = acc_c;
}
