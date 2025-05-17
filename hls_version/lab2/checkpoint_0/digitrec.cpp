//==========================================================================
// digitrec.cpp
//==========================================================================
// @brief: A k-nearest-neighbors implementation for digit recognition

#include "digitrec.h"

//----------------------------------------------------------
// Top function
//----------------------------------------------------------
// @param[in] : input - the testing instance
// @return : the recognized digit (0~9)

bit4 digitrec(digit input) {
#include "training_data.h"

  // This array stores K minimum distances per training set
  bit6 knn_set[10][K_CONST];

  // Initialize the knn set
  knn_init_outer: for (int i = 0; i < 10; ++i)
    knn_init_inner: for (int k = 0; k < K_CONST; ++k)
      // Note that the max distance is 49
      knn_set[i][k] = 50;

  training_outer: for (int i = 0; i < TRAINING_SIZE; ++i) {
    training_inner: for (int j = 0; j < 10; j++) {

      // Read a new instance from the training set
      digit training_instance = training_data[j * TRAINING_SIZE + i];
      // Update the KNN set
      update_knn(input, training_instance, knn_set[j]);
    }
  }

  // Compute the final output
  return knn_vote(knn_set);
}

//-----------------------------------------------------------------------
// update_knn function
//-----------------------------------------------------------------------
// Given the test instance and a (new) training instance, this
// function maintains/updates an array of K minimum
// distances per training set.

// @param[in] : test_inst - the testing instance
// @param[in] : train_inst - the training instance
// @param[in/out] : min_distances[K_CONST] - the array that stores the current
//                  K_CONST minimum distance values per training set

void update_knn(digit test_inst, digit train_inst,
                bit6 min_distances[K_CONST]) {

  // Compute the bitwise difference between two digits using XOR
  digit diff = test_inst ^ train_inst;
  bit6 dist = 0;
  // Calculate the hamming distance by counting the number of 1s
  popcount: for (int i = 0; i < 49; ++i) {
    dist += diff[i];
  }

  // -----------------------------
  // YOUR CODE GOES HERE
  // -----------------------------
}


//-----------------------------------------------------------------------
// sort_knn function
//-----------------------------------------------------------------------
// Given 10xK minimum distance values, this function
// flattens it into an one-dimensional array and sort it
// in ascending order according to the distances
// @param[in] : knn_set - 10xK_CONST min distance values
// @param[out] : sorted_distances - 10xK_CONST distances sorted in ascending order
// @param[out] : sorted_labels - the corresponding labels of the sorted distances

void sort_knn(
  bit6 knn_set[10][K_CONST],
  bit6 sorted_distances[10*K_CONST],
  bit4 sorted_labels[10*K_CONST]
) {
  // flatten knn_set into 1-d arrays
  flatten_outer:for (bit4 i = 0; i < 10; i++) {
    flatten_inner:for (int j = 0; j < K_CONST; j++) {
      sorted_distances[i * K_CONST + j] = knn_set[i][j];
      sorted_labels[i * K_CONST + j] = i;
    }
  }

  // bubble sort
  const int NUM_ENTRIES = 10 * K_CONST;
  bubble_outer: for (int i = 0; i < NUM_ENTRIES; i++) {
    bubble_inner: for (int j = 0; j < NUM_ENTRIES - 1; j++) {
      if (sorted_distances[j] > sorted_distances[j+1]) {
        bit6 t = sorted_distances[j+1];
        bit4 l = sorted_labels[j+1];
        sorted_distances[j+1] = sorted_distances[j];
        sorted_labels[j+1] = sorted_labels[j];
        sorted_distances[j] = t;
        sorted_labels[j] = l;
      }
    }
  }
}

//-----------------------------------------------------------------------
// knn_vote function
//-----------------------------------------------------------------------
// Given 10xK minimum distance values, this function
// finds the actual K nearest neighbors and determines the
// final output based on the most common digit represented by
// these nearest neighbors (i.e., a vote among KNNs).
//
// @param[in] : knn_set - 10xK_CONST min distance values
// @return : the recognized digit (0-9)
//

bit4 knn_vote(bit6 knn_set[10][K_CONST]) {
  bit6 sorted_distances[10*K_CONST];
  bit4 sorted_labels[10*K_CONST];

  sort_knn(knn_set, sorted_distances, sorted_labels);

  // -----------------------------
  // YOUR CODE GOES HERE
  // -----------------------------
}
