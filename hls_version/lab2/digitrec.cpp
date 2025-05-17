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

  // Initialize the knn set to a value larger than max possible distance (49)
  knn_init_outer: for (int i = 0; i < 10; ++i)
    knn_init_inner: for (int k = 0; k < K_CONST; ++k)
      knn_set[i][k] = 50;

  // Scan through all training instances
  training_outer: for (int idx = 0; idx < TRAINING_SIZE; ++idx) {
    training_inner: for (int cls = 0; cls < 10; ++cls) {
      digit train_inst = training_data[cls * TRAINING_SIZE + idx];
      // Update the KNN bucket for this class
      update_knn(input, train_inst, knn_set[cls]);
    }
  }

  // Vote among the global K closest
  return knn_vote(knn_set);
}

//-----------------------------------------------------------------------
// update_knn function
//-----------------------------------------------------------------------
// Maintain a sorted ascending array of the K smallest distances
// for one class. This does insertion into the sorted array.
//-----------------------------------------------------------------------
void update_knn(digit test_inst, digit train_inst,
                bit6 min_distances[K_CONST]) {

  // Hamming distance
  digit diff = test_inst ^ train_inst;
  bit6 dist = 0;
  popcount: for (int i = 0; i < 49; ++i) {
    dist += diff[i];
  }

  // Insert 'dist' into the sorted min_distances if it belongs
  ins_outer: for (int k = 0; k < K_CONST; ++k) {
    // If this slot is larger than the new dist, we shift down and insert
    if (dist < min_distances[k]) {
      // Shift the tail down by one
      ins_shift: for (int m = K_CONST - 1; m > k; --m) {
        min_distances[m] = min_distances[m - 1];
      }
      min_distances[k] = dist;
      break;
    }
  }
}

//-----------------------------------------------------------------------
// sort_knn and knn_vote unchanged except minor formatting
//-----------------------------------------------------------------------
void sort_knn(
  bit6 knn_set[10][K_CONST],
  bit6 sorted_distances[10*K_CONST],
  bit4 sorted_labels[10*K_CONST]
) {
  flatten_outer: for (bit4 i = 0; i < 10; i++) {
    flatten_inner: for (int j = 0; j < K_CONST; j++) {
      sorted_distances[i * K_CONST + j] = knn_set[i][j];
      sorted_labels[i * K_CONST + j]    = i;
    }
  }
  const int NUM_ENTRIES = 10 * K_CONST;
  bubble_outer: for (int i = 0; i < NUM_ENTRIES; i++) {
    bubble_inner: for (int j = 0; j < NUM_ENTRIES - 1; j++) {
      if (sorted_distances[j] > sorted_distances[j+1]) {
        bit6 td = sorted_distances[j];
        bit4 tl = sorted_labels[j];
        sorted_distances[j] = sorted_distances[j+1];
        sorted_labels[j]    = sorted_labels[j+1];
        sorted_distances[j+1] = td;
        sorted_labels[j+1]    = tl;
      }
    }
  }
}

bit4 knn_vote(bit6 knn_set[10][K_CONST]) {
  bit6 sorted_distances[10*K_CONST];
  bit4 sorted_labels[10*K_CONST];
  sort_knn(knn_set, sorted_distances, sorted_labels);

  // tally the K_CONST nearest
  bit4 vote_count[10];
  vote_init: for (int i = 0; i < 10; ++i) vote_count[i] = 0;
  vote_accum: for (int i = 0; i < K_CONST; ++i) {
    bit4 lbl = sorted_labels[i];
    vote_count[lbl]++;
  }
  // pick the max
  bit4 best = 0, best_count = vote_count[0];
  vote_decide: for (bit4 d = 1; d < 10; ++d) {
    if (vote_count[d] > best_count) {
      best = d;
      best_count = vote_count[d];
    }
  }
  return best;
}
