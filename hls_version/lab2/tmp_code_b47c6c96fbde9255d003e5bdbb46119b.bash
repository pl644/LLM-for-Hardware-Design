# Write the updated header and source files
cat << 'EOF' > digitrec.h
#ifndef DIGITREC_H
#define DIGITREC_H

#include "typedefs.h"
#include "training_data.h"

// The K_CONST value: number of nearest neighbors
#ifndef K_CONST
#define K_CONST 3
#endif

// Top function for digit recognition
bit4 digitrec(digit input);

// Given the testing instance and a (new) training instance,
// this function is expected to maintain/update an array of
// K minimum distances per training set
void update_knn(digit test_inst, digit train_inst, bit6 min_distances[K_CONST]);

// Among 10xK minimum distance values, this function finds
// the actual K nearest neighbors and determine the final
// output based on the most common digit represented by these
// nearest neighbors (i.e., a vote among KNNs).
bit4 knn_vote(bit6 min_distances[10][K_CONST]);

#endif
EOF

cat << 'EOF' > digitrec.cpp
#include "digitrec.h"

// Function to calculate the distance (Euclidean or other) between training and test digit
bit6 calculate_distance(digit a, digit b) {
    return std::abs(a - b); // Using standard abs
}

// Main digit recognition function
bit4 digitrec(digit input) {
    bit6 min_distances[10][K_CONST]; // 10 possible digits, each with K neighbors

    // Initialize min_distances to max value
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < K_CONST; j++) {
            min_distances[i][j] = std::numeric_limits<bit6>::max();
        }
    }

    // Loop through all training instances
    for (int i = 0; i < TRAINING_SIZE; i++) {
        digit train_inst = training_data[i]; // Read from training_data
        bit6 distance = calculate_distance(input, train_inst);

        // Call update_knn to update distances
        update_knn(input, train_inst, min_distances[i]); // Pass the correct part of the array
    }

    // Call knn_vote to classify based on the nearest neighbors
    return knn_vote(min_distances);
}

// Update nearest neighbors based on distance
void update_knn(digit test_inst, digit train_inst, bit6 min_distances[K_CONST]) {
    bit6 distance = calculate_distance(test_inst, train_inst);
    
    // Logic to maintain/update the K minimum distances for each digit class
    for (int digit_index = 0; digit_index < 10; digit_index++) {
        // Check if the current distance is smaller than the max distance in the K list
        for (int j = 0; j < K_CONST; j++) {
            if (distance < min_distances[digit_index][j]) {
                // Found position to insert new distance
                // Shift larger distances to the right
                for (int k = K_CONST - 1; k > j; k--) {
                    min_distances[digit_index][k] = min_distances[digit_index][k - 1];
                }
                // Insert new distance
                min_distances[digit_index][j] = distance;
                break; // Once inserted, exit
            }
        }
    }
}

bit4 knn_vote(bit6 min_distances[10][K_CONST]) {
    // Voting logic
    int votes[10] = {0}; // Initialize vote counts for digits 0-9

    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < K_CONST; j++) {
            if (min_distances[i][j] < std::numeric_limits<bit6>::max()) {
                votes[i]++; // Increment vote for this digit
            }
        }
    }

    // Determine which digit got the most votes
    int max_votes = 0;
    bit4 voted_digit = 0;

    for (int i = 0; i < 10; i++) {
        if (votes[i] > max_votes) {
            max_votes = votes[i];
            voted_digit = i;
        }
    }

    return voted_digit; // Return the most voted digit
}
EOF

# Now, compile again
make digitrec-sw