#include "RBFNetwork.h"



void RBFNetwork::initializeWeights(const std::vector<std::vector<float>>& targets) {
    if (centers.empty() || targets.empty()) {
        std::cerr << "Error: Centers or targets are empty." << std::endl;
        return;
    }

    weights.assign(centers.size(), std::vector<float>(targets[0].size()));
    for (auto& row : weights) {
        for (auto& weight : row) {
            weight = static_cast<float>(rand()) / RAND_MAX; // Valor aleatorio entre 0 y 1
        }
    }

    //// Initialize weights using targets
    //std::default_random_engine generator;
    //std::uniform_real_distribution<float> distribution(-1.0, 1.0);

    //for (size_t i = 0; i < centers.size(); ++i) {
    //    std::vector<float> tempWeights;
    //    for (size_t j = 0; j < targets[0].size(); ++j) {
    //        tempWeights.push_back(distribution(generator));
    //    }
    //    weights.push_back(tempWeights);
    //}
}

void RBFNetwork::initializeCentersKMeans(const std::vector<std::vector<float>>& trainingInputs, int numCenters) {
    if (trainingInputs.empty()) {
        std::cerr << "Error: Training data is empty." << std::endl;
        return;
    }
    /*//std::vector<std::vector<float>> shuffledInputs = trainingInputs;

    //std::random_device rd;
    //std::mt19937 g(rd());

    //// Distribution
    //for (size_t i = shuffledInputs.size() - 1; i > 0; --i) {
    //    std::uniform_int_distribution<size_t> distribution(0, i);
    //    size_t j = distribution(g);

    //    if (i != j) {
    //        std::swap(shuffledInputs[i], shuffledInputs[j]);
    //    }
    //}

    ////Assign centers 
    //centers.assign(shuffledInputs.begin(), shuffledInputs.begin() + numCenters);
    // 
    //K-means
    //runKMeans(trainingInputs, numCenters);*/

    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_int_distribution<size_t> distribution(0, trainingInputs.size() - 1);
    size_t firstCentroidIndex = distribution(generator);
    centers.push_back(trainingInputs[firstCentroidIndex]);

    // Calculates the squared distances from each point to the nearest existing centroid
    std::vector<float> minDistances(trainingInputs.size(), std::numeric_limits<float>::max());

    for (int i = 1; i < numCenters; ++i) {
        for (size_t j = 0; j < trainingInputs.size(); ++j) {
            float distance = calculateDistanceSquared(trainingInputs[j], centers.back());
            minDistances[j] = std::min(minDistances[j], distance);
        }

        // Selects the next centroid with a probability proportional to the distances
        std::discrete_distribution<size_t> probDistribution(minDistances.begin(), minDistances.end());
        size_t nextCentroidIndex = probDistribution(generator);
        centers.push_back(trainingInputs[nextCentroidIndex]);
    }    
}


float RBFNetwork::calculateDistanceSquared(const std::vector<float>& v1, const std::vector<float>& v2) {
    if (v1.size() != v2.size()) {
        std::cerr << "Error: Vectors must have the same dimension for distance calculation." << std::endl;
        return -1.0;
    }

    float distanceSquared = 0.0;

    for (size_t i = 0; i < v1.size(); ++i) {
        distanceSquared += (v1[i] - v2[i]) * (v1[i] - v2[i]);
    }

    return distanceSquared;
}

/*void RBFNetwork::runKMeans(const std::vector<std::vector<float>>& trainingInputs, int numCenters) {

    const int maxIterations = 10;

    for (int iteration = 0; iteration < maxIterations; ++iteration) {
        // Assign each point to the nearest center
        std::vector<int> assignments(trainingInputs.size());

        for (size_t i = 0; i < trainingInputs.size(); ++i) {
            assignments[i] = findClosestCenter(trainingInputs[i]);
        }

        //  Calculate new centers
        std::vector<std::vector<float>> newCenters(numCenters, std::vector<float>(trainingInputs[0].size(), 0.0));
        std::vector<int> counts(numCenters, 0);

        for (size_t i = 0; i < trainingInputs.size(); ++i) {
            int closestCenter = assignments[i];
            for (size_t j = 0; j < trainingInputs[i].size(); ++j) {
                newCenters[closestCenter][j] += trainingInputs[i][j];
            }
            counts[closestCenter]++;
        }

        // Update the centers
        for (int i = 0; i < numCenters; ++i) {
            if (counts[i] > 0) {
                for (size_t j = 0; j < newCenters[i].size(); ++j) {
                    centers[i][j] = newCenters[i][j] / counts[i];
                }
            }
        }
    }
}*/
//
//int RBFNetwork::findClosestCenter(const std::vector<float>& input) {
//    int closestCenter = 0;
//    float minDistance = calculateDistance(input, centers[0]);
//
//    for (size_t i = 1; i < centers.size(); ++i) {
//        float distance = calculateDistance(input, centers[i]);
//        if (distance < minDistance) {
//            minDistance = distance;
//            closestCenter = i;
//        }
//    }
//
//    return closestCenter;
//}

void RBFNetwork::train(const std::vector<std::vector<float>>& trainingInputs, 
    const std::vector<std::vector<float>>& trainingTargets) {
    if (trainingInputs.size() != trainingTargets.size() || trainingInputs.empty()) {
        std::cerr << "Error: Mismatched or empty training data." << std::endl;
        return;
    }
    float learningRate = 0.005;
    int maxEpochs = 100;
    for (int epoch = 0; epoch < maxEpochs; ++epoch) {
        float totalError = 0.0;

        for (size_t i = 0; i < trainingInputs.size(); ++i) {
            std::vector<float> output = processInput(trainingInputs[i], trainingTargets);

            std::vector<float> error = calculateError(output, trainingTargets[i]);

            // Update weights
            updateWeights(trainingInputs[i], error, learningRate);

            // Accumulate error 
            totalError += calculateTotalError(error);
        }

        std::cout << "Epoch " << epoch + 1 << ", Total Error: " << totalError << std::endl;
    }
}

std::vector<float> RBFNetwork::processInput(const std::vector<float>& input,
    const std::vector<std::vector<float>>& targets) {
    if (centers.empty() || weights.empty() || targets.empty()) {
        std::cerr << "Error: Centers, weights, or targets are empty." << std::endl;
        return {};
    }

    std::vector<float> output(targets[0].size(), 0.0);

    for (size_t i = 0; i < centers.size(); ++i) {
        float distance = calculateDistance(input, centers[i]);
        float rbfValue = calculateRBF(distance);

        for (size_t j = 0; j < output.size(); ++j) {
            output[j] += rbfValue * weights[i][j];
        }
    }

    return output;
}

float RBFNetwork::calculateDistance(const std::vector<float>& v1, 
    const std::vector<float>& v2) {
    if (v1.size() != v2.size()) {
        std::cerr << "Error: Vectors must have the same dimension"
            << std::endl;
        return -1.0;
    }

    float distance = 0.0;

    for (size_t i = 0; i < v1.size(); ++i) {
        distance += (v1[i] - v2[i]) * (v1[i] - v2[i]);
    }

    return std::sqrt(distance);
}

float RBFNetwork::calculateRBF(float distance) {
    return exp(-beta * distance);
}

void RBFNetwork::updateWeights(const std::vector<float>& input,
    const std::vector<float>& error, float learningRate) {

    for (size_t i = 0; i < centers.size(); ++i) {
        float rbfValue = calculateRBF(calculateDistance(input, centers[i]));
        for (size_t j = 0; j < weights[i].size(); ++j) {
            weights[i][j] += learningRate * error[j] * rbfValue;
        }
    }
}


float RBFNetwork::test(const std::vector<std::vector<float>>& testInputs,
    const std::vector<std::vector<float>>& testTargets) {
    if (testInputs.size() != testTargets.size() || testInputs.empty()) {
        std::cerr << "Error: Mismatched or empty test data." << std::endl;
        return -1.0;  // Indicate an error with a negative value
    }

    int correctPredictions = 0;

    for (size_t i = 0; i < testInputs.size(); ++i) {
        // Forward pass
        std::vector<float> output = processInput(testInputs[i], testTargets);

        // Check if the predicted output matches the target
        if (isPredictionCorrect(output, testTargets[i])) {
            correctPredictions++;
        }
    }

    // Calculate accuracy
    float accuracy = static_cast<float>(correctPredictions) / testInputs.size();
    return accuracy;
}

bool RBFNetwork::isPredictionCorrect(const std::vector<float>& output, const std::vector<float>& target) {

    int predictedClass = std::distance(output.begin(), std::max_element(output.begin(), output.end()));
    int trueClass = std::distance(target.begin(), std::max_element(target.begin(), target.end()));
    return predictedClass == trueClass;
}

std::vector<float> RBFNetwork::calculateError(const std::vector<float>& output, const std::vector<float>& target) {

    std::vector<float> error(output.size(), 0.0);
    for (size_t i = 0; i < output.size(); ++i) {
        error[i] = target[i] - output[i];
    }
    return error;
}



float RBFNetwork::calculateTotalError(const std::vector<float>& error) {

    float totalError = 0.0;
    for (float e : error) {
        totalError += e * e;
    }
    return totalError;
}
