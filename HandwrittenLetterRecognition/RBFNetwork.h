
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <random>
#include <iostream>

class RBFNetwork {
private:
    std::vector<std::vector<float>> centers;
    std::vector<std::vector<float>> weights;
    float beta;

public:
    // Default constructor
    RBFNetwork() : beta(0.0) {
        // Optionally initialize other members to default states
    }
    RBFNetwork(const std::vector<std::vector<float>>& trainingInputs, const std::vector<std::vector<float>>& targets, int numCenters, float beta) : beta(beta) {
      
        initializeCentersKMeans(trainingInputs, numCenters);
        initializeWeights(targets);
    }

    void initializeCentersKMeans(const std::vector<std::vector<float>>& trainingInputs, int numCenters);
    float calculateDistanceSquared(const std::vector<float>& v1, const std::vector<float>& v2);

    void setup(const std::vector<std::vector<float>>& trainingInputs, const std::vector<std::vector<float>>& targets, int numCenters, float newBeta) {
        this->beta = newBeta;

        // Reinitialize centers using K-Means or another method
        initializeCentersKMeans(trainingInputs, numCenters);

        // Reinitialize weights based on the new targets or another criterion
        initializeWeights(targets);
    }

    std::vector<float> predict(const std::vector<float>& input) {
        if (centers.empty() || weights.empty()) {
            std::cerr << "Error: Centers or weights are empty." << std::endl;
            return {};
        }

        std::vector<float> rbfActivations(centers.size(), 0.0);

        // Calculate the activation for each RBF
        for (size_t i = 0; i < centers.size(); ++i) {
            float distance = calculateDistance(input, centers[i]);
            rbfActivations[i] = calculateRBF(distance);
        }

        // Assuming a single output per class, sum the weighted RBF activations to compute the output
        std::vector<float> output(weights[0].size(), 0.0); // Initialize output vector with the size equal to the number of classes

        for (size_t i = 0; i < weights.size(); ++i) {
            for (size_t j = 0; j < weights[i].size(); ++j) {
                output[j] += rbfActivations[i] * weights[i][j];
            }
        }

        return output; // This vector contains the score for each class
    }


    void runKMeans(const std::vector<std::vector<float>>& trainingInputs, int numCenters);

    int findClosestCenter(const std::vector<float>& input);

    void train(const std::vector<std::vector<float>>& trainingInputs, const std::vector<std::vector<float>>& trainingTargets);

    void initializeWeights(const std::vector<std::vector<float>>& targets);
    float calculateRBF(float distance);

    std::vector<float> processInput(const std::vector<float>& input, const std::vector<std::vector<float>>& targets);

    float test(const std::vector<std::vector<float>>& testInputs, const std::vector<std::vector<float>>& testTargets);

    bool isPredictionCorrect(const std::vector<float>& output, const std::vector<float>& target);

private:
    std::vector<float> calculateError(const std::vector<float>& output, const std::vector<float>& target);

    void updateWeights(const std::vector<float>& input, const std::vector<float>& error, float learningRate);

    float calculateTotalError(const std::vector<float>& error);
    float calculateDistance(const std::vector<float>& v1, const std::vector<float>& v2);
};