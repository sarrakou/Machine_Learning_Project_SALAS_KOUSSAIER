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
    RBFNetwork(const std::vector<std::vector<float>>& trainingInputs, const std::vector<std::vector<float>>& targets, int numCenters, float beta) : beta(beta) {

        initializeCentersKMeans(trainingInputs, numCenters);
        initializeWeights(targets);
    }

    void initializeCentersKMeans(const std::vector<std::vector<float>>& trainingInputs, int numCenters);
    void initializeWeights(const std::vector<std::vector<float>>& targets);
    float calculateDistanceSquared(const std::vector<float>& v1, const std::vector<float>& v2);
    //void runKMeans(const std::vector<std::vector<float>>& trainingInputs, int numCenters);
    int findClosestCenter(const std::vector<float>& input);
    void train(const std::vector<std::vector<float>>& trainingInputs, const std::vector<std::vector<float>>& trainingTargets);
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
