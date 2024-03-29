#pragma once
#include <vector>
#include <iostream>
#include <numeric>
#include "json.hpp"
#include <iostream>
#include <fstream>

class SimpleSVM
{
private:
    std::vector<float> w;  // Weights
    float b;  // Bias
    float learning_rate;
    float lambda;  // Regularization parameter

public:
    // Default constructor with default values
    SimpleSVM()
        : w(), b(0.0f), learning_rate(0.0f), lambda(0.0f) {}

    SimpleSVM(int num_features, float lr, float lambda) : w(num_features, 0), b(0), learning_rate(lr), lambda(lambda) {}

    // Train function for a single data point
    void train(const std::vector<float>& input, int label) {
        // Calculate the decision function (dot product plus bias)
        float dot_product = std::inner_product(w.begin(), w.end(), input.begin(), 0.0f) + b;

        // Implementing the hinge loss update rule
        if (label * dot_product < 1) {
            // Misclassified or within margin
            for (size_t j = 0; j < w.size(); ++j) {
                // Update weights
                w[j] -= learning_rate * (2 * lambda * w[j] - label * input[j]);
            }
            // Update bias
            b -= learning_rate * (-label);
        }
        else {
            // Correctly classified and outside margin
            for (size_t j = 0; j < w.size(); ++j) {
                // Update weights for regularization only
                w[j] -= learning_rate * (2 * lambda * w[j]);
            }
        }
    }


    int predict(const std::vector<float>& input) {
        float dot_product = 0;
        for (size_t i = 0; i < w.size(); ++i) {
            dot_product += w[i] * input[i];
        }
        float y = dot_product + b;
        return (y >= 0) ? 1 : -1;
    }

    // Decision function to calculate the distance from the hyperplane
    float decisionFunction(const std::vector<float>& input) const{
        float score = 0.0;
        for (size_t i = 0; i < w.size(); ++i) {
            score += w[i] * input[i];  // Dot product of weight and input
        }
        score += b;  // Add the bias term

        return score;
    }


  /*   // Serialize the SVM model to a JSON object
    nlohmann::json serializeToJson() const {
        nlohmann::json jsonModel;
        jsonModel["weights"] = w;
        jsonModel["bias"] = b;
        jsonModel["learning_rate"] = learning_rate;
        jsonModel["lambda"] = lambda;
        return jsonModel;
    }

    // Deserialize the SVM model from a JSON object
    void deserializeFromJson(const nlohmann::json& jsonModel) {
        w = jsonModel["weights"].get<std::vector<float>>();
        b = jsonModel["bias"].get<float>();
        learning_rate = jsonModel["learning_rate"].get<float>();
        lambda = jsonModel["lambda"].get<float>();
    } */

};
