<<<<<<< HEAD

=======
>>>>>>> main
#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <filesystem>
<<<<<<< HEAD
#include <numeric>  
#include <algorithm>  // For std::shuffle
#include <vector>
#include <random>    // For std::mt19937 and std::random_device

class LinearModel
{
private:
    float learning_rate;
    std::vector<std::vector<float>> weights; // Each class has its own set of weights
    std::vector<float> biases; // Separate bias for each class

public:
    // Softmax function
    std::vector<float> softmax(const std::vector<float>& logits) {
        std::vector<float> exp_logits(logits.size());
        float sum_exp_logits = 0.0f;

        for (size_t i = 0; i < logits.size(); i++) {
            exp_logits[i] = std::exp(logits[i]);
            sum_exp_logits += exp_logits[i];
        }

        for (size_t i = 0; i < logits.size(); i++) {
            exp_logits[i] /= sum_exp_logits;
        }

        return exp_logits;
    }

    // Categorical Cross-Entropy Loss
    float categoricalCrossEntropyLoss(const std::vector<float>& probs, const std::vector<float>& targets) {
        float loss = 0.0f;
        for (size_t i = 0; i < probs.size(); i++) {
            loss -= targets[i] * std::log(probs[i] + 1e-6); // Adding epsilon for numerical stability
        }
        return loss;
    }

    LinearModel(int input_size, int num_classes, float lr) : learning_rate(lr) {
        weights.resize(num_classes, std::vector<float>(input_size));
        biases.resize(num_classes);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-1, 1);

        // Initialize weights and biases for each class
        for (int i = 0; i < num_classes; ++i) {
            for (float& weight : weights[i]) {
                weight = dis(gen);
            }
            biases[i] = dis(gen);
        }
    }

    std::vector<float> predict(const std::vector<float>& inputs) {
        std::vector<float> logits(biases.size());
        for (size_t i = 0; i < biases.size(); ++i) {
            logits[i] = biases[i];
            for (size_t j = 0; j < inputs.size(); ++j) {
                logits[i] += weights[i][j] * inputs[j];
            }
        }
        return softmax(logits);
    }

    void train(const std::vector<std::vector<float>>& input_data,
        const std::vector<std::vector<float>>& targets,
        int epochs) {
        // Check dimensions
        if (input_data.empty() || input_data.size() != targets.size() ||
            input_data[0].size() != weights[0].size()) {
            std::cerr << "Error: Dimensiones inconsistentes en datos de entrada o pesos." << std::endl;
            return;
        }

        for (int epoch = 0; epoch < epochs; ++epoch) {
            float epoch_loss = 0.0;
            float lambda = 0.01; // Regularization parameter

            // Optional: Shuffle the data indices for stochastic or mini-batch gradient descent
            std::vector<size_t> indices(input_data.size());
            iota(indices.begin(), indices.end(), 0);

            for (size_t i = 0; i < input_data.size(); ++i) {
                size_t idx = indices[i]; // for shuffled index
            
                // Forward pass to get predictions
                std::vector<float> predictions = predict(input_data[i]);

                // Calculate error for each class and update weights and biases
                for (size_t classIdx = 0; classIdx < weights.size(); ++classIdx) {
                    // Error calculation for categorical cross-entropy with softmax
                    float error = -(targets[i][classIdx] - predictions[classIdx]);
                    for (size_t j = 0; j < input_data[i].size(); ++j) {
                        // Regularization term for L2 regularization
                        float reg_term = lambda * weights[classIdx][j];
                        weights[classIdx][j] -= learning_rate * (error * input_data[i][j] + reg_term);
                    }
                    biases[classIdx] -= learning_rate * error;
                }

                // Calculate and accumulate the loss using categorical cross-entropy
                epoch_loss += categoricalCrossEntropyLoss(predictions, targets[i]);
            }

            epoch_loss /= input_data.size(); // Average loss over the epoch
            std::cout << "Epoch " << (epoch + 1) << ", Loss: " << epoch_loss << std::endl;
        }
    }

};

=======

class LinearModel {
private:
    std::vector<float> weights;
    float bias;
    float learning_rate;

    
public:

    /*//float sigmoid(float x) const;
    //LinearModel(int input_size, float lr) : learning_rate(lr) {
    //    // Inicialización de pesos con valores aleatorios
    //    weights.resize(input_size);
    //    std::random_device rd;
    //    std::mt19937 gen(rd()); //numero aleatorio
    //    std::uniform_real_distribution<> dis(-1, 1); // distribucion uniforme para asignar valores entre -1 y 1
    //    for (float& weight : weights) {
    //        weight = dis(gen);
    //    }
    //    // Inicialización de sesgo con valor aleatorio
    //    bias = dis(gen);
    //}
    //float crossEntropyLoss(const std::vector<float>& predictions, const std::vector<float>& targets);
    //std::vector<float> applySoftmax(const std::vector<float>& logits) const;
    //std::vector<float> predict(const std::vector<float>& inputs) const;
    ////float predict(const std::vector<float>& inputs);

    //void train(const std::vector<std::vector<float>>& input_data, const std::vector<std::vector<float>>& targets, int epochs);*/


    LinearModel(int input_size, float lr) : learning_rate(lr) {
        // Weight initialization with random values
        weights.resize(input_size);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-1, 1);
        for (float& weight : weights) {
            weight = dis(gen);
        }
        // Bias initialization with a random value
        bias = dis(gen);
    }

    float predict(const std::vector<float>& inputs);

    void train(const std::vector<std::vector<float>>& input_data, const std::vector<float>& targets, int epochs);





};
>>>>>>> main
