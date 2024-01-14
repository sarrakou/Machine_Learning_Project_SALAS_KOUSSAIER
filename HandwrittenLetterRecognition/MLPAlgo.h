#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <random>
#include <iostream>

class MLPAlgo
{
private:
    std::vector<std::vector<float>> weights_hidden, weights_output;
    std::vector<float> bias_hidden, bias_output;
    float learning_rate;
    int input_nodes, hidden_nodes, output_nodes;

    // Sigmoid Activation Function
    float sigmoid(float x) {
        return 1.0f / (1.0f + exp(-x));
    }

    // Derivative of Sigmoid Function
    float sigmoid_derivative(float x) {
        return x * (1.0f - x);
    }

    // Softmax Activation Function
    std::vector<float> softmax(const std::vector<float>& logits) {
        std::vector<float> exp_logits(logits.size());
        float sum_exp_logits = 0.0f;

        for (size_t i = 0; i < logits.size(); i++) {
            exp_logits[i] = exp(logits[i]);
            sum_exp_logits += exp_logits[i];
        }

        for (size_t i = 0; i < logits.size(); i++) {
            exp_logits[i] /= sum_exp_logits;
        }

        return exp_logits;
    }

    // Initialize weights and biases with random values
    void initialize() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-1, 1);

        weights_hidden.resize(hidden_nodes, std::vector<float>(input_nodes));
        weights_output.resize(output_nodes, std::vector<float>(hidden_nodes));
        bias_hidden.resize(hidden_nodes);
        bias_output.resize(output_nodes);

        for (int i = 0; i < hidden_nodes; ++i) {
            bias_hidden[i] = dis(gen);
            for (int j = 0; j < input_nodes; ++j) {
                weights_hidden[i][j] = dis(gen);
            }
        }

        for (int i = 0; i < output_nodes; ++i) {
            bias_output[i] = dis(gen);
            for (int j = 0; j < hidden_nodes; ++j) {
                weights_output[i][j] = dis(gen);
            }
        }
    }

public:
    MLPAlgo(int input, int hidden, int output, float lr) : input_nodes(input), hidden_nodes(hidden), output_nodes(output), learning_rate(lr) {
        initialize();
    }

    // Forward Propagation
    std::pair<std::vector<float>, std::vector<float>> forward(const std::vector<float>& inputs) {
        // Calculate hidden layer activations
        std::vector<float> hidden_activations(hidden_nodes, 0.0f);

        for (int i = 0; i < hidden_nodes; ++i) {
            for (int j = 0; j < input_nodes; ++j) {
                hidden_activations[i] += inputs[j] * weights_hidden[i][j];
            }
            hidden_activations[i] += bias_hidden[i];
            hidden_activations[i] = sigmoid(hidden_activations[i]);
        }

        // Calculate output layer logits
        std::vector<float> logits(output_nodes, 0.0f);
        for (int i = 0; i < output_nodes; ++i) {
            for (int j = 0; j < hidden_nodes; ++j) {
                logits[i] += hidden_activations[j] * weights_output[i][j];
            }
            logits[i] += bias_output[i];
        }

        // Apply softmax to output layer logits
        std::vector<float> outputs = softmax(logits);

        return { hidden_activations, outputs };
    }

    // Backpropagation 
    void train(const std::vector<float>& inputs, const std::vector<float>& targets) {

        // Capture both outputs and hidden_activations from the forward pass
        auto [hidden_activations, outputs] = forward(inputs);

        // Calculate output errors
        std::vector<float> output_errors(output_nodes);
        for (int i = 0; i < output_nodes; ++i) {
            output_errors[i] = targets[i] - outputs[i];
        }

        // Calculate hidden layer errors
        std::vector<float> hidden_errors(hidden_nodes, 0.0f);
        for (int i = 0; i < hidden_nodes; ++i) {
            for (int j = 0; j < output_nodes; ++j) {
                hidden_errors[i] += output_errors[j] * weights_output[j][i];
            }
        }

        // Update weights for the output layer
        for (int i = 0; i < output_nodes; ++i) {
            for (int j = 0; j < hidden_nodes; ++j) {
                weights_output[i][j] += learning_rate * output_errors[i] * sigmoid_derivative(outputs[i]) * hidden_activations[j];
            }
            bias_output[i] += learning_rate * output_errors[i] * sigmoid_derivative(outputs[i]);
        }

        // Update weights for the hidden layer
        for (int i = 0; i < hidden_nodes; ++i) {
            for (int j = 0; j < input_nodes; ++j) {
                weights_hidden[i][j] += learning_rate * hidden_errors[i] * sigmoid_derivative(hidden_activations[i]) * inputs[j];
            }
            bias_hidden[i] += learning_rate * hidden_errors[i] * sigmoid_derivative(hidden_activations[i]);
        }
    }

    // Cross-Entropy Loss Function
    float cross_entropy_loss(const std::vector<float>& outputs, const std::vector<float>& targets) {
        float loss = 0.0f;
        for (size_t i = 0; i < outputs.size(); i++) {
            // To avoid log(0) which is undefined, add a small value epsilon
            float epsilon = 1e-6;
            loss -= targets[i] * log(outputs[i] + epsilon);
        }
        return loss;
    }

    // Mean Squared Error Loss
    float mse_loss(const std::vector<float>& outputs, const std::vector<float>& targets) {
        float loss = 0.0f;
        for (size_t i = 0; i < outputs.size(); i++) {
            float error = targets[i] - outputs[i];
            loss += error * error;
        }
        return loss / outputs.size();
    }
};