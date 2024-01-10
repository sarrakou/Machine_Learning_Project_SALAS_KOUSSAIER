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

        // Calculate output layer activations
        std::vector<float> outputs(output_nodes, 0.0f);
        for (int i = 0; i < output_nodes; ++i) {
            for (int j = 0; j < hidden_nodes; ++j) {
                outputs[i] += hidden_activations[j] * weights_output[i][j];
            }
            outputs[i] += bias_output[i];
            outputs[i] = sigmoid(outputs[i]);
        }

        return { hidden_activations, outputs };
    }

    // Backpropagation (to be implemented)
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
};
