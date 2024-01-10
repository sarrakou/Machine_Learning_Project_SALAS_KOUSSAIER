#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <filesystem>

class LinearModel {
private:
    std::vector<float> weights;
    float bias;
    float learning_rate;

    
public:
    LinearModel(int input_size, float lr) : learning_rate(lr) {
        // Inicialización de pesos con valores aleatorios
        weights.resize(input_size);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-1, 1);
        for (float& weight : weights) {
            weight = dis(gen);
        }
        // Inicialización de sesgo con valor aleatorio
        bias = dis(gen);
    }

    float predict(const std::vector<float>& inputs);

    void train(const std::vector<std::vector<float>>& input_data, const std::vector<float>& targets, int epochs);



};
