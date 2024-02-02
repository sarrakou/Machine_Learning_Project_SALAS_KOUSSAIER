<<<<<<< HEAD
#include "LinearModel.h"

=======
/*#include "LinearModel.h"

float LinearModel::sigmoid(float x) const {
    return 1.0f / (1.0f + std::exp(-x));
}

// En LinearModel.cpp
float LinearModel::crossEntropyLoss(const std::vector<float>& predictions, const std::vector<float>& targets) {
   

    float loss = 0.0f;

    for (size_t j = 0; j < targets.size(); ++j) {
        for (size_t i = 0; i < predictions.size(); ++i) {
            loss += targets[j] * std::log(predictions[i] + 1e-10);
        }
    }

    return -loss / static_cast<float>(predictions.size());  // Pérdida promedio
}

std::vector<float> LinearModel::applySoftmax(const std::vector<float>& logits) const {
    std::vector<float> probabilities(logits.size());
    float expSum = 0.0f;

    // Exponenciación y suma
    for (size_t i = 0; i < logits.size(); ++i) {
        probabilities[i] = std::exp(logits[i]);
        expSum += probabilities[i];
    }

    // Normalización
    for (float& prob : probabilities) {
        prob /= expSum;
    }

    return probabilities;
}



std::vector<float> LinearModel::predict(const std::vector<float>& inputs) const {
    // Realiza la operación lineal y obtén logits
    std::vector<float> logits(weights.size());
    for (size_t i = 0; i < weights.size(); ++i) {
        logits[i] = bias + weights[i] * inputs[i];
    }

    return logits;
}



void LinearModel::train(const std::vector<std::vector<float>>& input_data, const std::vector<std::vector<float>>& targets, int epochs) {
    // Asegurémonos de que las dimensiones sean consistentes
    if (input_data.empty() || input_data.size() != targets.size() || input_data[0].size() != weights.size()) {
        std::cerr << "Error: Dimensiones inconsistentes en datos de entrada o pesos." << std::endl;
        return;
    }

    // Entrenamiento del modelo lineal utilizando descenso de gradiente
    for (int epoch = 0; epoch < epochs; ++epoch) {
        float epoch_loss = 0.0;

        for (size_t i = 0; i < input_data.size(); ++i) {
            std::vector<float> predictions = predict(input_data[i]);

            float error = crossEntropyLoss(predictions, targets[i]);

            for (size_t j = 0; j < weights.size(); ++j) {
                weights[j] -= learning_rate * error * input_data[i][j];
            }
            bias -= learning_rate * error;

            epoch_loss += error;
        }

        std::cout << "Epoch " << (epoch + 1) << ", Loss: " << epoch_loss << std::endl;
    }
}*/





#include "LinearModel.h"


float LinearModel::predict(const std::vector<float>& inputs) {
    // Perform linear prediction: y = w1x1 + w2x2 + ... + wn*xn + b
    float result = bias;
    for (size_t i = 0; i < inputs.size(); ++i) {
        result += weights[i] * inputs[i];
    }
    // Apply activation function (sigmoid in this case)
    return 1.0f / (1.0f + std::exp(-result));
}


void LinearModel::train(const std::vector<std::vector<float>>& input_data, const std::vector<float>& targets, int epochs) {
    // Make sure the dimensions are consistent
    if (input_data.empty() || input_data.size() != targets.size() || input_data[0].size() != weights.size()) {
        std::cerr << "Error: Dimensiones inconsistentes en datos de entrada o pesos." << std::endl;
        return;
    }

    // Training the linear model using gradient descent
    for (int epoch = 0; epoch < epochs; ++epoch) {
        float epoch_loss = 0.0;

        for (size_t i = 0; i < input_data.size(); ++i) {
            // Make the prediction
            float prediction = predict(input_data[i]);

            // Calculate the error
            float error = prediction - targets[i];

            //Update weights and bias using gradient descent
            for (size_t j = 0; j < weights.size(); ++j) {
                weights[j] -= learning_rate * error * input_data[i][j];
            }
            bias -= learning_rate * error;

            //Accumulate the loss (optional, for tracking)
            epoch_loss += 0.5 * error * error;
        }

        std::cout << "Epoch " << (epoch + 1) << ", Loss: " << epoch_loss << std::endl;
    }
}
>>>>>>> main
