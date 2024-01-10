#include "LinearModel.h"


float LinearModel::predict(const std::vector<float>& inputs) {
    // Realiza la predicción lineal: y = w1*x1 + w2*x2 + ... + wn*xn + b
    float result = bias;
    for (size_t i = 0; i < inputs.size(); ++i) {
        result += weights[i] * inputs[i];
    }
    // Aplicar función de activación (sigmoid en este caso)
    return 1.0f / (1.0f + std::exp(-result));
}


void LinearModel::train(const std::vector<std::vector<float>>& input_data, const std::vector<float>& targets, int epochs) {
    // Asegurémonos de que las dimensiones sean consistentes
    if (input_data.empty() || input_data.size() != targets.size() || input_data[0].size() != weights.size()) {
        std::cerr << "Error: Dimensiones inconsistentes en datos de entrada o pesos." << std::endl;
        return;
    }

    // Entrenamiento del modelo lineal utilizando descenso de gradiente
    for (int epoch = 0; epoch < epochs; ++epoch) {
        float epoch_loss = 0.0;

        for (size_t i = 0; i < input_data.size(); ++i) {
            // Realiza la predicción
            float prediction = predict(input_data[i]);

            // Calcula el error
            float error = prediction - targets[i];

            // Actualiza los pesos y el sesgo utilizando el descenso de gradiente
            for (size_t j = 0; j < weights.size(); ++j) {
                weights[j] -= learning_rate * error * input_data[i][j];
            }
            bias -= learning_rate * error;

            // Acumula la pérdida (opcional, para seguimiento)
            epoch_loss += 0.5 * error * error;
        }

        std::cout << "Epoch " << (epoch + 1) << ", Loss: " << epoch_loss << std::endl;
    }
}