#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <random>
#include <iostream>
#include "MLPAlgo.h" // Include your MLP class header
#include "LinearModel.h"
#include "RBFNetwork.h"
#include <filesystem> // Requires C++17


namespace fs = std::filesystem;

void preprocessImage(const std::string& imagePath, std::vector<float>& outputVector) {
    // Load the image in grayscale
    cv::Mat img = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);

    // Normalize the pixel values (0-1)
    img.convertTo(img, CV_32F, 1.0 / 255);

    // Resize image if necessary
   // cv::resize(img, img, cv::Size(new_width, new_height));

    // Flatten the image to a 1D array
    img.reshape(1, img.total()).copyTo(outputVector);
}

void loadData(std::vector<std::vector<float>>& inputs, std::vector<std::vector<float>>& targets) {
    // Assuming images are organized in directories named 'a', 'j', 'c'
    std::string baseDir = "D:/Adriana/ESGI/Cursos/Machine_Learning/proyecto/Machine_Learning_Project_SALAS_KOUSSAIER/TrainingDataset"; // the path to images
    std::vector<std::vector<float>> labels = { {1, 0, 0}, {0, 1, 0}, {0, 0, 1} }; // One-hot encoding for 'a', 'j', 'c'
    std::vector<std::string> folders = { "a", "j", "c" };

    for (size_t i = 0; i < folders.size(); ++i) {
        std::string folderPath = baseDir + "/" + folders[i];
        for (const auto& entry : fs::directory_iterator(folderPath)) {
            std::vector<float> imgData;
            preprocessImage(entry.path().string(), imgData);
            inputs.push_back(imgData);
            targets.push_back(labels[i]);
        }
    }
}


void loadTestData(std::vector<std::vector<float>>& testInputs, std::vector<std::vector<float>>& testTargets) {
    // Assuming test images are organized in directories named 'a', 'j', 'c'
    std::string baseDir = "D:/Adriana/ESGI/Cursos/Machine_Learning/proyecto/Machine_Learning_Project_SALAS_KOUSSAIER/TestingDataset"; // the path to test images
    std::vector<std::vector<float>> labels = { {1, 0, 0}, {0, 1, 0}, {0, 0, 1} }; // One-hot encoding for 'a', 'j', 'c'
    std::vector<std::string> folders = { "a", "j", "c" };

    for (size_t i = 0; i < folders.size(); ++i) {
        std::string folderPath = baseDir + "/" + folders[i];
        for (const auto& entry : fs::directory_iterator(folderPath)) {
            std::vector<float> imgData;
            preprocessImage(entry.path().string(), imgData);
            testInputs.push_back(imgData);
            testTargets.push_back(labels[i]);
        }
    }
}

void trainAndEvaluateMLP() {
    std::vector<std::vector<float>> inputs;  // To store input data
    std::vector<std::vector<float>> targets; // To store target labels

    loadData(inputs, targets); // Load data

    // Parameters for the MLP
    int input_size = 400 * 400; // Size of each input vector
    int hidden_size = 128;      // Number of neurons in the hidden layer
    int output_size = 3;        // Number of output neurons (3 classes)
    float learning_rate = 0.005; // Learning rate

    // Create the MLP
    MLPAlgo mlp(input_size, hidden_size, output_size, learning_rate);

    int epochs = 50; // Number of epochs for training

    // Training loop
    for (int epoch = 0; epoch < epochs; ++epoch) {
        float epoch_loss = 0.0f;
        int correctPredictions = 0;

        for (size_t i = 0; i < inputs.size(); ++i) {
            // Forward pass and training
            auto outputs = mlp.forward(inputs[i]).second;
            mlp.train(inputs[i], targets[i]);

            // Compute cross-entropy loss
            epoch_loss += mlp.cross_entropy_loss(outputs, targets[i]);

            // Accuracy calculation
            int predictedClass = std::distance(outputs.begin(), std::max_element(outputs.begin(), outputs.end()));
            int actualClass = std::distance(targets[i].begin(), std::max_element(targets[i].begin(), targets[i].end()));
            if (predictedClass == actualClass) {
                correctPredictions++;
            }
        }

        // Calculate average loss for the epoch
        epoch_loss /= inputs.size();

        // Calculate accuracy for the epoch
        float accuracy = static_cast<float>(correctPredictions) / static_cast<float>(inputs.size());

        std::cout << "Epoch " << (epoch + 1) << " - Loss: " << epoch_loss << ", Accuracy: " << accuracy * 100.0f << "%" << std::endl;
    }

    std::cout << "Training completed." << std::endl;

    // After training, test the model on your test data
    // and evaluate its performance 

    std::vector<std::vector<float>> testInputs;
    std::vector<std::vector<float>> testTargets;
    loadTestData(testInputs, testTargets); // Load test data

    int correctPredictions = 0;

    for (size_t i = 0; i < testInputs.size(); ++i) {
        std::vector<float> output = mlp.forward(testInputs[i]).second; // Get the output from the model

        // Determine the predicted class (index of max value in output vector)
        int predictedClass = std::distance(output.begin(), std::max_element(output.begin(), output.end()));

        // Determine the actual class
        int actualClass = std::distance(testTargets[i].begin(), std::max_element(testTargets[i].begin(), testTargets[i].end()));

        if (predictedClass == actualClass) {
            correctPredictions++;
        }
    }

    float accuracy = static_cast<float>(correctPredictions) / static_cast<float>(testInputs.size());
    std::cout << "Accuracy: " << accuracy * 100.0f << "%" << std::endl;
}

void trainAndEvaluateLinearModel() {
    
    // Training and testing data for the linear model
    std::vector<std::vector<float>> trainInputs;  //  training data for the linear model
    std::vector<std::vector<float>> trainTargets;  //  training labels for the linear model
    std::vector<std::vector<float>> testInputs;   //  testing data for the linear model
    std::vector<std::vector<float>> testTargets;  //  testing labels for the linear model

    loadData(trainInputs, trainTargets);
    loadTestData(testInputs, testTargets);

    int inputSize = 400 * 400;
    float learningRate = 0.01;
    int epochs = 50;

    LinearModel linearModel(inputSize, learningRate);

    std::vector<float> flatTrainTargets;
    for (const auto& targetVector : trainTargets) {
        if (!targetVector.empty()) {
            flatTrainTargets.push_back(targetVector[0]);
        }
        else {
            std::cerr << "Error: Empty target vector encountered." << std::endl;

        }
    }
    // Training the linear model
    linearModel.train(trainInputs, flatTrainTargets, epochs);

    // Evaluating the linear model
    int correctPredictionsLinear = 0;

    for (size_t i = 0; i < testInputs.size(); ++i) {
        // Get the prediction from the linear model
        float linearPrediction = linearModel.predict(testInputs[i]);

        // Assuming labels are continuous values, if greater than 0.5, predict positive class
        int predictedClassLinear = (linearPrediction > 0.5) ? 1 : 0;

        // Determine the actual class
        int actualClass = static_cast<int>(flatTrainTargets[i]); 

        if (predictedClassLinear == actualClass) {
            correctPredictionsLinear++;
        }
    }

    float accuracyLinear = static_cast<float>(correctPredictionsLinear) / static_cast<float>(testInputs.size());
    std::cout << "Accuracy (Linear Model): " << accuracyLinear * 100.0f << "%" << std::endl;
}


void predictSingleImage(const std::string& imagePath, RBFNetwork& rbfNetwork, std::vector<std::vector<float>>& trainingTargets) {

    // Vector to store the preprocessed image
    std::vector<float> inputVector;

    // Load and preprocess the image
    preprocessImage(imagePath, inputVector);

    std::vector<float> output = rbfNetwork.processInput(inputVector, trainingTargets);

    // Identify the predicted class (index of the maximum value)
    int predictedClass = std::distance(output.begin(), std::max_element(output.begin(), output.end()));

    std::vector<char> classLetters = { 'a', 'j', 'c' };

    std::cout << "The predicted letter is: " << classLetters[predictedClass] << std::endl;
}

void trainAndEvaluateRBFNetwork() {
    std::vector<std::vector<float>> inputs, targets;
    std::vector<std::vector<float>> testInputs, testTargets;

    loadData(inputs, targets);

    loadTestData(testInputs, testTargets);

    // Number of centers and beta 
    int numCenters = 15;
    float beta = 0.01;

    // Create an instance of RBFNetwork
    RBFNetwork rbfNetwork(inputs, targets, numCenters, beta);

    rbfNetwork.train(inputs, targets);

    float accuracy = rbfNetwork.test(testInputs, testTargets);

    std::cout << "Accuracy: " << accuracy * 100 << "%" << std::endl;

    //////////////////////////////////////////////////////
        

    predictSingleImage("D:/Adriana/ESGI/Cursos/Machine_Learning/Machine_Learning_Project_SALAS_KOUSSAIER/TrainingDataset/c/24.png", rbfNetwork, targets);



}



int main() {

    // Menu to choose the model
    std::cout << "Choose a model to train and evaluate:" << std::endl;
    std::cout << "1. MLP" << std::endl;
    std::cout << "2. Linear Model" << std::endl;
    std::cout << "3. RBFNetwork" << std::endl;

    int choice;
    std::cin >> choice;

    switch (choice) {
    case 1:
        trainAndEvaluateMLP();
        break;

    case 2:
        trainAndEvaluateLinearModel();
        break;
    case 3:
        trainAndEvaluateRBFNetwork();
        break;

    default:
        std::cout << "Invalid choice." << std::endl;
        break;
    }

    return 0;
}
