#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <random>
#include <iostream>
#include "MLPAlgo.h" // Include your MLP class header
#include <filesystem> // Requires C++17


namespace fs = std::filesystem;

void preprocessImage(const std::string& imagePath, std::vector<float>& outputVector) {
    cv::Mat img = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    img.convertTo(img, CV_32F, 1.0 / 255);

    // Resize image if necessary
   // cv::resize(img, img, cv::Size(new_width, new_height));
   
    // Flatten the image to a 1D array
    img.reshape(1, img.total()).copyTo(outputVector);
}

void loadData(std::vector<std::vector<float>>& inputs, std::vector<std::vector<float>>& targets) {
    // Assuming images are organized in directories named 'a', 'j', 'c'
    std::string baseDir = "C:/Users/sarra/source/repos/HandwrittenLetterRecognition/TrainingDataset"; // the path to your images
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
    // Implement loading of your test data similar to how you did with the training data
    // Assuming test images are organized in directories named 'a', 'j', 'c'
    std::string baseDir = "C:/Users/sarra/source/repos/HandwrittenLetterRecognition/TrainingDataset"; // Update with the path to your test images
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

int main() {
    std::vector<std::vector<float>> inputs;  // To store input data
    std::vector<std::vector<float>> targets; // To store target labels

    loadData(inputs, targets); // Load your data

    // Parameters for the MLP
    int input_size = 400 * 400; // Size of each input vector
    int hidden_size = 128;      // Number of neurons in the hidden layer
    int output_size = 3;        // Number of output neurons (3 classes)
    float learning_rate = 0.01; // Learning rate

    // Create the MLP
    MLPAlgo mlp(input_size, hidden_size, output_size, learning_rate);

    int epochs = 50; // Number of epochs for training

    // Training loop
    for (int epoch = 0; epoch < epochs; ++epoch) {
        float epoch_loss = 0.0;

        for (size_t i = 0; i < inputs.size(); ++i) {
            // Train the network with each input and target pair
            mlp.train(inputs[i], targets[i]);

            // Optionally calculate and accumulate loss here (requires implementation)
        }

        std::cout << "Epoch " << (epoch + 1) << " completed." << std::endl;
        // Optionally output average loss per epoch here
    }

    std::cout << "Training completed." << std::endl; 

    // After training, you can test the model on your test data
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

    return 0;
}