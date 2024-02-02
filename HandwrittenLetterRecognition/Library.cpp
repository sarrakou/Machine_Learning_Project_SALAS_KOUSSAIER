#include <opencv2/opencv.hpp>
#include "Library.h"
#include <vector>
#include <cmath>
#include <random>
#include <iostream>
#include "MLPAlgo.h" 
#include "LinearModel.h"
#include "SimpleSVM.h"
#include "RBFNetwork.h"
#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>
#include <filesystem> // Requires C++17
#include <nlohmann/json.hpp>


namespace fs = std::filesystem;
using namespace cv;
using namespace cv::ml;

void preprocessImage(const std::string& imagePath, std::vector<float>& outputVector) {
    // Load the image in grayscale
    cv::Mat img = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);

    // Normalize the pixel values (0-1)
    img.convertTo(img, CV_32F, 1.0 / 255);

    // Resize image
   // cv::resize(img, img, cv::Size(new_width, new_height));
   
    // Flatten the image to a 1D array
    img.reshape(1, img.total()).copyTo(outputVector);
}

void loadData(std::vector<std::vector<float>>& inputs, std::vector<std::vector<float>>& targets) {
    // Assuming images are organized in directories named 'a', 'j', 'c'
    std::string baseDir = "C:/Users/sarra/source/repos/HandwrittenLetterRecognition/TrainingDataset";
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
    // Implement loading of  test data similar to how we did with the training data
    //  test images are organized in directories named 'a', 'j', 'c'
    std::string baseDir = "C:/Users/sarra/source/repos/HandwrittenLetterRecognition/TestingDataset";
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

// Parameters for the MLP
int input_size = 400 * 400; // Size of each input vector
int hidden_size = 128;      // Number of neurons in the hidden layer
int output_size = 3;        // Number of output neurons (3 classes)
float learning_rate = 0.005; // Learning rate

// Create the MLP
MLPAlgo mlp(input_size, hidden_size, output_size, learning_rate);

void trainAndEvaluateMLP() {
    std::vector<std::vector<float>> inputs;  // To store input data
    std::vector<std::vector<float>> targets; // To store target labels

    loadData(inputs, targets); // Load data

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


int inputSize = 400 * 400;  // Number of input features
int numClasses = 3;         // Number of classes (e.g., 'a', 'j', 'c')
float learningRate = 0.005; // Learning rate

LinearModel linearModel(inputSize, numClasses, learningRate);

// Función para entrenar y evaluar el modelo lineal
void trainAndEvaluateLinearModel() {
    //// Datos de entrenamiento y prueba para el modelo lineal
    std::vector<std::vector<float>> trainInputs;  // Tus datos de entrenamiento para el modelo lineal
    std::vector<std::vector<float>> trainTargets;  // Tus etiquetas de entrenamiento para el modelo lineal
    std::vector<std::vector<float>> testInputs;   // Tus datos de prueba para el modelo lineal
    std::vector<std::vector<float>> testTargets;  // Tus etiquetas de prueba para el modelo lineal

    // Llamada a la función para cargar los datos
    loadData(trainInputs, trainTargets);
    loadTestData(testInputs, testTargets);

    int epochs = 50;
    // Entrenamiento del modelo lineal
    linearModel.train(trainInputs, trainTargets, epochs);

    // Evaluación del modelo lineal
    int correctPredictionsLinear = 0;

    for (size_t i = 0; i < testInputs.size(); ++i) {
        // Obtener la predicción del modelo lineal
        std::vector<float> linearPrediction = linearModel.predict(testInputs[i]);

        // Determine the predicted class (index of max probability)
        int predictedClassLinear = std::distance(linearPrediction.begin(), std::max_element(linearPrediction.begin(), linearPrediction.end()));

        // Determine la clase real (index of 1 in one-hot encoded vector)
        int actualClass = std::distance(testTargets[i].begin(), std::max_element(testTargets[i].begin(), testTargets[i].end()));

        if (predictedClassLinear == actualClass) {
            correctPredictionsLinear++;
        }
    }

    float accuracyLinear = static_cast<float>(correctPredictionsLinear) / static_cast<float>(testInputs.size());
    std::cout << "Accuracy (Linear Model): " << accuracyLinear * 100.0f << "%" << std::endl;
}

void prepareDataForClass(const std::vector<std::vector<float>>& inputs,
    const std::vector<std::vector<float>>& originalLabels,
    int classIndex,
    std::vector<std::vector<float>>& classInputs,
    std::vector<int>& classLabels) {
    classInputs = inputs;  // Copy all inputs
    classLabels.resize(originalLabels.size());

    for (size_t i = 0; i < originalLabels.size(); ++i) {
        // Assuming originalLabels are one-hot encoded, classLabels[i] is 1 if the class index matches, else -1
        classLabels[i] = (originalLabels[i][classIndex] == 1.0f) ? 1 : -1;
    }
}


extern "C" {
    __declspec(dllexport) int predictClass(const std::vector<SimpleSVM>& svms, const std::vector<float>& input) {
        float maxScore = std::numeric_limits<float>::lowest();
        int predictedClass = -1;

        for (size_t i = 0; i < svms.size(); ++i) {
            float score = svms[i].decisionFunction(input);  // Assume decisionFunction returns the distance from the hyperplane
            if (score > maxScore) {
                maxScore = score;
                predictedClass = i;  // Class index with the highest score
            }
        }

        return predictedClass;
    }
}

/* // Function to save all SVM models to a JSON file
void saveModelsToJson(const std::vector<SimpleSVM>& svms, const std::string& filename) {
    nlohmann::json jsonModels;

    for (const auto& svm : svms) {
        nlohmann::json jsonModel = svm.serializeToJson();
        jsonModels.push_back(jsonModel);
    }

    std::ofstream file(filename);

    if (file.is_open()) {
        file << jsonModels.dump(4);  // Pretty-print with indentation
        file.close();
        std::cout << "Models saved to " << filename << std::endl;
    }
    else {
        std::cerr << "Unable to open file for saving models: " << filename << std::endl;
    }
}

// Function to load all SVM models from a JSON file
bool loadModelsFromJson(std::vector<SimpleSVM>& svms, const std::string& filename) {
    std::ifstream file(filename);

    if (file.is_open()) {
        nlohmann::json jsonModels;
        file >> jsonModels;

        svms.clear();  // Clear existing models

        for (const auto& jsonModel : jsonModels) {
            SimpleSVM svm;
            svm.deserializeFromJson(jsonModel);
            svms.push_back(svm);
        }

        file.close();
        std::cout << "Models loaded from " << filename << std::endl;
        return true;
    }
    else {
        std::cerr << "Unable to open file for loading models: " << filename << std::endl;
        return false;
    }
} */

static std::vector<SimpleSVM> svms; // One SVM for each class

void trainAndEvaluateSVM() {
    std::vector<std::vector<float>> inputs;  // To store input data
    std::vector<std::vector<float>> originalLabels; // Multi-class labels

    loadData(inputs, originalLabels); // Load data

    int inputSize = 400 * 400; // Size of each input vector
    float learningRate = 0.005;
    int epochs = 50;
    int numClasses = 3; // Assuming 3 classes for 'a', 'j', and 'c'
    float lambda = 0.1; // Regularization parameter

    svms.clear(); // Clear existing models if re-training
    svms.resize(numClasses, SimpleSVM(inputSize, learningRate, lambda));


    // Train SVM for each class in One-vs-Rest manner
    for (int classIdx = 0; classIdx < numClasses; ++classIdx) {
        std::vector<std::vector<float>> classInputs;
        std::vector<int> classLabels;
        prepareDataForClass(inputs, originalLabels, classIdx, classInputs, classLabels);

        for (int epoch = 0; epoch < epochs; ++epoch) {
            float epoch_loss = 0.0f;
            int correctPredictions = 0;

            for (size_t i = 0; i < classInputs.size(); ++i) {
                float output = svms[classIdx].decisionFunction(classInputs[i]);
                int predicted = output >= 0 ? 1 : -1;
                correctPredictions += (predicted == classLabels[i]) ? 1 : 0;

                // Hinge loss calculation
                epoch_loss += std::max(0.0f, 1 - classLabels[i] * output);

                // Update the SVM
                svms[classIdx].train(classInputs[i], classLabels[i]);
            }

            float accuracy = static_cast<float>(correctPredictions) / classInputs.size();
            epoch_loss /= classInputs.size(); // Average loss

            std::cout << "SVM for class " << classIdx << ", Epoch " << (epoch + 1)
                << ", Loss: " << epoch_loss
                << ", Accuracy: " << accuracy * 100.0f << "%" << std::endl;
        }
    }

    // Load test data
    std::vector<std::vector<float>> testInputs;
    std::vector<std::vector<float>> testLabels; // Multi-class test labels
    loadTestData(testInputs, testLabels);

    int correctPredictions = 0;

    // Evaluate the model
    for (size_t i = 0; i < testInputs.size(); ++i) {
        int predictedClass = predictClass(svms, testInputs[i]);

        // Determine the actual class from the one-hot encoded labels
        int actualClass = std::distance(testLabels[i].begin(), std::max_element(testLabels[i].begin(), testLabels[i].end()));

        if (predictedClass == actualClass) {
            correctPredictions++;
        }
    }

    float accuracy = static_cast<float>(correctPredictions) / static_cast<float>(testInputs.size());
    std::cout << "Accuracy (SVM): " << accuracy * 100.0f << "%" << std::endl;

}

RBFNetwork rbfNetwork;

void trainAndEvaluateRBFNetwork() {
    std::vector<std::vector<float>> inputs, targets;
    std::vector<std::vector<float>> testInputs, testTargets;

    loadData(inputs, targets);

    loadTestData(testInputs, testTargets);

    // Number of centers and beta 
    int numCenters = 15;
    float beta = 0.1;

    // Create an instance of RBFNetwork
    rbfNetwork.setup(inputs, targets, numCenters, beta);

    rbfNetwork.train(inputs, targets);

    float accuracy = rbfNetwork.test(testInputs, testTargets);

    std::cout << "Accuracy: " << accuracy * 100 << "%" << std::endl;
}

extern "C" {
    __declspec(dllexport) int PredictValueUsingSVM(const float* inputArray, int arrayLength) {
        std::vector<float> inputVector(inputArray, inputArray + arrayLength);
        trainAndEvaluateSVM();

        float maxScore = std::numeric_limits<float>::lowest();
        int predictedClass = -1; // Default to an invalid class

        for (int i = 0; i < svms.size(); ++i) {
            float score = svms[i].decisionFunction(inputVector);
            if (score > maxScore) {
                maxScore = score;
                predictedClass = i;
            }
        }

        return predictedClass;
    }

    __declspec(dllexport) int PredictValueUsingLinearModel(const float* inputArray, int arrayLength) {
        std::vector<float> inputVector(inputArray, inputArray + arrayLength);

        // Ensure the linearModel has been initialized and trained
        std::vector<float> predictionScores = linearModel.predict(inputVector);

        // Identify the class with the highest score
        int predictedClass = std::distance(predictionScores.begin(), std::max_element(predictionScores.begin(), predictionScores.end()));


        return predictedClass;
    }

    __declspec(dllexport) int PredictValueUsingMLP(const float* inputArray, int arrayLength) {
        // Convert the input array to a vector<float>
        std::vector<float> inputVector(inputArray, inputArray + arrayLength);

        // Use the MLP model to predict the output for the given input
        auto predictionScores = mlp.forward(inputVector).second;

        // Find the index of the maximum score in the prediction scores, which corresponds to the predicted class
        int predictedClass = std::distance(predictionScores.begin(), std::max_element(predictionScores.begin(), predictionScores.end()));

        return predictedClass;
    }

    __declspec(dllexport) int PredictValueUsingRBF(const float* inputArray, int arrayLength) {
        // Convert the input array to a vector<float>
        std::vector<float> inputVector(inputArray, inputArray + arrayLength);

        // Use the RBF network to predict the output for the given input
        auto predictionScores = rbfNetwork.predict(inputVector);

        // Find the index of the maximum score in the prediction scores, which corresponds to the predicted class
        int predictedClass = std::distance(predictionScores.begin(), std::max_element(predictionScores.begin(), predictionScores.end()));

        return predictedClass;
    }

}


void testSimpleLinearSVM() {
    // Simple linear dataset
    std::vector<std::vector<float>> X = {
        {1.0, 1.0},
        {2.0, 3.0},
        {3.0, 3.0}
    };
    std::vector<int> Y = { 1, -1, -1 };

    // Parameters for the SVM
    int inputSize = 2; // Number of features in the dataset
    float learningRate = 0.005;
    float lambda = 0.1; // Regularization parameter
    int epochs = 100; // Number of epochs for training

    // Create and train the SVM
    SimpleSVM svm(inputSize, learningRate, lambda);
    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (size_t i = 0; i < X.size(); ++i) {
            svm.train(X[i], Y[i]);
        }
    }

    // Test the trained SVM (optional, for demonstration)
    for (const auto& x : X) {
        float prediction = svm.decisionFunction(x);
        std::cout << "Prediction for (" << x[0] << ", " << x[1] << "): "
            << (prediction >= 0 ? 1 : -1) << std::endl;
    }

    // Evaluate the trained model
    int correctPredictions = 0;
    for (size_t i = 0; i < X.size(); ++i) {
        float output = svm.decisionFunction(X[i]);
        int predicted = output >= 0 ? 1 : -1;
        if (predicted == Y[i]) {
            correctPredictions++;
        }
    }

    float accuracy = static_cast<float>(correctPredictions) / X.size();
    std::cout << "Simple Linear SVM Test Accuracy: " << accuracy * 100.0f << "%" << std::endl;
}



void testSimple3DSVM() {
    // Simple 3D dataset
    std::vector<std::vector<float>> X = {
        {1.0, 1.0, 2.0},   // Point 1
        {2.0, 2.0, 3.0},   // Point 2
        {3.0, 1.0, 2.5}    // Point 3
    };
    std::vector<int> Y = { 1, -1, 1 };  // Binary labels for classification

    // Parameters for the SVM
    int inputSize = 3; // Number of features (3D points)
    float learningRate = 0.005;
    float lambda = 0.1; // Regularization parameter
    int epochs = 100;

    // Create and train the SVM
    SimpleSVM svm(inputSize, learningRate, lambda);
    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (size_t i = 0; i < X.size(); ++i) {
            svm.train(X[i], Y[i]);
        }
    }

    // Test the trained SVM (optional, for demonstration)
    for (const auto& x : X) {
        float prediction = svm.decisionFunction(x);
        std::cout << "Prediction for (" << x[0] << ", " << x[1] << ", " << x[2] << "): "
            << (prediction >= 0 ? 1 : -1) << std::endl;
    }

    // Evaluate the trained model
    int correctPredictions = 0;
    for (size_t i = 0; i < X.size(); ++i) {
        float output = svm.decisionFunction(X[i]);
        int predicted = output >= 0 ? 1 : -1;
        if (predicted == Y[i]) {
            correctPredictions++;
        }
    }

    float accuracy = static_cast<float>(correctPredictions) / X.size();
    std::cout << "Simple 3D SVM Test Accuracy: " << accuracy * 100.0f << "%" << std::endl;
}

int main() {
    trainAndEvaluateSVM();
    // Menú para elegir el modelo
    /* std::cout << "Choose a model to train and evaluate:" << std::endl;
    std::cout << "1. MLP" << std::endl;
    std::cout << "2. Linear Model" << std::endl;
    std::cout << "3. SVM" << std::endl;
    std::cout << "4. test" << std::endl;

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
        trainAndEvaluateSVM();
        break;
    case 4:
        //testSimpleLinearSVM();
        testSimple3DSVM();
        break;

    default:
        std::cout << "Invalid choice." << std::endl;
        break;
    } */

    return 0; 
}