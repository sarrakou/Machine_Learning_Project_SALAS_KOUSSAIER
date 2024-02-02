using System;
using System.Runtime.InteropServices;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using UnityEditor;
using TMPro;

public class Prediction : MonoBehaviour
{
    [DllImport("HandwrittenLetterRecognition")] 
    private static extern int PredictValueUsingSVM(float[] inputData, int length);

    [DllImport("HandwrittenLetterRecognition")] 
    private static extern int PredictValueUsingLinearModel(float[] inputData, int arrayLength);

    [DllImport("HandwrittenLetterRecognition")] 
    private static extern int PredictValueUsingMLP(float[] inputData, int arrayLength);

    [DllImport("HandwrittenLetterRecognition")] 
    private static extern int PredictValueUsingRBF(float[] inputData, int arrayLength);

    [SerializeField] private Image inputImage; 
    [SerializeField] private TMP_Text resultText; 
    [SerializeField] private TMP_Dropdown ModelsDropdown; 

    // Method to be called when you want to predict the class of the image
    public void PredictImageClass()
    {
        Texture2D texture = inputImage.sprite.texture;
        if (texture)
        {
            if (ModelsDropdown.options[ModelsDropdown.value].text == "SVM")
            {
                float[] imageData = PreprocessImage(texture);
                int predictedClass = PredictValueUsingSVM(imageData, imageData.Length);
                if (predictedClass == 0)
                {
                    resultText.text = "Predicted Class: a ";
                    resultText.color = Color.black;
                } else if (predictedClass == 1)
                {
                    resultText.text = "Predicted Class: j ";
                    resultText.color = Color.black;
                } else if (predictedClass == 2)
                {
                    resultText.text = "Predicted Class: c ";
                    resultText.color = Color.black;
                } else
                {
                    resultText.text = "Error!";
                    resultText.color = Color.red;
                }
            } else if (ModelsDropdown.options[ModelsDropdown.value].text == "Linear Model")
            {
               float[] imageData = PreprocessImage(texture);
                int predictedClass = PredictValueUsingLinearModel(imageData, imageData.Length);
                if (predictedClass == 0)
                {
                    resultText.text = "Predicted Class: a ";
                    resultText.color = Color.black;
                } else if (predictedClass == 1)
                {
                    resultText.text = "Predicted Class: j ";
                    resultText.color = Color.black;
                } else if (predictedClass == 2)
                {
                    resultText.text = "Predicted Class: c ";
                    resultText.color = Color.black;
                } else
                {
                    resultText.text = "Error!";
                    resultText.color = Color.red;
                }
            } else if (ModelsDropdown.options[ModelsDropdown.value].text == "MLP")
            {
               float[] imageData = PreprocessImage(texture);
                int predictedClass = PredictValueUsingMLP(imageData, imageData.Length);
                if (predictedClass == 0)
                {
                    resultText.text = "Predicted Class: a ";
                    resultText.color = Color.black;
                } else if (predictedClass == 1)
                {
                    resultText.text = "Predicted Class: j ";
                    resultText.color = Color.black;
                } else if (predictedClass == 2)
                {
                    resultText.text = "Predicted Class: c ";
                    resultText.color = Color.black;
                } else
                {
                    resultText.text = "Error!";
                    resultText.color = Color.red;
                }
            }
            else if (ModelsDropdown.options[ModelsDropdown.value].text == "RBF Network")
            {
               float[] imageData = PreprocessImage(texture);
                int predictedClass = PredictValueUsingRBF(imageData, imageData.Length);
                if (predictedClass == 0)
                {
                    resultText.text = "Predicted Class: a ";
                    resultText.color = Color.black;
                } else if (predictedClass == 1)
                {
                    resultText.text = "Predicted Class: j ";
                    resultText.color = Color.black;
                } else if (predictedClass == 2)
                {
                    resultText.text = "Predicted Class: c ";
                    resultText.color = Color.black;
                } else
                {
                    resultText.text = "Error!";
                    resultText.color = Color.red;
                }
            } else
            {
                Debug.Log("other model");
            }
            
        } else 
        {
            resultText.text = "Please enter an image to predict";
            resultText.color = Color.red;
        } 
        
        
    }

    // Converts the image to grayscale and flattens it to a 1D array
    private float[] PreprocessImage(Texture2D texture)
    {
        var pixels = texture.GetPixels();
        float[] grayscalePixels = new float[pixels.Length];
        for (int i = 0; i < pixels.Length; i++)
        {
            grayscalePixels[i] = pixels[i].grayscale; // Unity's Color.grayscale calculates the luminance of the color
        }
        return grayscalePixels;
    }
}
