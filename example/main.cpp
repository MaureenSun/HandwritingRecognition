#include <iostream>
#include <vector>
#include <string>
#include <algorithm> // for std::shuffle
#include <random> // for std::default_random_engine
#include <opencv2/opencv.hpp>
#include <tensorflow/core/public/session.h>
#include <tensorflow/cc/ops/standard_ops.h>

using namespace tensorflow;
using namespace tensorflow::ops;

int main()
{
    std::vector<cv::Mat> images;
    std::vector<int> labels;

    // Load MNIST dataset
    // Assuming you have MNIST dataset stored as images and labels

    // Adjust MNIST dataset path accordingly

    std::string datasetPath = "C:\Users\\Maureen\\MNIST_dataset";

    for (int i = 0; i < numImages; ++i)
    {
        // Load image
        cv::Mat image = cv::imread(datasetPath + "/image_" + std::to_string(i) + ".png", cv::IMREAD_GRAYSCALE);
        images.push_back(image);

        // Load label
        // Assuming labels are integer values representing digits (0-9)
        int label;
        // Load label for the image, adjust this part accordingly based on how labels are stored
        labels.push_back(label);
    }

    // Split data into training and test sets
    int numTrain = static_cast<int>(images.size() * 0.8); // 80% for training
    std::vector<int> indices(images.size());
    for (int i = 0; i < indices.size(); ++i)
    {
        indices[i] = i;
    }

    // Shuffling indices
    std::random_device rd;
    std::default_random_engine engine(rd());
    std::shuffle(indices.begin(), indices.end(), engine);

    std::vector<cv::Mat> trainImages;
    std::vector<int> trainLabels;
    std::vector<cv::Mat> testImages;
    std::vector<int> testLabels;

    for (int i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        if (i < numTrain)
        {
            trainImages.push_back(images[idx]);
            trainLabels.push_back(labels[idx]);
        }
        else
        {
            testImages.push_back(images[idx]);
            testLabels.push_back(labels[idx]);
        }
    }

    // CNN architecture
    auto input = Placeholder(DT_FLOAT, Placeholder::Shape({None, 28, 28, 1}));
    auto conv1 = Conv2D(input, 32, {5, 5});
    auto pool1 = MaxPool(conv1, {2, 2}, {2, 2});
    auto conv2 = Conv2D(pool1, 64, {5, 5});
    auto pool2 = MaxPool(conv2, {2, 2}, {2, 2});
    auto flat = Reshape(pool2, {None, 7 * 7 * 64});
    auto dense = Dense(flat, 1024);
    auto dropout = Dropout(dense, 0.4);
    auto output = Dense(dropout, 10);

    // Loss function
    auto labels = Placeholder(DT_FLOAT, Placeholder::Shape({None, 10}));
    auto cross_entropy = ReduceMean(-ReduceSum(labels * Log(output), {1}));

    // Optimizer
    auto optimizer = GradientDescentOptimizer(0.001);
    auto train = optimizer->Minimize(cross_entropy);

    // Session initialization and training loop
    Session* session;
    TF_CHECK_OK(NewSession(SessionOptions(), &session));
    TF_CHECK_OK(session->Run({Assign(weights, ...), Assign(bias, ...)}, nullptr)); // Initialize weights and biases

    // Training loop
    int num_epochs = 10; // Adjust the number of epochs as needed
    int batch_size = 32; // Adjust the batch size as needed

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        for (int batch = 0; batch < trainImages.size() / batch_size; batch++) {
            // Extract batch images and labels
            std::vector<cv::Mat> batchImages(trainImages.begin() + batch * batch_size, trainImages.begin() + (batch + 1) * batch_size);
            std::vector<int> batchLabels(trainLabels.begin() + batch * batch_size, trainLabels.begin() + (batch + 1) * batch_size);

            // Preprocess batch images (resize, normalize, etc.) and convert to TensorFlow format
            auto x_batch = ...; // Convert batch images to TensorFlow tensor format
            auto y_batch = ...; // Convert batch labels to TensorFlow tensor format

            // Run training operation
            TF_CHECK_OK(session->Run({{input, x_batch}, {labels, y_batch}}, {}, {train}, nullptr));
        }
    }

    // Evaluation
    // Assuming you have a separate test set for evaluation
    auto x_test = ...; // Convert test images to TensorFlow tensor format
    auto y_test = ...; // Convert test labels to TensorFlow tensor format

    auto correct_predictions = ...; // Calculate correct predictions

    // Output evaluation metrics (accuracy, etc.)

    return 0;
}
