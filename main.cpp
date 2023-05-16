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
    std::vector<std::string> labels;

    std::string datasetPath = "C:\Users\\Maureen\\People-Counting\\Ifw";
    std::vector<std::string> personDirs = {"person_0", "person_1", "person_2"};

    for (const auto& personDir : personDirs)
    {
        std::string personPath = datasetPath + "\\" + personDir;
        std::vector<std::string> imageFiles;
        cv::glob(personPath + "/*.jpg", imageFiles, false);

        for (const auto& imageFile : imageFiles)
        {
            cv::Mat image = cv::imread(imageFile, cv::IMREAD_GRAYSCALE);
            std::string label = personDir;
            images.push_back(image);
            labels.push_back(label);
        }
    }

    // Split data in traiing and test
    int numImages = 45;
    int numTrain = static_cast<int>(numImages * 0.8); // 80% for training
    std::vector<int> indices(numImages);
    for (int i = 0; i < numImages; ++i)
    {
        indices[i] = i;
    }

    // Shuffling indices....
    std::random_device rd;
    std::default_random_engine engine(rd());
    std::shuffle(indices.begin(), indices.end(), engine);

    std::vector<cv::Mat> trainImages;
    std::vector<std::string> trainLabels;
    std::vector<cv::Mat> testImages;
    std::vector<std::string> testLabels;

    for (int i = 0; i < numImages; ++i)
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
    auto input = Placeholder(DT_FLOAT, Placeholder::Shape({None, 784}));
    auto weights = Variable(RandomNormal(Shape({784, 10})));
    auto bias = Variable(RandomNormal(Shape({10})));
    auto output = Softmax(MatMul(input, weights) + bias);

    //  loss function
    auto labels = Placeholder(DT_FLOAT, Placeholder::Shape({None, 10}));
    auto cross_entropy = ReduceMean(-ReduceSum(labels * Log(output), {1}));

    auto optimizer = GradientDescentOptimizer(0.5);
    auto train = optimizer->Minimize(cross_entropy);

    Session* session;
    TF_CHECK_OK(NewSession(SessionOptions(), &session));
    TF_CHECK_OK(session->Run({Assign(weights, ...), Assign(bias, ...)}, nullptr));
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        for (int batch = 0; batch < num_batches; batch++) {
            // Next part of data
            auto x_batch = ...;
            auto y_batch = ...;

            TF_CHECK_OK(session->Run({{input, x_batch}, {labels, y_batch}}, {}, {train}, nullptr));
        }
    }

    auto x_test = ...;
    auto y_test = ...;
    auto correct_predictions = ...;
    
    
    return 0;
}
