#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

int main()
{
    std::vector<cv::Mat> images;
    std::vector<std::string> labels;

    std::string datasetPath = "C:\\Users\\Maureen\\People-Counting\\Ifw";
    std::vector<std::string> personDirs = {"person_0", "person_1", "person_2"};

    for (const auto& personDir : personDirs)
    {
        std::string personPath = datasetPath + "/" + personDir;
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

    // Split the data into training and test sets
    int numImages = 45;
    int numTrain = static_cast<int>(numImages * 0.8); // 80% for training
    std::vector<int> indices(numImages);
    for (int i = 0; i < numImages; ++i)
    {
        indices[i] = i;
    }    std::random_shuffle(indices.begin(), indices.end());

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

    // Train the face recognition model
    // ...
    
    return 0;
}
