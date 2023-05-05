#include <iostream>
#include <opencv2/opencv.hpp>

int main()
{
    // Load an image of Aaron Eckhart from the Labeled Faces in the Wild dataset
    cv::Mat image = cv::imread("C:/Users/Maureen/People-Counting/lfw/Aaron_Eckhart/Aaron_Eckhart_0001.jpg");

    // Check if the image was loaded successfully
    if (!image.data)
    {
        std::cout << "Failed to load image" << std::endl;
        return -1;
    }

    // Display the image
    cv::imshow("Image", image);
    cv::waitKey(0);

    return 0;
}
