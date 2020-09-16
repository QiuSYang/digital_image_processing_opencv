/*
# Ö÷º¯Êý
*/
#include "yolo4_detection.h"

int main(int argc, char *argv[])
{
    Yolo4Detection *detectObj = new Yolo4Detection();
    string inputImgPath = "test.jpg";
    string outputImgPath = "test_result.jpg";
    Mat inputImage = imread(inputImgPath, IMREAD_COLOR);
    detectObj->Initialize(inputImage.rows, inputImage.cols);
    detectObj->Detecting(inputImage);
    namedWindow("Show result image", WINDOW_AUTOSIZE);
    imshow("Show result image", detectObj->GetFrame());
    waitKey();
    destroyAllWindows();
    imwrite(outputImgPath, detectObj->GetFrame());

    if (detectObj != NULL)
    {
        delete detectObj;
        detectObj = NULL;
    }
    system("pause");
    return 0;
}