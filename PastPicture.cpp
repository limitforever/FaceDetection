
#include <iostream>
#include <fstream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/stitching/stitcher.hpp"


using namespace std;
using namespace cv;

bool try_use_gpu = false;
vector<Mat> imgs;
string result_name = "C:\\Users\\Administrator\\Desktop\\face.test\\test\\test\\result.jpg";

//void printUsage();
//int parseCmdArgs(int argc, char** argv);

int main(int argc, char* argv[])
{

	Mat img = imread("C:\\Users\\Administrator\\Desktop\\face.test\\test\\face\\1.bmp");
	imgs.push_back(img);
	img = imread("C:\\Users\\Administrator\\Desktop\\face.test\\test\\face\\2.bmp");
	imgs.push_back(img);
	img = imread("C:\\Users\\Administrator\\Desktop\\face.test\\test\\face\\3.bmp");
	imgs.push_back(img);

	Mat pano;
	Stitcher stitcher = Stitcher::createDefault(try_use_gpu);
	Stitcher::Status status = stitcher.stitch(imgs, pano);

	if (status != Stitcher::OK)
	{
		cout << "Can't stitch images, error code = " << int(status) << endl;
		return -1;
	}

	imwrite(result_name, pano);
	return 0;
}