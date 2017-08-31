#include<iostream>
#include <string.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <math.h>
#include <cv.h>  
#include "highgui.h"
#include "Feature.h"
#include "WeakClassifier.h"
#include "StrongClassifier.h"
using namespace std;
const int faces = 2429;
const int nonfaces = 28121;
CvHaarClassifierCascade *cascade;
//static CvMemStorage *storage = 0;
CvRect  rs1[1000000];
CvRect  rs2[10000];
//CvRect  rs[100000000];
//float pBuffer[1000000];
//float *pGrayBuffer = new float[1000000];
//static double totaltime = 0;
//char outfilename[] = "e://integra_set.txt";
//ofstream out(outfilename);
int pBuffer[361];
struct sc_WeakClassifier_set{
	int type;
	int xc;
	int yc; 
	int width;
	int height;
	float threshold;
	bool polarity;
}SC[200];
int get_para(int i);
/*
int rectangleValue(int *ii, int iiwidth, int rx, int ry, int rw, int rh) {
	int value = ii[((ry + rh - 1)*iiwidth) + (rx + rw - 1)];    out << "pBuffer[" << ((ry + rh - 1)*iiwidth) + (rx + rw - 1) << "]";     //A+B+C+D
	if (rx > 0) {
		value -= ii[((ry + rh - 1)*iiwidth) + (rx - 1)]; out << "-" << "pBuffer[" << ((ry + rh - 1)*iiwidth) + (rx - 1) << "]";
	}//B+D(x>0)
	if (ry > 0) {
		value -= ii[((ry - 1)*iiwidth) + (rx + rw - 1)]; out << "-" << "pBuffer[" << ((ry - 1)*iiwidth) + (rx + rw - 1) << "]";
	}//C+D(y>0)
	if (rx > 0 && ry > 0) {
		value += ii[((ry - 1)*iiwidth) + (rx - 1)]; out << "+" << "pBuffer[" << ((ry - 1)*iiwidth) + (rx - 1) << "]";
	}  //D(x>0且y>0)
	//out << "+";
	return value;
}

int getValue(int *ii,int i) {
	// 5 types of feature (A, B, C, C^t, D)
	int iiwidth = 19;
	int t = 0;
	out << "case " << i << ":" << endl;
	out << "t = ";
	switch (SC[i].type) {
		//switch (type) {  //xc,yc为积分图起始位置的坐标（左上角）	
	case 0:
		
		rectangleValue(ii, iiwidth, SC[i].xc + (SC[i].width / 2), SC[i].yc, SC[i].width / 2, SC[i].height);
		out << "-(";
		rectangleValue(ii, iiwidth,  SC[i].xc, SC[i].yc, SC[i].width / 2, SC[i].height);
		out << ");" ;
		out << endl << "break;";
		//std::cout << "特征1的特征值为：" << t << std::endl;
		return t;
		break;
	case 1:
		rectangleValue(ii, iiwidth, SC[i].xc, SC[i].yc, SC[i].width, SC[i].height / 2);
		out << "-(";
		rectangleValue(ii, iiwidth, SC[i].xc, SC[i].yc + (SC[i].height / 2), SC[i].width, SC[i].height / 2);
		out << ");";
		out << endl << "break;";
		//std::cout << "特征2的特征值为：" << t << std::endl;
		return t;
		break;
	case 2:
		out << "(";
		rectangleValue(ii, iiwidth, SC[i].xc, SC[i].yc, SC[i].width / 3, SC[i].height);
		out << "+";
		rectangleValue(ii, iiwidth, SC[i].xc + (SC[i].width * 2 / 3), SC[i].yc, SC[i].width / 3, SC[i].height);
		out << "- 2 *(";
		rectangleValue(ii, iiwidth, SC[i].xc + (SC[i].width / 3), SC[i].yc, SC[i].width / 3, SC[i].height);
		out << "))/ 2;";
		out << endl << "break;";
		//std::cout << "特征3的特征值为：" << t << std::endl;
		return t;
		break;
	case 3:
		out << "(";
		rectangleValue(ii, iiwidth, SC[i].xc, SC[i].yc, SC[i].width, SC[i].height / 3);
		out << "+";
		rectangleValue(ii, iiwidth, SC[i].xc, SC[i].yc + (SC[i].height * 2 / 3), SC[i].width, SC[i].height / 3);
		out << "- 2 *(";
		rectangleValue(ii, iiwidth, SC[i].xc, SC[i].yc + (SC[i].height / 3), SC[i].width, SC[i].height / 3);
		out << "))/ 2;";
		out << endl << "break;";
		//std::cout << "特征4的特征值为：" << t << std::endl;
		return t;
		break;
	case 4:
		out << "(";
		rectangleValue(ii, iiwidth, SC[i].xc, SC[i].yc, SC[i].width * 2 / 5, SC[i].height);
		out << "+";
		rectangleValue(ii, iiwidth, SC[i].xc + (SC[i].width * 3 / 5), SC[i].yc, SC[i].width * 2 / 5, SC[i].height);
		out << "- 4 *(";
		rectangleValue(ii, iiwidth, SC[i].xc + (SC[i].width * 2 / 5), SC[i].yc, SC[i].width / 5, SC[i].height);
		out << "))/ 4;";
		out << endl << "break;";
		//std::cout << "特征3的特征值为：" << t << std::endl;
		return t;
		break;
	case 5:
		out << "(";
		rectangleValue(ii, iiwidth, SC[i].xc, SC[i].yc, SC[i].width, SC[i].height * 2 / 5);
		out << "+";
		rectangleValue(ii, iiwidth, SC[i].xc, SC[i].yc + (SC[i].height * 3 / 5), SC[i].width, SC[i].height * 2 / 5);
		out << "- 4 *(";
		rectangleValue(ii, iiwidth, SC[i].xc, SC[i].yc + (SC[i].height * 2 / 5), SC[i].width, SC[i].height / 5);
		out << "))/ 4;";
		out << endl << "break;";
		//std::cout << "特征4的特征值为：" << t << std::endl;
		return t;
		break;

	case 6:
		out << "(";
		rectangleValue(ii, iiwidth, SC[i].xc, SC[i].yc, SC[i].width / 5, SC[i].height);
		out << "+";
		rectangleValue(ii, iiwidth, SC[i].xc + (SC[i].width * 4 / 5), SC[i].yc, SC[i].width / 5, SC[i].height);
		out << "- 2/3 *(";
		rectangleValue(ii, iiwidth, SC[i].xc + (SC[i].width / 5), SC[i].yc, SC[i].width * 3 / 5, SC[i].height);
		out << "))*1.5;";
		out << endl << "break;";
		//std::cout << "特征3的特征值为：" << t << std::endl;
		return t;
		break;
	case 7:
		out << "(";
		rectangleValue(ii, iiwidth, SC[i].xc, SC[i].yc, SC[i].width, SC[i].height / 5);
		out << "+";
		rectangleValue(ii, iiwidth, SC[i].xc, SC[i].yc + (SC[i].height * 4 / 5), SC[i].width, SC[i].height / 5);
		out << "- 2/3 *(";
		rectangleValue(ii, iiwidth, SC[i].xc, SC[i].yc + (SC[i].height / 5), SC[i].width, SC[i].height * 3 / 5);
		out << "))*1.5;";
		out << endl << "break;";
		//std::cout << "特征4的特征值为：" << t << std::endl;
		return t;
		break;

	case 8:
		out << "(";
		rectangleValue(ii, iiwidth, SC[i].xc + (SC[i].width / 2), SC[i].yc, SC[i].width / 2, SC[i].height / 2);
		out << "+";
		rectangleValue(ii, iiwidth, SC[i].xc, SC[i].yc + (SC[i].height / 2), SC[i].width / 2, SC[i].height / 2);
		out << "-(";
		rectangleValue(ii, iiwidth, SC[i].xc + (SC[i].width / 2), SC[i].yc + (SC[i].height / 2), SC[i].width / 2, SC[i].height / 2);
		out << ")-(";
		rectangleValue(ii, iiwidth, SC[i].xc, SC[i].yc, SC[i].width / 2, SC[i].height / 2);
		out << "))/2;";
		out << endl << "break;";
		//std::cout << "特征5的特征值为：" << t << std::endl;
		return t;
		break;
	case 9:
		rectangleValue(ii, iiwidth, SC[i].xc, SC[i].yc, SC[i].width, SC[i].height);
		out << "- 9 *(";
		rectangleValue(ii, iiwidth, SC[i].xc + SC[i].width / 3, SC[i].yc + (SC[i].height / 3), SC[i].width / 3, SC[i].height / 3);
		out << ")/8;";
		out << endl << "break;";
		//std::cout << "特征5的特征值为：" << t << std::endl;
		return t;
		break;
	default:
		std::cout << "错误的特征：" << SC[i].type << " 并不存在！" << std::endl;
		exit(-1);
		break;
	}
	
}*/
/*
float* integral_image(float *img, int width, int height) {    //利用Adaboost提供的公式，s(x,y)=s(x,y-1)+img(x,y)  坐标定位
	float* ii = new float[width*height];                      //                        ii(x,y)=ii(x-1,y)+s(x,y)
	float* s = new float[width*height];
	int x, y;

	for (x = 0; x < height; x++) {
		for (y = 0; y < width; y++) {
			if (y == 0) s[(x*width) + y] = img[(x*width) + y];
			else s[(x*width) + y] = s[(x*width) + y - 1] + img[(x*width) + y];

			if (x == 0) ii[(x*width) + y] = s[(x*width) + y];
			else ii[(x*width) + y] = ii[((x - 1)*width) + y] + s[(x*width) + y];
		}
	}
	return ii;
}
float evaluate_integral_rectangle(float *ii, int iiwidth, int x, int y, int w, int h,int type) {    //利用积分图，计算特征值
	
	float value = ii[((y + h - 1)*iiwidth) + (x + w - 1)];                    //  初始化特征值为4位置，即A+B+C+D
	if (x > 0) value -= ii[((y + h - 1)*iiwidth) + (x - 1)];                  //  计算x>0时候的特征值，即4-3，即B+D
	if (y > 0) value -= ii[(y - 1)*iiwidth + (x + w - 1)];                    //  计算y>0时候的特征值，即4-2，即C+D
	if (x > 0 && y > 0) value += ii[(y - 1)*iiwidth + (x - 1)];               //  计算x>0且y>0的特征值，即4+1-（2+3），即D
	return value;
}*/
/*float* Create_test_sample(IplImage *img)
{
	char test_img_name[10];
	//sprintf(test_img_name, "C:\\Users\\Administrator\\Desktop\\face.test\\test\\test\\%d.jpg", 1);
	//IplImage* img = cvLoadImage(test_simg_name, 0);
	CvScalar ss;
	int height = img->height, width = img->width;
	//float *pGrayBuffer = new float[img->imageSize]; //存取图像像素值
	uchar* ptr = (uchar *)(img->imageData);
	for (int i = 0; i < height; i++)
	{
		uchar* ptr = (uchar *)(img->imageData + i*img->widthStep);
		for (int x = 0; x < img->width; x++)
		{
			ss = cvGet2D(img, i, x);
			//	*ptr = ((uchar *)(img->imageData + i*img->widthStep))[x];
			pGrayBuffer[i*img->width + x] = float(ss.val[0]);
			//pGrayBuffer[i*img->width + x] = (float)(ptr[x]);
			//if (pGrayBuffer[i*img->width + x] == 0)
			//cout<<"error  "<<x<<endl;
		}
	}
	return pGrayBuffer;
}
*/
void Create_Test_Sample(IplImage *iimg,CvRect r)
{
	
	//double tt = (double)cvGetTickCount();//精确测量函数的执行时间
	//char test_img_name[10];
	CvScalar ss;
	int height = r.height, width = r.width;
	int xs = r.x, ys = r.y;
	//uchar* ptr = (uchar *)(iimg->imageData);
	int k1 = 0, k2 = 0;
	//double t = (double)cvGetTickCount();//精确测量函数的执行时间
	int xbinary = xs + width, ybinary = ys + height;
	for (int i = ys; i < ybinary && i < iimg->height; i++)
	{
		uchar* ptr = (uchar *)(iimg->imageData + i*iimg->widthStep);
		//memset(pBuffer, char(ptr + xs), width);
		//cout << "pBuffer" << float((ptr+xs)[0]) << endl;
		int k11 = k1*width;
		for (int x = xs; x < xbinary && x<iimg->width; x++)
		{

	//	ss = cvGet2D(iimg, i, x);

		//cvSet2D(tmp_img, k1, k2, ss);
	//	pBuffer[k1*width + k2] = float(ss.val[0]);
			pBuffer[k11 + k2] = int(ptr[x]);
		//	cout << pBuffer[k1*width + k2] << endl;

		//cvSetReal2D(tmp_img, k1, k2, pBuffer[k1*width + k2]);
		k2++;
		}
		k2 = 0;
		k1++;
	}
	
	//t = (double)cvGetTickCount() - t; //计算检测到人脸所需时间
	//totaltime = totaltime + t;
	//printf("检测所用时间 = %gms\n", t / ((double)cvGetTickFrequency() * 1000));//打印到屏幕
	//cvShowImage("TestifyImage", tmp_img);
	//cvWaitKey(0); //等待按键
	//cvDestroyWindow("TestifyImage");//销毁窗口
	//return pBuffer;
	//tt = (double)cvGetTickCount() - tt; //计算检测到人脸所需时间
	//printf("检测所用总时间 = %gms\n", tt / ((double)cvGetTickFrequency() * 1000));//打印到屏幕
	//system("pause");
}

//此函数用来计算两个矩形的重合部分，若无返回空
CvSeq * Common_Rect(CvSeq *seq)
{
	
	CvSeq* result_seq = 0;
	CvMemStorage* temp_storage = 0;
	temp_storage = cvCreateMemStorage(0);
	result_seq = cvCreateSeq(0, sizeof(CvSeq), sizeof(CvRect), temp_storage);
	//cvSeqPush(result_seq, (CvRect*)cvGetSeqElem(seq, 0));
	//int *flag = new int[seq->total];
	//int flag;
	for (int i = 0; i < seq->total; i++)
	{
		//cout << seq->total << endl;
		CvRect rect1 = *(CvRect*)cvGetSeqElem(seq, i);
		//flag =0;
		for (int j = i + 1; j < seq->total; j++)
		{
			//CvRect rect1 = *(CvRect*)cvGetSeqElem(seq, i);
			CvRect rect2 = *(CvRect*)cvGetSeqElem(seq, j);
			CvPoint p11 = cvPoint(rect1.x, rect1.y);								//矩形 rect1 左上顶点
			CvPoint p12 = cvPoint(rect1.x + rect1.width, rect1.y + rect1.height);	//矩形 rect1 右下顶点
			CvPoint p21 = cvPoint(rect2.x, rect2.y);								//矩形 rect2 左上顶点
			CvPoint p22 = cvPoint(rect2.x + rect2.width, rect2.y + rect2.height);	//矩形 rect2 右下顶点
			int x1, y1, x2, y2;
			if ((p11.x > p22.x || p11.y > p22.y || p12.x < p21.x || p12.y < p21.y))
			{
				continue;
			}
			else
			{
				cvSeqRemove(seq, j);
			//	cout << "Rect" << j << "has been moved" << endl;
			}
				//rect2 = cvRect(0, 0, 0, 0);
		}
			cvSeqPush(result_seq, &rect1);
			//cout << "Rect" << i << "has been added" << endl;
	}
	return result_seq;
	
}



static int is_equal(const void* _r1, const void* _r2, void*)
{
	const CvRect* r1 = (const CvRect*)_r1;
	const CvRect* r2 = (const CvRect*)_r2;
	int distance = cvRound(r1->width*0.3);

	return r2->x <= r1->x + distance &&
		r2->x >= r1->x - distance &&
		r2->y <= r1->y + distance &&
		r2->y >= r1->y - distance &&
		r2->width <= cvRound(r1->width * 1.5) &&
		cvRound(r2->width * 1.5) >= r1->width;
}
//给出一个矩形序列 rs ，将其合并 , 结果放入 result_seq 中返回
CvSeq * Merge(CvRect * rs, int count)
{

	CvSeq* seq = 0;
	CvSeq* seq2 = 0;
	CvSeq* idx_seq = 0;
	CvSeq* result_seq = 0;
	CvMemStorage* temp_storage = 0;
	CvMemStorage* storage = 0;
	CvAvgComp* comps = 0;
	int i;
	int min_neighbors = 1;

	//CV_CALL( temp_storage = cvCreateChildMemStorage( storage ));
	temp_storage = cvCreateMemStorage(0);
	storage = cvCreateMemStorage(0);
	seq = cvCreateSeq(0, sizeof(CvSeq), sizeof(CvRect), temp_storage);
	seq2 = cvCreateSeq(0, sizeof(CvSeq), sizeof(CvAvgComp), temp_storage);
	result_seq = cvCreateSeq(0, sizeof(CvSeq), sizeof(CvAvgComp), storage);

	//if( min_neighbors == 0 )
	//    seq = result_seq;

	//CvRect rect = cvRect(ix,iy,win_size.width,win_size.height);
	//cvSeqPush( seq, &rect );

	for (i = 0; i<count; i++)
	{
		//cout << rs[i].x << " " << rs[i].y << " " << rs[i].width << " " << rs[i].height << endl;
		cvSeqPush(seq, &rs[i]);
	}

	if (min_neighbors != 0)
	{
		// group retrieved rectangles in order to filter out noise 
		int ncomp = cvSeqPartition(seq, 0, &idx_seq, is_equal, 0);//拆分seq序列为等效的类,函数返回nomp等效类的数目
		//cout << "ncomp:" << ncomp << endl;
		comps = (CvAvgComp*)cvAlloc((ncomp + 1)*sizeof(comps[0])); // 
		memset(comps, 0, (ncomp + 1)*sizeof(comps[0]));
		//cout << "seq->total:" << seq->total<<endl;
		// count number of neighbors
		for (i = 0; i < seq->total; i++)
		{
			CvRect r1 = *(CvRect*)cvGetSeqElem(seq, i);
			int idx = *(int*)cvGetSeqElem(idx_seq, i);
			assert((unsigned)idx < (unsigned)ncomp);
			//cout << "idx[" << i << "]:" << idx << endl;
			comps[idx].neighbors++;
			//cout << "comps[i].neighbors:" << comps[idx].neighbors << endl;
			comps[idx].rect.x += r1.x;
			comps[idx].rect.y += r1.y;
			comps[idx].rect.width += r1.width;
			comps[idx].rect.height += r1.height;
		}
		
		// calculate average bounding box
		for (i = 0; i < ncomp; i++)
		{
			
			int n = comps[i].neighbors;
			if (n >= min_neighbors)
			{
				CvAvgComp comp;
				comp.rect.x = (comps[i].rect.x * 2 + n) / (2 * n);
				comp.rect.y = (comps[i].rect.y * 2 + n) / (2 * n);
				comp.rect.width = (comps[i].rect.width * 2 + n) / (2 * n);
				comp.rect.height = (comps[i].rect.height * 2 + n) / (2 * n);
				comp.neighbors = comps[i].neighbors;
				
				cvSeqPush(seq2, &comp);
			}
		}

		// filter out small face rectangles inside large face rectangles
		for (i = 0; i < seq2->total; i++)
		{
			CvAvgComp r1 = *(CvAvgComp*)cvGetSeqElem(seq2, i);
			int j, flag = 1;

			for (j = 0; j < seq2->total; j++)
			{
				CvAvgComp r2 = *(CvAvgComp*)cvGetSeqElem(seq2, j);
				//int distance = cvRound(r2.rect.width * 0.3);
				int distance = 0;
				if (i != j &&
					r1.rect.x >= r2.rect.x - distance &&
					r1.rect.y >= r2.rect.y - distance &&
					r1.rect.x + r1.rect.width <= r2.rect.x + r2.rect.width + distance &&
					r1.rect.y + r1.rect.height <= r2.rect.y + r2.rect.height + distance
					&& (r2.neighbors > MAX(1, r1.neighbors) || r1.neighbors < 1))
				{
					flag = 0;
					break;
				}
			}

			if (flag)
			{
				//cout << " " << r1.rect.x << " " << r1.rect.y << " " << r1.rect.width << " " << r1.rect.height << endl;
				cvSeqPush(result_seq, &r1);
				/* cvSeqPush( result_seq, &r1.rect ); */
			}
		}
	}
	return result_seq;

}



void Detect_And_Draw(IplImage* img1, IplImage* img , double *sc_weight, double Threshold)
{
	double final_threshold = Threshold *0.54;
	double tt = (double)cvGetTickCount();//精确测量函数的执行时间
	if (img1->nChannels!=1)
	{
		cvCvtColor(img1, img, CV_BGR2GRAY);//cvCvtColor(src,des,CV_BGR2GRAY)  
	}
	//cout << img->nChannels << endl;
	cvSmooth(img, img, CV_GAUSSIAN, 5, 5, 0, 0);//3x3
	
	vector<WeakClassifier*>::iterator it;

	//vector<double>::iterator iit;
	double scale = 1.3;
	//IplImage* gray = cvCreateImage(cvSize(img->width, img->height), 8, 1);  //提取图像宽、高，位深度8，通道数1，赋予灰度值
	const int k = 6;
	//int i;
	
	int i = 0, j, a, b, increment;
	int x=0, y=0;
	double s;
	//int fnotfound = 0;
	float mean, stdev;
	
	int base_resolution = 19;
	int count_window = 0;
	int ct;
	int cn;
	cn = 0;
	ct = 0;
	int cv = 0;
	int smaller = (img->width < img->height) ? img->width : img->height;	//图像的长宽较小者
	
	IplImage* small_img = 0;
	//double tt = (double)cvGetTickCount();//精确测量函数的执行时间
	for (s = 1; (double)base_resolution* pow(scale, cvRound(cv * 0.4)) <= (double)smaller; s *= scale)
	{
		cv++;
	
		small_img = cvCreateImage(cvSize(cvRound(double(img->width) / s),   //提取图像宽缩小规模，高缩小规模，位深度8，通道数1，赋予small_img
			cvRound(double(img->height) / s)),
			img->depth, img->nChannels);
		cvResize(img, small_img, CV_INTER_LINEAR);   //双线性插值
		//cvSmooth(small_img, small_img, CV_GAUSSIAN, 5, 5, 0, 0);//3x3
	//	cvEqualizeHist(small_img, small_img);//灰度图象直方图均衡化
		//cvShowImage("TestifyImage", small_img);
	   //  cvWaitKey(0); //等待按键
		//cvDestroyWindow("TestifyImage");//销毁窗口
		int width = small_img->width, height = small_img->height;
	
		//double w = j;
		int xbinary = width - base_resolution + 1;
		int ybinary = height - base_resolution + 1;
		double plo = -1;
		while (x <= xbinary)
		{
			while (y <= ybinary)
			{
			
				double w = 0;
				//定义矩形框，获得感兴趣区域（19*19）
				//float* img_test = 
				*(rs2 + count_window) = cvRect(x, y, base_resolution, base_resolution);
				Create_Test_Sample(small_img, rs2[count_window]);//提取该窗口的像素值
				
			//	cout << img_test[0] << " " << img_test[100] << img_test[200] << endl;
			//	double ttt = (double)cvGetTickCount();
				for (ct = 0; ct<200; ct++)
				{
					
					int hvalue = get_para(ct);
					//out << endl;
					//cout << SC[ct].type << " " << SC[ct].xc << " " << SC[ct].yc << " " << SC[ct].width << " " << SC[ct].height << " "<<hvalue<<endl;
					//system("pause");
					//int ht = 0;
					//plo = -1;
					//double thre = (*it)->getthreshold();
					//cout << hvalue << endl;
					double thre = SC[ct].threshold;
					if (SC[ct].polarity)
					{
						//plo = 1;
						if (thre > hvalue)
						{
							//	ht = 1;
							//	w++;
							//w += ht * 2;
							w += (*(sc_weight + ct));
						}
					}
					else
					{
						if (thre < hvalue)
						{
							//	ht = 1;
							//	w++;
							//w += ht * 2;
							w += (*(sc_weight + ct));
						}
					}
					
					//cout << hvalue << endl;
				}
				//out.close();
				//cout << endl;
				//cout << ct << endl;
				//system("pause");
			//	ttt = (double)cvGetTickCount() - ttt; //计算检测到人脸所需时间
			//	printf("计算value所用总时间 = %gms\n", ttt / ((double)cvGetTickFrequency() * 1000));//打印到屏幕
				ct = 0;
			
				//cout << "w：" << w <<"     Threshold：" << Threshold / 2<<endl;
				if (w >= final_threshold)			// rs2 用来暂时保存通过此强分类器的窗口

				{

					rs1[cn].height = rs2[count_window].height *s+0.5; 
					rs1[cn].width = rs2[count_window].width *s + 0.5;
					rs1[cn].x = rs2[count_window].x *s + 0.5;
					rs1[cn].y = rs2[count_window].y *s + 0.5;
					cn++;
					//	system("pause");
				}
				//cout <<  w << "  " << Threshold / 2<<endl;
				count_window++;
				y += k;
			
			}
			x += k;
			y = 0;

		}
	  
		x = 0;
		smaller = (small_img->width < small_img->height) ? small_img->width : small_img->height;
		cvReleaseImage(&small_img);
		//cout << smaller << "  " << count_window << endl;
	}
	
	//printf("检测像素所用时间 = %gms\n", totaltime / ((double)cvGetTickFrequency() * 1000));//打印到屏幕

	//cout << "cv:" << cv << endl;
    std::cout << "可能的人脸个数"<<count_window << endl;
	count_window = cn;

	//for (i = 0; i < count_window; i++)
//	{
	//	cvRectangle(img1, cvPoint(rs1[i].x, rs1[i].y), cvPoint(rs1[i].x + rs1[i].width, rs1[i].y + rs1[i].height), CV_RGB(255, 0, 0), 1);
		//cout << rs2[i].x << " " << rs2[i].y << " " << rs2[i].x + rs2[i].width << " " << rs2[i].y + rs2[i].height << endl;
		//cvNamedWindow("result1", CV_WINDOW_AUTOSIZE);
		//cvShowImage("result1", img1);
		//cvWaitKey(0);
	//}

	//cvNamedWindow("result", CV_WINDOW_AUTOSIZE);
	//cvShowImage("result", img1);
	//cvWaitKey(0);
	//printf("*****************************************\n");

	CvSeq * faces = Merge(rs1, count_window);
	//std::cout << "faces=" << faces->total << endl;
	//std::cout << "faces->total" << faces->total << endl;
	static CvScalar colors[] =
	{
		{ { 0, 0, 255 } },
		{ { 0, 128, 255 } },
		{ { 0, 255, 255 } },
		{ { 0, 255, 0 } },
		{ { 255, 128, 0 } },
		{ { 255, 255, 0 } },
		{ { 255, 0, 0 } },
		{ { 255, 0, 255 } }
	};

	CvSeq * newface = Common_Rect(faces);
	cout << " 检测到的人脸总数：" << newface->total << endl;
	for (i = 0; i < (newface ? newface->total : 0); i++)
	
	{
		
		CvRect* r = (CvRect*)cvGetSeqElem(faces, i);//函数 cvGetSeqElem 查找序列中索引所指定的元素，并返回指向该元素的指针
	/*	CvPoint center;
		int radius;
		if (r->width<60)
		{
			center.x = cvRound((r->x + r->width*0.5));
			center.y = cvRound((r->y + r->height*0.5));
		}
		else
		{
			center.x = cvRound((r->x + r->width*0.3)*scale);
			center.y = cvRound((r->y + r->height*0.3)*scale);
		}
		
		if ((radius = cvRound((r->width + r->height)*0.25*scale))>0)
			cvCircle(img1, center, radius, colors[i % 8], 3, 8, 0);*/
		CvPoint pt1, pt2;
		pt1.x = r->x;
		pt2.x = (r->x + r->width);
		pt1.y = (r->y);
		pt2.y = (r->y + r->height);

		//画出矩形
		 cvRectangle( img1, pt1, pt2, colors[i%8], 3, 8, 0 );
		//cvRectangle(img1, pt1, pt2, CV_RGB(255, 0, 0), CV_FILLED, 8, 0);
		 
		
	}
	tt = (double)cvGetTickCount() - tt; //计算检测到人脸所需时间
	printf("检测所用总时间 = %gms\n", tt / ((double)cvGetTickFrequency() * 1000));//打印到屏幕
	cvShowImage("result", img1);
	//cvWaitKey(0);
	//用矩形检测
	
	
	//cvReleaseImage( &small_img );

	

	
}

int  main()
{
	char face[100],nonface[100];
	int ct = 0;
	
	char test_img_name[100];

	ifstream myfile("e://feature_1.txt");
	float scalefstep = 1.25;
	float slidefstep = 0.1;
	StrongClassifier *sc = new StrongClassifier();
	CvMemStorage *storage = 0;
	CvCapture *capture = 0;//初始化从摄像头中获取视频
	IplImage  *frame_copy = 0;
	IplImage *frame;
	const char *input_name;
	storage = cvCreateMemStorage(0);//创建内存块
   capture = cvCaptureFromCAM(0);//获取摄像头
	//cvNamedWindow("人脸识别", 1);//创建格式化窗口
	//char buffer[256]; 
	int a, b,c,d,e,g;
	double f,h;
	char str[100];
	int i = 0, j = 0;
	WeakClassifier *wc;
	Feature *fea;
	double weight_set[200];
	vector<WeakClassifier*> sc_WeakClassifier_set;
	double Threshold = 0;
	double tt = (double)cvGetTickCount();//精确测量函数的执行时间
	/*for (i = 0; i <= 3000000000;)
	{
		i = i + 1;
	}
	tt = (double)cvGetTickCount() - tt; //计算检测到人脸所需时间
	printf("检测所用总时间 = %gms\n", tt / ((double)cvGetTickFrequency() * 1000));//打印到屏幕
	system("pause");*/
	while (!myfile.eof())
	{
		
		myfile.getline(str, 100);
		sscanf(str, "%d,%d,%d,%d,%d,%lf,%d,%lf", &a, &b, &c, &d, &e, &f, &g, &h);
		//	cout << a << " " << b << " " << c << " " << d << " " << e << " " << f << " " << g << " "<<h<<endl;
		if (i < 200)
		{
			SC[i].type = a;
			SC[i].xc = b;
			SC[i].yc = c;
			SC[i].width = d;
			SC[i].height = e;
			SC[i].threshold = f;
			SC[i].polarity = g;
			//fea= new Feature(a,b,c,d,e);
			//wc = new WeakClassifier(fea,f,g);

			//	system("pause");
			//sc_set.push_back(wc, h);
			Threshold += h;

			weight_set[i] = h;
			//sc_WeakClassifier_set.push_back(wc);
			//cout << i << "  " << weight_set[i - 1] << endl;
		}
		i++;
	}
		 //sc->add(wc, h);
	
	//system("pause");
	myfile.close();
	/*
   for (int count = 1; count <= faces; count += 1)
	{
		sprintf(face, "C:\\Users\\Administrator\\Desktop\\face.test\\test\\face\\%d.bmp", count);
		IplImage* img_small = cvLoadImage(face, 0);
		//cvShowImage("result1", img_small);
		//cvWaitKey(0);
	//	test_facesamples(img_small, sc_WeakClassifier_set, weight_set, Threshold);
		if (test_facesamples(img_small, sc_WeakClassifier_set, weight_set, Threshold))
		{
			ct++;
		}
	}
	double face_rate = (double)ct / (double)faces;
	cout << "face识别出：" << ct << "张" << endl;
	cout << "检测率为：" << face_rate << endl;
	ct = 0;
	for (int count = 1; count <= nonfaces; count += 1)
	{
		if (count != 9999)
		{
			sprintf(nonface, "C:\\Users\\Administrator\\Desktop\\face.test\\test\\non-face\\cmu_%d.pgm", count);
			IplImage* img_small = cvLoadImage(nonface, 0);
			
			//test_nonfacesamples(img_small, sc_WeakClassifier_set, weight_set, Threshold);
			if (test_nonfacesamples(img_small, sc_WeakClassifier_set, weight_set, Threshold))
			{
				ct++;
			}
		}
	}
	double nonface_rate = (double)ct / (double)(nonfaces-1);
	cout << "face识别出：" << ct << "张" << endl;
	cout << "检测率为：" << nonface_rate << endl;
	*/
	/*
	for (int i = 1; i <= 10; i++)
	{
		sprintf(test_img_name, "C:\\Users\\Administrator\\Desktop\\face.test\\test\\test\\%d.jpg", i);
		IplImage* img = cvLoadImage(test_img_name, 1);
		IplImage* load_img = 0;
		load_img = cvCreateImage(cvSize(img->width, img->height), IPL_DEPTH_8U, 1);
		Detect_And_Draw(img, load_img, weight_set, Threshold); // 检测并且标识人脸
		cvReleaseImage(&load_img);
	}*/
	//
	//
	//
	
	//sc_WeakClassifier_set,
	
	if (capture)
	{
		IplImage* load_img = 0;
		frame = cvRetrieveFrame(capture); //获得由cvGrabFrame函数抓取的图片
		//if (!frame_copy){
			frame_copy = cvCreateImage(cvSize(frame->width, frame->height), IPL_DEPTH_8U, frame->nChannels);
			load_img = cvCreateImage(cvSize(frame->width, frame->height), IPL_DEPTH_8U, 1);
		//}
		//图像原点或者是左上角 (img->origin=IPL_ORIGIN_TL)或者是左下角(img->origin=IPL_ORIGIN_BL)
		if (frame->origin == IPL_ORIGIN_TL){
			cvCopy(frame, frame_copy, 0);
		}
		else{
			//flip_mode = 0 沿X-轴翻转, flip_mode > 0 (如 1) 沿Y-轴翻转， flip_mode < 0 (如 -1) 沿X-轴和Y-轴翻转.见下面的公式
			//函数cvFlip 以三种方式之一翻转数组 (行和列下标是以0为基点的):
			cvFlip(frame, frame_copy, 0);//反转图像
		}

		//循环从摄像头读出图片进行检测
		while (1)
		{
			//detect_and_draw(img, sc_WeakClassifier_set, weight_set, Threshold, scalefstep, slidefstep); // 检测并且标识人脸
			//从摄像头或者视频文件中抓取帧
			//函数cvQueryFrame从摄像头或者文件中抓取一帧，然后解压并返回这一帧。
			//这个函数仅仅是函数cvGrabFrame和函数cvRetrieveFrame在一起调用的组合。返回的图像不可以被用户释放或者修改。
			if (!cvGrabFrame(capture)){
				break;
			}
			frame = cvRetrieveFrame(capture); //获得由cvGrabFrame函数抓取的图片
			if (!frame){ break; }
		//	if (!frame_copy){
		//		frame_copy = cvCreateImage(cvSize(frame->width, frame->height), IPL_DEPTH_8U, frame->nChannels);
		//	}
			//图像原点或者是左上角 (img->origin=IPL_ORIGIN_TL)或者是左下角(img->origin=IPL_ORIGIN_BL)
			if (frame->origin == IPL_ORIGIN_TL){
				cvCopy(frame, frame_copy, 0);
			}
			else{
				//flip_mode = 0 沿X-轴翻转, flip_mode > 0 (如 1) 沿Y-轴翻转， flip_mode < 0 (如 -1) 沿X-轴和Y-轴翻转.见下面的公式
				//函数cvFlip 以三种方式之一翻转数组 (行和列下标是以0为基点的):
				cvFlip(frame, frame_copy, 0);//反转图像
			}

			Detect_And_Draw(frame_copy, load_img,  weight_set, Threshold); // 检测并且标识人脸

			if (cvWaitKey(10) >= 0)
				break;

		}

		
		//释放指针
		//cvReleaseImage(&imgg);
		cvReleaseImage(&frame_copy);
		cvReleaseImage(&load_img);
		cvReleaseCapture(&capture);
	}
	
	//cvClearMemStorage(storage);//释放内存块
	
	//cvDestroyWindow("人脸识别");
	
	system("pause");
	return 0;
}






inline int get_para(int i)
{
	int t;
	switch (i)
	{
	case 0:
		t = pBuffer[105] - pBuffer[104] - (pBuffer[104] - pBuffer[103]);
		break;
	case 1:
		t = pBuffer[16] - pBuffer[8] - (pBuffer[8] - pBuffer[0]);
		break;
	case 2:
		t = pBuffer[166] - pBuffer[71] - (pBuffer[261] - pBuffer[166]);
		break;
	case 3:
		t = (pBuffer[184] - pBuffer[180] - pBuffer[127] + pBuffer[123] + pBuffer[298] - pBuffer[294] - pBuffer[241] + pBuffer[237] - 2 * (pBuffer[241] - pBuffer[237] - pBuffer[184] + pBuffer[180])) / 2;
		break;
	case 4:
		t = pBuffer[134] - pBuffer[58] - (pBuffer[210] - pBuffer[134]);
		break;
	case 5:
		t = (pBuffer[195] - pBuffer[194] - pBuffer[138] + pBuffer[137] + pBuffer[309] - pBuffer[308] - pBuffer[252] + pBuffer[251] - 2 * (pBuffer[252] - pBuffer[251] - pBuffer[195] + pBuffer[194])) / 2;
		break;
	case 6:
		t = (pBuffer[316] - pBuffer[313] - pBuffer[297] + pBuffer[294] + pBuffer[354] - pBuffer[351] - pBuffer[335] + pBuffer[332] - 2 * (pBuffer[335] - pBuffer[332] - pBuffer[316] + pBuffer[313])) / 2;
		break;
	case 7:
		t = pBuffer[105] - pBuffer[104] - (pBuffer[104] - pBuffer[103]);
		break;
	case 8:
		t = (pBuffer[159] - pBuffer[155] + pBuffer[165] - pBuffer[161] - 4 * (pBuffer[161] - pBuffer[159])) / 4;
		break;
	case 9:
		t = (pBuffer[313] - pBuffer[294] + pBuffer[351] - pBuffer[332] - 2 * (pBuffer[332] - pBuffer[313])) / 2;
		break;
	case 10:
		t = (pBuffer[69] - pBuffer[12] + pBuffer[183] - pBuffer[126] - 2 * (pBuffer[126] - pBuffer[69])) / 2;
		break;
	case 11:
		t = (pBuffer[314] - pBuffer[295] + pBuffer[352] - pBuffer[333] - 2 * (pBuffer[333] - pBuffer[314])) / 2;
		break;
	case 12:
		t = pBuffer[225] - pBuffer[221] - (pBuffer[221] - pBuffer[217]);
		break;
	case 13:
		t = pBuffer[355] - pBuffer[348] - pBuffer[70] + pBuffer[63] - (pBuffer[348] - pBuffer[63]);
		break;
	case 14:
		t = (pBuffer[53] - pBuffer[34] + pBuffer[91] - pBuffer[72] - 2 * (pBuffer[72] - pBuffer[53])) / 2;
		break;
	case 15:
		t = pBuffer[317] - pBuffer[310] - pBuffer[127] + pBuffer[120] - (pBuffer[310] - pBuffer[120]);
		break;
	case 16:
		t = (pBuffer[79] - pBuffer[78] - pBuffer[60] + pBuffer[59] + pBuffer[97] - pBuffer[96] - pBuffer[78] + pBuffer[77] - (pBuffer[98] - pBuffer[97] - pBuffer[79] + pBuffer[78]) - (pBuffer[78] - pBuffer[77] - pBuffer[59] + pBuffer[58])) / 2;
		break;
	case 17:
		t = pBuffer[322] - pBuffer[321] - (pBuffer[321] - pBuffer[320]);
		break;
	case 18:
		t = pBuffer[169] - pBuffer[154] - pBuffer[17] + pBuffer[2] - (pBuffer[321] - pBuffer[306] - pBuffer[169] + pBuffer[154]);
		break;
	case 19:
		t = pBuffer[35] - pBuffer[16] - (pBuffer[54] - pBuffer[35]);
		break;
	case 20:
		t = pBuffer[322] - pBuffer[284] - (pBuffer[360] - pBuffer[322]);
		break;
	case 21:
		t = (pBuffer[42] - pBuffer[40] + pBuffer[50] - pBuffer[48] - 2 / 3 * (pBuffer[48] - pBuffer[42]))*1.5;
		break;
	case 22:
		t = (pBuffer[53] - pBuffer[38] - pBuffer[34] + pBuffer[19] + pBuffer[91] - pBuffer[76] - pBuffer[72] + pBuffer[57] - 2 * (pBuffer[72] - pBuffer[57] - pBuffer[53] + pBuffer[38])) / 2;
		break;
	case 23:
		t = (pBuffer[228] - pBuffer[0] + pBuffer[232] - pBuffer[231] - pBuffer[4] + pBuffer[3] - 2 / 3 * (pBuffer[231] - pBuffer[228] - pBuffer[3] + pBuffer[0]))*1.5;
		break;
	case 24:
		t = pBuffer[321] - pBuffer[316] - pBuffer[302] + pBuffer[297] - (pBuffer[340] - pBuffer[335] - pBuffer[321] + pBuffer[316]);
		break;
	case 25:
		t = (pBuffer[315] - pBuffer[296] + pBuffer[353] - pBuffer[334] - 2 * (pBuffer[334] - pBuffer[315])) / 2;
		break;
	case 26:
		t = (pBuffer[190] - pBuffer[152] + pBuffer[266] - pBuffer[228] - 2 * (pBuffer[228] - pBuffer[190])) / 2;
		break;
	case 27:
		t = pBuffer[34] - pBuffer[15] - (pBuffer[53] - pBuffer[34]);
		break;
	case 28:
		t = pBuffer[34] - pBuffer[29] - pBuffer[15] + pBuffer[10] - (pBuffer[53] - pBuffer[48] - pBuffer[34] + pBuffer[29]);
		break;
	case 29:
		t = (pBuffer[284] - pBuffer[265] + pBuffer[360] - pBuffer[341] - 2 / 3 * (pBuffer[341] - pBuffer[284]))*1.5;
		break;
	case 30:
		t = pBuffer[290] - pBuffer[289] - pBuffer[271] + pBuffer[270] - (pBuffer[289] - pBuffer[288] - pBuffer[270] + pBuffer[269]);
		break;
	case 31:
		t = pBuffer[105] - pBuffer[104] - (pBuffer[104] - pBuffer[103]);
		break;
	case 32:
		t = pBuffer[357] - pBuffer[354] - pBuffer[110] + pBuffer[107] - (pBuffer[354] - pBuffer[351] - pBuffer[107] + pBuffer[104]);
		break;
	case 33:
		t = (pBuffer[41] - pBuffer[22] + pBuffer[79] - pBuffer[60] - 2 * (pBuffer[60] - pBuffer[41])) / 2;
		break;
	case 34:
		t = (pBuffer[248] - pBuffer[77] + pBuffer[251] - pBuffer[249] - pBuffer[80] + pBuffer[78] - 4 * (pBuffer[249] - pBuffer[248] - pBuffer[78] + pBuffer[77])) / 4;
		break;
	case 35:
		t = (pBuffer[303] - pBuffer[292] - pBuffer[284] + pBuffer[273] + pBuffer[341] - pBuffer[330] - pBuffer[322] + pBuffer[311] - 2 * (pBuffer[322] - pBuffer[311] - pBuffer[303] + pBuffer[292])) / 2;
		break;
	case 36:
		t = pBuffer[226] - pBuffer[225] - pBuffer[150] + pBuffer[149] - (pBuffer[225] - pBuffer[224] - pBuffer[149] + pBuffer[148]);
		break;
	case 37:
		t = (pBuffer[292] - pBuffer[273] + pBuffer[330] - pBuffer[311] - 2 * (pBuffer[311] - pBuffer[292])) / 2;
		break;
	case 38:
		t = (pBuffer[225] - pBuffer[212] - pBuffer[149] + pBuffer[136] + pBuffer[339] - pBuffer[326] - pBuffer[263] + pBuffer[250] - 4 * (pBuffer[263] - pBuffer[250] - pBuffer[225] + pBuffer[212])) / 4;
		break;
	case 39:
		t = pBuffer[100] - pBuffer[24] - (pBuffer[176] - pBuffer[100]);
		break;
	case 40:
		t = (pBuffer[84] - pBuffer[46] + pBuffer[160] - pBuffer[122] - 2 * (pBuffer[122] - pBuffer[84])) / 2;
		break;
	case 41:
		t = pBuffer[32] - pBuffer[13] - (pBuffer[51] - pBuffer[32]);
		break;
	case 42:
		t = (pBuffer[33] - pBuffer[32] - pBuffer[14] + pBuffer[13] + pBuffer[51] - pBuffer[50] - pBuffer[32] + pBuffer[31] - (pBuffer[52] - pBuffer[51] - pBuffer[33] + pBuffer[32]) - (pBuffer[32] - pBuffer[31] - pBuffer[13] + pBuffer[12])) / 2;
		break;
	case 43:
		t = pBuffer[256] - pBuffer[253] - pBuffer[161] + pBuffer[158] - (pBuffer[253] - pBuffer[250] - pBuffer[158] + pBuffer[155]);
		break;
	case 44:
		t = pBuffer[30] - pBuffer[28] - pBuffer[11] + pBuffer[9] - (pBuffer[49] - pBuffer[47] - pBuffer[30] + pBuffer[28]);
		break;
	case 45:
		t = (pBuffer[219] - pBuffer[200] + pBuffer[257] - pBuffer[238] - 2 * (pBuffer[238] - pBuffer[219])) / 2;
		break;
	case 46:
		t = pBuffer[10] - pBuffer[9] - (pBuffer[9] - pBuffer[8]);
		break;
	case 47:
		t = (pBuffer[198] - pBuffer[197] - pBuffer[8] + pBuffer[7] + pBuffer[202] - pBuffer[201] - pBuffer[12] + pBuffer[11] - 2 / 3 * (pBuffer[201] - pBuffer[198] - pBuffer[11] + pBuffer[8]))*1.5;
		break;
	case 48:
		t = pBuffer[229] - pBuffer[228] - pBuffer[210] + pBuffer[209] - (pBuffer[248] - pBuffer[247] - pBuffer[229] + pBuffer[228]);
		break;
	case 49:
		t = pBuffer[150] - pBuffer[149] - pBuffer[131] + pBuffer[130] - (pBuffer[149] - pBuffer[148] - pBuffer[130] + pBuffer[129]);
		break;
	case 50:
		t = pBuffer[148] - pBuffer[147] - pBuffer[110] + pBuffer[109] - (pBuffer[147] - pBuffer[146] - pBuffer[109] + pBuffer[108]);
		break;
	case 51:
		t = (pBuffer[199] - pBuffer[180] + pBuffer[237] - pBuffer[218] - 2 * (pBuffer[218] - pBuffer[199])) / 2;
		break;
	case 52:
		t = pBuffer[246] - pBuffer[189] - (pBuffer[303] - pBuffer[246]);
		break;
	case 53:
		t = (pBuffer[217] - pBuffer[179] + pBuffer[274] - pBuffer[236] - 4 * (pBuffer[236] - pBuffer[217])) / 4;
		break;
	case 54:
		t = (pBuffer[266] - pBuffer[171] + pBuffer[270] - pBuffer[269] - pBuffer[175] + pBuffer[174] - 2 / 3 * (pBuffer[269] - pBuffer[266] - pBuffer[174] + pBuffer[171]))*1.5;
		break;
	case 55:
		t = (pBuffer[26] + pBuffer[83] - pBuffer[45] - 4 * (pBuffer[45] - pBuffer[26])) / 4;
		break;
	case 56:
		t = (pBuffer[200] - pBuffer[193] - pBuffer[162] + pBuffer[155] + pBuffer[276] - pBuffer[269] - pBuffer[238] + pBuffer[231] - 2 * (pBuffer[238] - pBuffer[231] - pBuffer[200] + pBuffer[193])) / 2;
		break;
	case 57:
		t = (pBuffer[177] - pBuffer[176] - pBuffer[158] + pBuffer[157] + pBuffer[195] - pBuffer[194] - pBuffer[176] + pBuffer[175] - (pBuffer[196] - pBuffer[195] - pBuffer[177] + pBuffer[176]) - (pBuffer[176] - pBuffer[175] - pBuffer[157] + pBuffer[156])) / 2;
		break;
	case 58:
		t = pBuffer[157] - pBuffer[155] - pBuffer[24] + pBuffer[22] - (pBuffer[155] - pBuffer[153] - pBuffer[22] + pBuffer[20]);
		break;
	case 59:
		t = pBuffer[33] - pBuffer[14] - (pBuffer[52] - pBuffer[33]);
		break;
	case 60:
		t = (pBuffer[167] - pBuffer[156] - pBuffer[129] + pBuffer[118] + pBuffer[319] - pBuffer[308] - pBuffer[281] + pBuffer[270] - 2 / 3 * (pBuffer[281] - pBuffer[270] - pBuffer[167] + pBuffer[156]))*1.5;
		break;
	case 61:
		t = pBuffer[140] - pBuffer[139] - pBuffer[83] + pBuffer[82] - (pBuffer[139] - pBuffer[138] - pBuffer[82] + pBuffer[81]);
		break;
	case 62:
		t = pBuffer[262] - pBuffer[251] - pBuffer[205] + pBuffer[194] - (pBuffer[319] - pBuffer[308] - pBuffer[262] + pBuffer[251]);
		break;
	case 63:
		t = (pBuffer[282] - pBuffer[278] - pBuffer[263] + pBuffer[259] + pBuffer[320] - pBuffer[316] - pBuffer[301] + pBuffer[297] - 2 * (pBuffer[301] - pBuffer[297] - pBuffer[282] + pBuffer[278])) / 2;
		break;
	case 64:
		t = pBuffer[284] - pBuffer[279] - pBuffer[265] + pBuffer[260] - (pBuffer[303] - pBuffer[298] - pBuffer[284] + pBuffer[279]);
		break;
	case 65:
		t = (pBuffer[25] - pBuffer[23] + pBuffer[28] - pBuffer[26] - 4 * (pBuffer[26] - pBuffer[25])) / 4;
		break;
	case 66:
		t = pBuffer[204] - pBuffer[201] - (pBuffer[201] - pBuffer[198]);
		break;
	case 67:
		t = (pBuffer[74] - pBuffer[17] + pBuffer[188] - pBuffer[131] - 2 * (pBuffer[131] - pBuffer[74])) / 2;
		break;
	case 68:
		t = pBuffer[265] - pBuffer[264] - (pBuffer[264] - pBuffer[263]);
		break;
	case 69:
		t = (pBuffer[345] - pBuffer[344] - pBuffer[212] + pBuffer[211] + pBuffer[347] - pBuffer[346] - pBuffer[214] + pBuffer[213] - 2 * (pBuffer[346] - pBuffer[345] - pBuffer[213] + pBuffer[212])) / 2;
		break;
	case 70:
		t = pBuffer[169] - pBuffer[162] - (pBuffer[162] - pBuffer[155]);
		break;
	case 71:
		t = pBuffer[59] - pBuffer[58] - pBuffer[40] + pBuffer[39] - (pBuffer[58] - pBuffer[57] - pBuffer[39] + pBuffer[38]);
		break;
	case 72:
		t = (pBuffer[284] - pBuffer[268] - pBuffer[246] + pBuffer[230] + pBuffer[360] - pBuffer[344] - pBuffer[322] + pBuffer[306] - 2 * (pBuffer[322] - pBuffer[306] - pBuffer[284] + pBuffer[268])) / 2;
		break;
	case 73:
		t = (pBuffer[293] - pBuffer[274] + pBuffer[331] - pBuffer[312] - 2 * (pBuffer[312] - pBuffer[293])) / 2;
		break;
	case 74:
		t = pBuffer[245] - pBuffer[229] - pBuffer[207] + pBuffer[191] - (pBuffer[283] - pBuffer[267] - pBuffer[245] + pBuffer[229]);
		break;
	case 75:
		t = pBuffer[311] - pBuffer[292] - (pBuffer[330] - pBuffer[311]);
		break;
	case 76:
		t = (pBuffer[246] - pBuffer[230] - pBuffer[208] + pBuffer[192] + pBuffer[303] - pBuffer[287] - pBuffer[265] + pBuffer[249] - 4 * (pBuffer[265] - pBuffer[249] - pBuffer[246] + pBuffer[230])) / 4;
		break;
	case 77:
		t = pBuffer[99] - pBuffer[23] - (pBuffer[175] - pBuffer[99]);
		break;
	case 78:
		t = (pBuffer[45] - pBuffer[7] + pBuffer[121] - pBuffer[83] - 2 * (pBuffer[83] - pBuffer[45])) / 2;
		break;
	case 79:
		t = pBuffer[188] - pBuffer[169] - (pBuffer[207] - pBuffer[188]);
		break;
	case 80:
		t = (pBuffer[208] - pBuffer[205] - pBuffer[189] + pBuffer[186] + pBuffer[284] - pBuffer[281] - pBuffer[265] + pBuffer[262] - 2 / 3 * (pBuffer[265] - pBuffer[262] - pBuffer[208] + pBuffer[205]))*1.5;
		break;
	case 81:
		t = (pBuffer[319] - pBuffer[311] - pBuffer[300] + pBuffer[292] + pBuffer[330] - pBuffer[311] - (pBuffer[338] - pBuffer[330] - pBuffer[319] + pBuffer[311]) - (pBuffer[311] - pBuffer[292])) / 2;
		break;
	case 82:
		t = pBuffer[185] - pBuffer[173] - pBuffer[14] + pBuffer[2] - (pBuffer[356] - pBuffer[344] - pBuffer[185] + pBuffer[173]);
		break;
	case 83:
		t = (pBuffer[22] - pBuffer[3] + pBuffer[60] - pBuffer[41] - 2 * (pBuffer[41] - pBuffer[22])) / 2;
		break;
	case 84:
		t = pBuffer[214] - pBuffer[213] - pBuffer[81] + pBuffer[80] - (pBuffer[213] - pBuffer[212] - pBuffer[80] + pBuffer[79]);
		break;
	case 85:
		t = pBuffer[82] - pBuffer[79] - pBuffer[25] + pBuffer[22] - 9 * (pBuffer[62] - pBuffer[61] - pBuffer[43] + pBuffer[42]) / 8;
		break;
	case 86:
		t = pBuffer[241] - pBuffer[127] - (pBuffer[355] - pBuffer[241]);
		break;
	case 87:
		t = pBuffer[355] - pBuffer[354] - (pBuffer[354] - pBuffer[353]);
		break;
	case 88:
		t = pBuffer[360] - pBuffer[351] - (pBuffer[351] - pBuffer[342]);
		break;
	case 89:
		t = (pBuffer[280] - pBuffer[279] - pBuffer[261] + pBuffer[260] + pBuffer[282] - pBuffer[281] - pBuffer[263] + pBuffer[262] - 2 * (pBuffer[281] - pBuffer[280] - pBuffer[262] + pBuffer[261])) / 2;
		break;
	case 90:
		t = pBuffer[73] - pBuffer[71] - pBuffer[54] + pBuffer[52] - (pBuffer[92] - pBuffer[90] - pBuffer[73] + pBuffer[71]);
		break;
	case 91:
		t = pBuffer[235] - pBuffer[234] - pBuffer[178] + pBuffer[177] - (pBuffer[234] - pBuffer[233] - pBuffer[177] + pBuffer[176]);
		break;
	case 92:
		t = (pBuffer[80] - pBuffer[42] + pBuffer[156] - pBuffer[118] - 2 * (pBuffer[118] - pBuffer[80])) / 2;
		break;
	case 93:
		t = pBuffer[98] - pBuffer[3] - (pBuffer[193] - pBuffer[98]);
		break;
	case 94:
		t = (pBuffer[151] - pBuffer[135] - pBuffer[94] + pBuffer[78] + pBuffer[265] - pBuffer[249] - pBuffer[208] + pBuffer[192] - 2 * (pBuffer[208] - pBuffer[192] - pBuffer[151] + pBuffer[135])) / 2;
		break;
	case 95:
		t = pBuffer[135] - pBuffer[134] - pBuffer[116] + pBuffer[115] - (pBuffer[154] - pBuffer[153] - pBuffer[135] + pBuffer[134]);
		break;
	case 96:
		t = pBuffer[201] - pBuffer[182] - (pBuffer[220] - pBuffer[201]);
		break;
	case 97:
		t = pBuffer[223] - pBuffer[204] - (pBuffer[242] - pBuffer[223]);
		break;
	case 98:
		t = pBuffer[204] - pBuffer[202] - pBuffer[128] + pBuffer[126] - (pBuffer[280] - pBuffer[278] - pBuffer[204] + pBuffer[202]);
		break;
	case 99:
		t = pBuffer[21] - pBuffer[20] - pBuffer[2] + pBuffer[1] - (pBuffer[20] - pBuffer[19] - pBuffer[1] + pBuffer[0]);
		break;
	case 100:
		t = pBuffer[122] - pBuffer[121] - pBuffer[103] + pBuffer[102] - (pBuffer[121] - pBuffer[120] - pBuffer[102] + pBuffer[101]);
		break;
	case 101:
		t = pBuffer[199] - pBuffer[193] - pBuffer[9] + pBuffer[3] - 9 * (pBuffer[121] - pBuffer[119] - pBuffer[64] + pBuffer[62]) / 8;
		break;
	case 102:
		t = (pBuffer[304] - pBuffer[133] + pBuffer[306] - pBuffer[305] - pBuffer[135] + pBuffer[134] - 2 * (pBuffer[305] - pBuffer[304] - pBuffer[134] + pBuffer[133])) / 2;
		break;
	case 103:
		t = pBuffer[283] - pBuffer[266] - pBuffer[264] + pBuffer[247] - (pBuffer[302] - pBuffer[285] - pBuffer[283] + pBuffer[266]);
		break;
	case 104:
		t = (pBuffer[227] - pBuffer[208] + pBuffer[265] - pBuffer[246] - 2 * (pBuffer[246] - pBuffer[227])) / 2;
		break;
	case 105:
		t = pBuffer[40] - pBuffer[2] - (pBuffer[78] - pBuffer[40]);
		break;
	case 106:
		t = (pBuffer[229] - pBuffer[20] + pBuffer[237] - pBuffer[235] - pBuffer[28] + pBuffer[26] - 2 / 3 * (pBuffer[235] - pBuffer[229] - pBuffer[26] + pBuffer[20]))*1.5;
		break;
	case 107:
		t = (pBuffer[322] - pBuffer[316] - pBuffer[303] + pBuffer[297] + pBuffer[360] - pBuffer[354] - pBuffer[341] + pBuffer[335] - 2 * (pBuffer[341] - pBuffer[335] - pBuffer[322] + pBuffer[316])) / 2;
		break;
	case 108:
		t = (pBuffer[206] - pBuffer[195] - pBuffer[130] + pBuffer[119] + pBuffer[358] - pBuffer[347] - pBuffer[282] + pBuffer[271] - 2 * (pBuffer[282] - pBuffer[271] - pBuffer[206] + pBuffer[195])) / 2;
		break;
	case 109:
		t = pBuffer[67] - pBuffer[66] - (pBuffer[66] - pBuffer[65]);
		break;
	case 110:
		t = pBuffer[324] - pBuffer[323] - (pBuffer[323]);
		break;
	case 111:
		t = (pBuffer[81] - pBuffer[78] - pBuffer[5] + pBuffer[2] + pBuffer[93] - pBuffer[90] - pBuffer[17] + pBuffer[14] - 2 / 3 * (pBuffer[90] - pBuffer[81] - pBuffer[14] + pBuffer[5]))*1.5;
		break;
	case 112:
		t = pBuffer[229] - pBuffer[228] - pBuffer[191] + pBuffer[190] - (pBuffer[228] - pBuffer[190]);
		break;
	case 113:
		t = (pBuffer[295] - pBuffer[276] + pBuffer[333] - pBuffer[314] - 2 * (pBuffer[314] - pBuffer[295])) / 2;
		break;
	case 114:
		t = pBuffer[109] - pBuffer[14] - (pBuffer[204] - pBuffer[109]);
		break;
	case 115:
		t = pBuffer[113] - pBuffer[111] - pBuffer[18] + pBuffer[16] - (pBuffer[111] - pBuffer[109] - pBuffer[16] + pBuffer[14]);
		break;
	case 116:
		t = (pBuffer[314] - pBuffer[309] - pBuffer[295] + pBuffer[290] + pBuffer[352] - pBuffer[347] - pBuffer[333] + pBuffer[328] - 2 * (pBuffer[333] - pBuffer[328] - pBuffer[314] + pBuffer[309])) / 2;
		break;
	case 117:
		t = pBuffer[118] - pBuffer[117] - pBuffer[4] + pBuffer[3] - (pBuffer[232] - pBuffer[231] - pBuffer[118] + pBuffer[117]);
		break;
	case 118:
		t = (pBuffer[137] - pBuffer[42] + pBuffer[147] - pBuffer[142] - pBuffer[52] + pBuffer[47] - 2 * (pBuffer[142] - pBuffer[137] - pBuffer[47] + pBuffer[42])) / 2;
		break;
	case 119:
		t = pBuffer[259] - pBuffer[164] - (pBuffer[354] - pBuffer[259]);
		break;
	case 120:
		t = (pBuffer[258] - pBuffer[239] + pBuffer[296] - pBuffer[277] - 2 * (pBuffer[277] - pBuffer[258])) / 2;
		break;
	case 121:
		t = pBuffer[284] - pBuffer[266] - pBuffer[246] + pBuffer[228] - (pBuffer[322] - pBuffer[304] - pBuffer[284] + pBuffer[266]);
		break;
	case 122:
		t = pBuffer[187] - pBuffer[183] - (pBuffer[183] - pBuffer[179]);
		break;
	case 123:
		t = pBuffer[340] - pBuffer[330] - pBuffer[321] + pBuffer[311] - (pBuffer[359] - pBuffer[349] - pBuffer[340] + pBuffer[330]);
		break;
	case 124:
		t = pBuffer[204] - pBuffer[196] - pBuffer[71] + pBuffer[63] - (pBuffer[337] - pBuffer[329] - pBuffer[204] + pBuffer[196]);
		break;
	case 125:
		t = pBuffer[255] - pBuffer[236] - (pBuffer[274] - pBuffer[255]);
		break;
	case 126:
		t = pBuffer[255] - pBuffer[251] - pBuffer[236] + pBuffer[232] - (pBuffer[274] - pBuffer[270] - pBuffer[255] + pBuffer[251]);
		break;
	case 127:
		t = (pBuffer[125] - pBuffer[123] - pBuffer[106] + pBuffer[104] + pBuffer[163] - pBuffer[161] - pBuffer[144] + pBuffer[142] - 2 * (pBuffer[144] - pBuffer[142] - pBuffer[125] + pBuffer[123])) / 2;
		break;
	case 128:
		t = (pBuffer[291] - pBuffer[290] - pBuffer[253] + pBuffer[252] + pBuffer[328] - pBuffer[327] - pBuffer[290] + pBuffer[289] - (pBuffer[329] - pBuffer[328] - pBuffer[291] + pBuffer[290]) - (pBuffer[290] - pBuffer[289] - pBuffer[252] + pBuffer[251])) / 2;
		break;
	case 129:
		t = pBuffer[2] - pBuffer[1] - (pBuffer[1] - pBuffer[0]);
		break;
	case 130:
		t = pBuffer[250] - pBuffer[248] - pBuffer[3] + pBuffer[1] - (pBuffer[248] - pBuffer[1]);
		break;
	case 131:
		t = pBuffer[305] - pBuffer[304] - (pBuffer[304]);
		break;
	case 132:
		t = (pBuffer[87] - pBuffer[49] + pBuffer[163] - pBuffer[125] - 2 * (pBuffer[125] - pBuffer[87])) / 2;
		break;
	case 133:
		t = (pBuffer[50] - pBuffer[12] + pBuffer[126] - pBuffer[88] - 2 * (pBuffer[88] - pBuffer[50])) / 2;
		break;
	case 134:
		t = (pBuffer[52] - pBuffer[14] + pBuffer[109] - pBuffer[71] - 4 * (pBuffer[71] - pBuffer[52])) / 4;
		break;
	case 135:
		t = (pBuffer[105] - pBuffer[48] + pBuffer[219] - pBuffer[162] - 2 * (pBuffer[162] - pBuffer[105])) / 2;
		break;
	case 136:
		t = (pBuffer[59] + pBuffer[173] - pBuffer[97] - 4 * (pBuffer[97] - pBuffer[59])) / 4;
		break;
	case 137:
		t = (pBuffer[154] + pBuffer[160] - pBuffer[157] - 2 * (pBuffer[157] - pBuffer[154])) / 2;
		break;
	case 138:
		t = pBuffer[237] - pBuffer[218] - (pBuffer[256] - pBuffer[237]);
		break;
	case 139:
		t = (pBuffer[125] - pBuffer[123] - pBuffer[49] + pBuffer[47] + pBuffer[277] - pBuffer[275] - pBuffer[201] + pBuffer[199] - 2 * (pBuffer[201] - pBuffer[199] - pBuffer[125] + pBuffer[123])) / 2;
		break;
	case 140:
		t = pBuffer[275] - pBuffer[274] - pBuffer[85] + pBuffer[84] - (pBuffer[274] - pBuffer[273] - pBuffer[84] + pBuffer[83]);
		break;
	case 141:
		t = pBuffer[340] - pBuffer[339] - (pBuffer[339] - pBuffer[338]);
		break;
	case 142:
		t = pBuffer[311] - pBuffer[310] - pBuffer[64] + pBuffer[63] - (pBuffer[310] - pBuffer[309] - pBuffer[63] + pBuffer[62]);
		break;
	case 143:
		t = (pBuffer[244] - pBuffer[238] - pBuffer[225] + pBuffer[219] + pBuffer[282] - pBuffer[276] - pBuffer[263] + pBuffer[257] - 2 * (pBuffer[263] - pBuffer[257] - pBuffer[244] + pBuffer[238])) / 2;
		break;
	case 144:
		t = pBuffer[168] - pBuffer[165] - (pBuffer[165] - pBuffer[162]);
		break;
	case 145:
		t = pBuffer[166] - pBuffer[165] - pBuffer[71] + pBuffer[70] - (pBuffer[165] - pBuffer[164] - pBuffer[70] + pBuffer[69]);
		break;
	case 146:
		t = (pBuffer[42] - pBuffer[4] + pBuffer[99] - pBuffer[61] - 4 * (pBuffer[61] - pBuffer[42])) / 4;
		break;
	case 147:
		t = (pBuffer[153] - pBuffer[115] + pBuffer[305] - pBuffer[267] - 2 / 3 * (pBuffer[267] - pBuffer[153]))*1.5;
		break;
	case 148:
		t = (pBuffer[265] - pBuffer[254] - pBuffer[246] + pBuffer[235] + pBuffer[303] - pBuffer[292] - pBuffer[284] + pBuffer[273] - 2 * (pBuffer[284] - pBuffer[273] - pBuffer[265] + pBuffer[254])) / 2;
		break;
	case 149:
		t = (pBuffer[147] - pBuffer[146] - pBuffer[33] + pBuffer[32] + pBuffer[149] - pBuffer[148] - pBuffer[35] + pBuffer[34] - 2 * (pBuffer[148] - pBuffer[147] - pBuffer[34] + pBuffer[33])) / 2;
		break;
	case 150:
		t = (pBuffer[52] - pBuffer[33] + pBuffer[90] - pBuffer[71] - 2 * (pBuffer[71] - pBuffer[52])) / 2;
		break;
	case 151:
		t = pBuffer[243] - pBuffer[242] - pBuffer[224] + pBuffer[223] - (pBuffer[262] - pBuffer[261] - pBuffer[243] + pBuffer[242]);
		break;
	case 152:
		t = pBuffer[75] - pBuffer[68] - (pBuffer[68] - pBuffer[61]);
		break;
	case 153:
		t = pBuffer[337] - pBuffer[333] - pBuffer[318] + pBuffer[314] - (pBuffer[356] - pBuffer[352] - pBuffer[337] + pBuffer[333]);
		break;
	case 154:
		t = (pBuffer[185] - pBuffer[176] - pBuffer[71] + pBuffer[62] + pBuffer[356] - pBuffer[347] - pBuffer[242] + pBuffer[233] - 4 * (pBuffer[242] - pBuffer[233] - pBuffer[185] + pBuffer[176])) / 4;
		break;
	case 155:
		t = pBuffer[69] - pBuffer[68] - (pBuffer[68] - pBuffer[67]);
		break;
	case 156:
		t = (pBuffer[175] - pBuffer[174] - pBuffer[118] + pBuffer[117] + pBuffer[177] - pBuffer[176] - pBuffer[120] + pBuffer[119] - 2 * (pBuffer[176] - pBuffer[175] - pBuffer[119] + pBuffer[118])) / 2;
		break;
	case 157:
		t = pBuffer[215] - pBuffer[214] - pBuffer[196] + pBuffer[195] - (pBuffer[214] - pBuffer[213] - pBuffer[195] + pBuffer[194]);
		break;
	case 158:
		t = pBuffer[36] - pBuffer[26] - pBuffer[17] + pBuffer[7] - (pBuffer[55] - pBuffer[45] - pBuffer[36] + pBuffer[26]);
		break;
	case 159:
		t = pBuffer[121] - pBuffer[115] - pBuffer[26] + pBuffer[20] - (pBuffer[216] - pBuffer[210] - pBuffer[121] + pBuffer[115]);
		break;
	case 160:
		t = (pBuffer[127] - pBuffer[32] + pBuffer[317] - pBuffer[222] - 2 * (pBuffer[222] - pBuffer[127])) / 2;
		break;
	case 161:
		t = pBuffer[176] - pBuffer[175] - pBuffer[157] + pBuffer[156] - (pBuffer[175] - pBuffer[174] - pBuffer[156] + pBuffer[155]);
		break;
	case 162:
		t = (pBuffer[186] - pBuffer[176] - pBuffer[110] + pBuffer[100] + pBuffer[338] - pBuffer[328] - pBuffer[262] + pBuffer[252] - 2 * (pBuffer[262] - pBuffer[252] - pBuffer[186] + pBuffer[176])) / 2;
		break;
	case 163:
		t = pBuffer[72] - pBuffer[53] - (pBuffer[91] - pBuffer[72]);
		break;
	case 164:
		t = pBuffer[148] - pBuffer[137] - pBuffer[110] + pBuffer[99] - (pBuffer[186] - pBuffer[175] - pBuffer[148] + pBuffer[137]);
		break;
	case 165:
		t = (pBuffer[314] - pBuffer[310] - pBuffer[219] + pBuffer[215] + pBuffer[322] - pBuffer[318] - pBuffer[227] + pBuffer[223] - 2 * (pBuffer[318] - pBuffer[314] - pBuffer[223] + pBuffer[219])) / 2;
		break;
	case 166:
		t = (pBuffer[15] - pBuffer[13] + pBuffer[91] - pBuffer[89] - pBuffer[72] + pBuffer[70] - 2 / 3 * (pBuffer[72] - pBuffer[70] - pBuffer[15] + pBuffer[13]))*1.5;
		break;
	case 167:
		t = pBuffer[224] - pBuffer[223] - pBuffer[205] + pBuffer[204] - (pBuffer[243] - pBuffer[242] - pBuffer[224] + pBuffer[223]);
		break;
	case 168:
		t = (pBuffer[200] - pBuffer[196] + pBuffer[206] - pBuffer[202] - 4 * (pBuffer[202] - pBuffer[200])) / 4;
		break;
	case 169:
		t = pBuffer[137] - pBuffer[136] - pBuffer[118] + pBuffer[117] - (pBuffer[136] - pBuffer[135] - pBuffer[117] + pBuffer[116]);
		break;
	case 170:
		t = (pBuffer[51] - pBuffer[41] - pBuffer[32] + pBuffer[22] + pBuffer[127] - pBuffer[117] - pBuffer[108] + pBuffer[98] - 2 / 3 * (pBuffer[108] - pBuffer[98] - pBuffer[51] + pBuffer[41]))*1.5;
		break;
	case 171:
		t = (pBuffer[33] - pBuffer[19] + pBuffer[90] - pBuffer[76] - pBuffer[52] + pBuffer[38] - 4 * (pBuffer[52] - pBuffer[38] - pBuffer[33] + pBuffer[19])) / 4;
		break;
	case 172:
		t = pBuffer[252] - pBuffer[248] - pBuffer[62] + pBuffer[58] - 9 * (pBuffer[174] - pBuffer[173] - pBuffer[117] + pBuffer[116]) / 8;
		break;
	case 173:
		t = pBuffer[313] - pBuffer[312] - pBuffer[294] + pBuffer[293] - (pBuffer[312] - pBuffer[311] - pBuffer[293] + pBuffer[292]);
		break;
	case 174:
		t = pBuffer[30] - pBuffer[29] - pBuffer[11] + pBuffer[10] - (pBuffer[29] - pBuffer[28] - pBuffer[10] + pBuffer[9]);
		break;
	case 175:
		t = pBuffer[20] - pBuffer[19] - pBuffer[1] + pBuffer[0] - (pBuffer[39] - pBuffer[38] - pBuffer[20] + pBuffer[19]);
		break;
	case 176:
		t = pBuffer[282] - pBuffer[266] - pBuffer[206] + pBuffer[190] - (pBuffer[358] - pBuffer[342] - pBuffer[282] + pBuffer[266]);
		break;
	case 177:
		t = (pBuffer[205] - pBuffer[203] - pBuffer[186] + pBuffer[184] + pBuffer[222] - pBuffer[220] - pBuffer[203] + pBuffer[201] - (pBuffer[224] - pBuffer[222] - pBuffer[205] + pBuffer[203]) - (pBuffer[203] - pBuffer[201] - pBuffer[184] + pBuffer[182])) / 2;
		break;
	case 178:
		t = (pBuffer[232] - pBuffer[213] + pBuffer[308] - pBuffer[289] - 2 / 3 * (pBuffer[289] - pBuffer[232]))*1.5;
		break;
	case 179:
		t = pBuffer[204] - pBuffer[203] - pBuffer[185] + pBuffer[184] - (pBuffer[203] - pBuffer[202] - pBuffer[184] + pBuffer[183]);
		break;
	case 180:
		t = (pBuffer[317] - pBuffer[316] - pBuffer[298] + pBuffer[297] + pBuffer[355] - pBuffer[354] - pBuffer[336] + pBuffer[335] - 2 * (pBuffer[336] - pBuffer[335] - pBuffer[317] + pBuffer[316])) / 2;
		break;
	case 181:
		t = (pBuffer[241] - pBuffer[239] - pBuffer[13] + pBuffer[11] + pBuffer[245] - pBuffer[243] - pBuffer[17] + pBuffer[15] - 2 * (pBuffer[243] - pBuffer[241] - pBuffer[15] + pBuffer[13])) / 2;
		break;
	case 182:
		t = (pBuffer[21] - pBuffer[20] - pBuffer[2] + pBuffer[1] + pBuffer[39] - pBuffer[38] - pBuffer[20] + pBuffer[19] - (pBuffer[40] - pBuffer[39] - pBuffer[21] + pBuffer[20]) - (pBuffer[20] - pBuffer[19] - pBuffer[1] + pBuffer[0])) / 2;
		break;
	case 183:
		t = pBuffer[286] - pBuffer[267] - (pBuffer[305] - pBuffer[286]);
		break;
	case 184:
		t = (pBuffer[233] - pBuffer[232] - pBuffer[214] + pBuffer[213] + pBuffer[251] - pBuffer[250] - pBuffer[232] + pBuffer[231] - (pBuffer[252] - pBuffer[251] - pBuffer[233] + pBuffer[232]) - (pBuffer[232] - pBuffer[231] - pBuffer[213] + pBuffer[212])) / 2;
		break;
	case 185:
		t = (pBuffer[350] - pBuffer[348] - pBuffer[274] + pBuffer[272] + pBuffer[353] - pBuffer[351] - pBuffer[277] + pBuffer[275] - 4 * (pBuffer[351] - pBuffer[350] - pBuffer[275] + pBuffer[274])) / 4;
		break;
	case 186:
		t = (pBuffer[232] - pBuffer[231] - pBuffer[213] + pBuffer[212] + pBuffer[270] - pBuffer[269] - pBuffer[251] + pBuffer[250] - 2 * (pBuffer[251] - pBuffer[250] - pBuffer[232] + pBuffer[231])) / 2;
		break;
	case 187:
		t = (pBuffer[127] - pBuffer[125] - pBuffer[70] + pBuffer[68] + pBuffer[131] - pBuffer[129] - pBuffer[74] + pBuffer[72] - 2 * (pBuffer[129] - pBuffer[127] - pBuffer[72] + pBuffer[70])) / 2;
		break;
	case 188:
		t = pBuffer[337] - pBuffer[332] - (pBuffer[332] - pBuffer[327]);
		break;
	case 189:
		t = pBuffer[331] - pBuffer[330] - (pBuffer[330] - pBuffer[329]);
		break;
	case 190:
		t = (pBuffer[125] - pBuffer[87] + pBuffer[277] - pBuffer[239] - 2 / 3 * (pBuffer[239] - pBuffer[125]))*1.5;
		break;
	case 191:
		t = (pBuffer[165] - pBuffer[164] - pBuffer[146] + pBuffer[145] + pBuffer[167] - pBuffer[166] - pBuffer[148] + pBuffer[147] - 2 * (pBuffer[166] - pBuffer[165] - pBuffer[147] + pBuffer[146])) / 2;
		break;
	case 192:
		t = (pBuffer[245] - pBuffer[226] + pBuffer[321] - pBuffer[302] - 2 / 3 * (pBuffer[302] - pBuffer[245]))*1.5;
		break;
	case 193:
		t = pBuffer[292] - pBuffer[286] - pBuffer[273] + pBuffer[267] - (pBuffer[311] - pBuffer[305] - pBuffer[292] + pBuffer[286]);
		break;
	case 194:
		t = (pBuffer[189] - pBuffer[170] + pBuffer[227] - pBuffer[208] - 2 * (pBuffer[208] - pBuffer[189])) / 2;
		break;
	case 195:
		t = pBuffer[49] - pBuffer[30] - (pBuffer[68] - pBuffer[49]);
		break;
	case 196:
		t = pBuffer[97] - pBuffer[96] - (pBuffer[96] - pBuffer[95]);
		break;
	case 197:
		t = (pBuffer[274] - pBuffer[255] + pBuffer[312] - pBuffer[293] - 2 * (pBuffer[293] - pBuffer[274])) / 2;
		break;
	case 198:
		t = (pBuffer[189] - pBuffer[171] - pBuffer[151] + pBuffer[133] + pBuffer[246] - pBuffer[228] - pBuffer[208] + pBuffer[190] - 4 * (pBuffer[208] - pBuffer[190] - pBuffer[189] + pBuffer[171])) / 4;
		break;
	case 199:
		t = pBuffer[159] - pBuffer[156] - pBuffer[140] + pBuffer[137] - (pBuffer[156] - pBuffer[153] - pBuffer[137] + pBuffer[134]);
		break;

	default:
		break;
	}
	return t;
}