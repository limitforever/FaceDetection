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
const int faces = 2901;
const int nonfaces = 28121;
CvHaarClassifierCascade *cascade;

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


float* squared_integral_image(float *img, int width, int height) {  //这是圆形的积分图，可以不用
	float* ii = new float[width*height];
	float* s = new float[width*height];
	int x, y;

	for (y = 0; y < height; y++) {
		for (x = 0; x < width; x++) {
			if (x == 0) s[(y*width) + x] = pow(img[(y*width) + x], 2);
			else s[(y*width) + x] = s[(y*width) + x - 1] + pow(img[(y*width) + x], 2);
			if (y == 0) ii[(y*width) + x] = s[(y*width) + x];
			else ii[(y*width) + x] = ii[((y - 1)*width) + x] + s[(y*width) + x];
		}
	}
	return ii;
}
float evaluate_integral_rectangle(float *ii, int iiwidth, int x, int y, int w, int h) {    //利用积分图，计算特征值
	float value = ii[((y + h - 1)*iiwidth) + (x + w - 1)];                    //  初始化特征值为4位置，即A+B+C+D
	if (x > 0) value -= ii[((y + h - 1)*iiwidth) + (x - 1)];                  //  计算x>0时候的特征值，即4-3，即B+D
	if (y > 0) value -= ii[(y - 1)*iiwidth + (x + w - 1)];                    //  计算y>0时候的特征值，即4-2，即C+D
	if (x > 0 && y > 0) value += ii[(y - 1)*iiwidth + (x - 1)];               //  计算x>0且y>0的特征值，即4+1-（2+3），即D
	return value;
}
float* create_test_sample(IplImage *img)
{
	char test_img_name[10];
	//sprintf(test_img_name, "C:\\Users\\Administrator\\Desktop\\face.test\\test\\test\\%d.jpg", 1);
	//IplImage* img = cvLoadImage(test_img_name, 0);

	int height = img->height, width = img->width;
	float *pGrayBuffer = new float[img->imageSize]; //存取图像像素值
	uchar* ptr = (uchar *)(img->imageData);
	for (int i = 0; i < height; i++)
	{
		uchar* ptr = (uchar *)(img->imageData + i*img->widthStep);
		for (int x = 0; x < img->width; x++)
		{
			//	*ptr = ((uchar *)(img->imageData + i*img->widthStep))[x];
			pGrayBuffer[i*img->width + x] = (float)(ptr[x]);
			//if (pGrayBuffer[i*img->width + x] == 0)
			//cout<<"error  "<<x<<endl;
		}
	}
	return pGrayBuffer;
}

void draw_square(IplImage *img, int x, int y, int size) {
	int thickness = 1 + (size / 100);
	//img = cvCreateImage(cvSize(640, 480), IPL_DEPTH_8U, 1);
	//png::rgb_pixel red;
	//red.red = 255;
	//red.green = 0;
	//red.blue = 0;
	//cvCircle(img, cvPoint(250, 250), 100, cvScalar(255, 0, 0));
	CvScalar s;
	for (size_t i = x; i < x + size; ++i) {
		for (size_t j = y; j < y + thickness; ++j) {
			s = cvGet2D(img, i, j);
			s.val[0] = 0;
			s.val[1] = 0;
			s.val[2] = 255;
			cvSet2D(img, i, j, s);
		}
		for (size_t k = y + size - 1; k >(y + size - 1) - thickness; --k){
			s = cvGet2D(img, i, k);
			s.val[0] = 0;
			s.val[1] = 0;
			s.val[2] = 255;
			cvSet2D(img, i, k, s);
		}
	}
	for (size_t l = y; l < y + size; ++l) {
		for (size_t m = x; m < x + thickness; ++m) {
			s = cvGet2D(img, m, l);
			s.val[0] = 0;
			s.val[1] = 0;
			s.val[2] = 255;
			cvSet2D(img, m, l, s);
		}
		for (size_t n = x + size - 1; n >(x + size - 1) - thickness; --n) {
			s = cvGet2D(img, n, l);
			s.val[0] = 0;
			s.val[1] = 0;
			s.val[2] = 255;
			cvSet2D(img, n, l, s);
		}
	}
}
void merge_detections(vector<int*> detections) {

	int x1, y1, x2, y2, s1, s2;
	int minx, miny, maxx, maxy;
	for (int i = 0; i < detections.size(); i++)
	{
		x1 = detections[i][0]; y1 = detections[i][1]; s1 = detections[i][2];
		for (int j = i + 1; j < detections.size(); j++)
		{
			x2 = detections[j][0]; y2 = detections[j][1]; s2 = detections[j][2];
			if (j != i && ((x1 < x2 + s2) && (x2 < x1 + s1) && (y1 < y2 + s2) && (y2 < y1 + s1)))
			{
				// There's overlapping between detections
				if (x1 > x2)
				{
					minx = x2;
					maxx = x1;
				}
				else
				{
					minx = x1;
					maxx = x2;
				}
				if (y1 > y2)
				{
					miny = y2;
					maxy = y1;
				}
				else
				{
					miny = y1;
					maxy = y2;
				}
				detections[i][0] = minx; detections[i][1] = miny; detections[i][2] = max(maxx - minx, maxy - miny);
				detections.erase(detections.begin() + j);
				j = -1;
			}
		}
	}

}



static int is_equal(const void* _r1, const void* _r2, void*)
{
	const CvRect* r1 = (const CvRect*)_r1;
	const CvRect* r2 = (const CvRect*)_r2;
	int distance = cvRound(r1->width*0.2);

	return r2->x <= r1->x + distance &&
		r2->x >= r1->x - distance &&
		r2->y <= r1->y + distance &&
		r2->y >= r1->y - distance &&
		r2->width <= cvRound(r1->width * 1.2) &&
		cvRound(r2->width * 1.2) >= r1->width;
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
		cvSeqPush(seq, &rs[i]);
	}

	if (min_neighbors != 0)
	{
		// group retrieved rectangles in order to filter out noise 
		int ncomp = cvSeqPartition(seq, 0, &idx_seq, is_equal, 0);
		comps = (CvAvgComp*)cvAlloc((ncomp + 1)*sizeof(comps[0]));
		memset(comps, 0, (ncomp + 1)*sizeof(comps[0]));

		// count number of neighbors
		for (i = 0; i < seq->total; i++)
		{
			CvRect r1 = *(CvRect*)cvGetSeqElem(seq, i);
			int idx = *(int*)cvGetSeqElem(idx_seq, i);
			assert((unsigned)idx < (unsigned)ncomp);

			comps[idx].neighbors++;

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
				int distance = cvRound(r2.rect.width * 0.2);

				if (i != j &&
					r1.rect.x >= r2.rect.x - distance &&
					r1.rect.y >= r2.rect.y - distance &&
					r1.rect.x + r1.rect.width <= r2.rect.x + r2.rect.width + distance &&
					r1.rect.y + r1.rect.height <= r2.rect.y + r2.rect.height + distance
					&& (r2.neighbors > MAX(3, r1.neighbors) || r1.neighbors < 3))
				{
					flag = 0;
					break;
				}
			}

			if (flag)
			{
				cvSeqPush(result_seq, &r1);
				/* cvSeqPush( result_seq, &r1.rect ); */
			}
		}
	}
	return result_seq;

}




void detect_and_draw(IplImage* img, vector<WeakClassifier*> sc_wcs, double sc_weight[201], double Threshold, float fscale, float fincrement)

{
	/*static CvScalar colors[] =
	{
	{ { 0, 0, 255 } },
	{ { 0, 128, 255 } },
	{ { 0, 255, 255 } },
	{ { 0, 255, 0 } },
	{ { 255, 128, 0 } },
	{ { 255, 255, 0 } },
	{ { 255, 0, 0 } },
	{ { 255, 0, 255 } }
	};*/
	vector<WeakClassifier*>::iterator it;
	vector<double>::iterator iit;
	double scale = 1.2;
	IplImage* gray = cvCreateImage(cvSize(img->width, img->height), 8, 1);  //提取图像宽、高，位深度8，通道数1，赋予灰度值
	IplImage* small_img = cvCreateImage(cvSize(cvRound(img->width / scale),   //提取图像宽缩小规模，高缩小规模，位深度8，通道数1，赋予small_img
		cvRound(img->height / scale)),
		8, 1);
	//int i;
	float *gsimg, *iimg, *siimg;
	int i, j, a, b, increment;
	int x, y;
	double s;
	int fnotfound = 0;
	float mean, stdev;
	int* detection;
	const int k = 2;
	int base_resolution = 19;
	vector<int*> detections;
	CvMemStorage *storage = 0;
	//cvCvtColor(img, gray, CV_BGR2GRAY);//把输入的彩色图像转化为灰度图像
	//cvResize(gray, small_img, CV_INTER_LINEAR);//缩小灰色图片
	//cvEqualizeHist(small_img, small_img);//灰度图象直方图均衡化
	//cvClearMemStorage(storage);//释放内存块

	// Calculate integral image and squared integral image
	float* img_test = create_test_sample(img);
	iimg = integral_image(img_test, img->width, img->height);
	siimg = squared_integral_image(img_test, img->width, img->height);
	//delete[] gsimg;
	int width = img->width, height = img->height;
	int smaller = (img->width<img->height) ? img->width : img->height;	//图像的长宽较小者
	CvRect  rs1[1000000];
	CvRect  rs2[1000000];
	int count_window = 0;
	for (s = 1; s * (double)base_resolution <= (double)smaller + 1; s *= 1.2)
	{
		//count = 0;

		for (x = 0; x <= img->width - cvRound(s * (double)base_resolution) + 1; x += cvRound(k * s))//把检测窗口的左上顶点放在待检图像的(x,y)坐标上

		for (y = 0; y <= img->height - cvRound(s * (double)base_resolution) + 1; y += cvRound(k * s))
		{

			rs2[i] = cvRect(x, y, cvRound(base_resolution * s), cvRound(base_resolution * s));

			i++;

		}
	}
	count_window = i - 1;
	int ct;

	int cn;

	for (j = 1; j <= count_window; j++)

		rs1[j] = rs2[j];

	cn = 1;
	ct = 1;
	for (j = 1; j <= count_window; j++)		//对于每个待检窗口
	{
		double s = (double)rs1[j].width / base_resolution;

		double w = 0;

		for (it = sc_wcs.begin(); it != sc_wcs.end() && ct <= 200; ++it)
		{
			ct++;
			//printf("CVRect:%d %d %d %d %d\n",HSC[i].classifier[ct].kind,HSC[i].classifier[ct].r.x,
			//	HSC[i].classifier[ct].r.y,HSC[i].classifier[ct].r.width,HSC[i].classifier[ct].r.height);
			//printf("rs1[j]:%d %d %d %d\n",rs1[j].x,rs1[j].y,rs1[j].width,rs1[j].height);

			CvRect r = cvRect(rs1[j].x + cvRound((*it)->getFeature()->getxc() * s), rs1[j].y + cvRound((*it)->getFeature()->getyc() * s),
				cvRound((*it)->getFeature()->getWidth() * s), cvRound((*it)->getFeature()->getHeight() * s));

			float hvalue = evaluate_integral_rectangle(iimg, width, (*it)->getFeature()->getxc(), (*it)->getFeature()->getyc(),
				(*it)->getFeature()->getWidth(), (*it)->getFeature()->getHeight());

			int ht = 0;

			if ((*it)->getPolarity() * hvalue <(*it)->getPolarity() * (*it)->getthreshold() * s * s)

				ht = 1;

			w += ht * sc_weight[ct];
		}

		if (w >= Threshold)			// rs2 用来暂时保存通过此强分类器的窗口

			rs2[cn++] = rs1[j];
	}

	count_window = cn - 1;


	printf("count_window=%d\n", count_window);
	for (i = 1; i <= count_window; i++)
	{
		cvRectangle(img, cvPoint(rs2[i].x, rs2[i].y), cvPoint(rs2[i].x + rs2[i].width, rs2[i].y + rs2[i].height), CV_RGB(255, 0, 0), 1);
	}
	cvNamedWindow("result", CV_WINDOW_AUTOSIZE);
	cvShowImage("result", img);
	cvWaitKey(0);
	printf("*****************************************\n");
	CvSeq * faces = Merge(rs2, count_window);
	printf("faces=%d\n", faces->total);

	system("pause");
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

	//	double scale = 1.1;
	for (i = 0; i < (faces ? faces->total : 0); i++)
	{
		CvRect* r = (CvRect*)cvGetSeqElem(faces, i);//函数 cvGetSeqElem 查找序列中索引所指定的元素，并返回指向该元素的指针
		CvPoint center;
		int radius;
		center.x = cvRound((r->x + r->width*0.5)*scale);
		center.y = cvRound((r->y + r->height*0.5)*scale);
		if ((radius = cvRound((r->width + r->height)*0.25*scale))>0)
			cvCircle(img, center, radius, colors[i % 8], 3, 8, 0);
	}
	printf("detect end1!\n"); system("pause");
	cvShowImage("result", img);
	cvWaitKey(0);
	printf("detect end!\n"); system("pause");
	//cvReleaseImage( &small_img );

	// Run face detection on multiple scales
	//int base_resolution = cc->getBaseResolutionjkh;

	/*while (width  >= base_resolution && height  >= base_resolution) {
	//increment = base_resolution*fincrement;
	//if (increment < 1) increment = 1;

	// Slide window over image
	for (i = 0; (i + base_resolution) <= width; i = i + 3)
	{
	for (j = 0; (j + base_resolution) <= height; j = j+3)
	{
	// Calculate mean and std. deviation for current window
	//	mean = evaluate_integral_rectangle(iimg, img->width, i, j, base_resolution, base_resolution) / pow(base_resolution, 2);
	//	stdev = sqrt((evaluate_integral_rectangle(siimg, img->width, i, j, base_resolution, base_resolution) / pow(base_resolution, 2)) - pow(mean, 2));

	// Classify window (post-normalization of feature values using mean and stdev)
	if (sc->classify(iimg, width, i, j, 0, 1) == true) {
	//if (true){
	detection = new int[3];
	detection[0] = i; detection[1] = j; detection[2] = base_resolution;
	detections.push_back(detection);
	}
	else fnotfound++;
	}
	}
	height = height / scale;
	width = width / scale;
	//sc->scale(fscale);
	//	base_resolution = cc->getBaseResolution();
	}*/
	/*while (base_resolution <= width && base_resolution <= height) {
	increment = base_resolution*fincrement;
	if (increment < 1) increment = 1;

	// Slide window over image
	for (i = 0; (i + base_resolution) <= width; i += increment) {
	for (j = 0; (j + base_resolution) <= height; j += increment) {
	// Calculate mean and std. deviation for current window
	mean = evaluate_integral_rectangle(iimg, width, i, j, base_resolution, base_resolution) / pow(base_resolution, 2);
	stdev = sqrt((evaluate_integral_rectangle(siimg, width, i, j, base_resolution, base_resolution) / pow(base_resolution, 2)) - pow(mean, 2));

	// Classify window (post-normalization of feature values using mean and stdev)
	if (sc->classify(iimg, width, i, j, mean, stdev) == true) {
	detection = new int[3];
	detection[0] = i; detection[1] = j; detection[2] = base_resolution;
	detections.push_back(detection);
	}
	else fnotfound++;
	}
	}


	base_resolution = base_resolution*1.2;
	}

	// Merge overlapping detections
	merge_detections(detections);

	std::cout << detections.size() << " objects found (" << detections.size() + fnotfound << " total subwindows checked)" << endl;
	for (std::vector<int*>::iterator it = detections.begin(); it != detections.end(); ++it) {
	draw_square(gray, (*it)[0], (*it)[1], (*it)[2]);
	}*/
	//if (cascade)
	//{
	/*double t = (double)cvGetTickCount();//精确测量函数的执行时间
	//从目标图像small_img中检测出人脸
	CvSeq *faces = cvHaarDetectObjects(small_img, cascade, storage, 1.1, 2, 0, cvSize(30, 30));
	t = (double)cvGetTickCount() - t; //计算检测到人脸所需时间
	// printf("检测所用时间 = %gms\n",t/((double)cvGetTickFrequency()*1000.));//打印到屏幕
	//画出检测到的人脸外框(可检测到多个人脸)
	for (i = 0; i < (faces ? faces->total : 0); i++)
	{
	//返回索引所指定的元素指针
	CvRect *r = (CvRect*)cvGetSeqElem(faces, i);
	//用矩形
	//确定两个点来确定人脸位置因为用cvRetangle
	CvPoint pt1, pt2;
	//找到画矩形的两个点

	//用圆形检测
	CvPoint center;
	int radius;
	center.x = cvRound((r->x + r->width*0.5)*scale);
	center.y = cvRound((r->y + r->height*0.5)*scale);
	radius = cvRound((r->width + r->height)*0.25*scale);
	cvCircle(img, center, radius, colors[i % 8], 3, 8, 0);

	//用矩形检测
	//pt1.x = r->x*scale;
	//pt2.x = (r->x+r->width)*scale;
	//pt1.y = (r->y-20)*scale;
	//pt2.y = (r->y+r->height*1.2)*scale;
	//画出矩形
	// cvRectangle( img, pt1, pt2, colors[i%8], 3, 8, 0 );
	//cvRectangle( img, pt1, pt2, CV_RGB(0,0,0), CV_FILLED, 8, 0 );
	//}
	//}
	}*/
	/*cvShowImage("人脸识别", gray);
	cvWaitKey(0);
	cvReleaseImage(&gray);
	cvReleaseImage(&small_img);
	return img;*/
}

int  main()
{

	//char *cascade_name = "D:\\openCV\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_alt2.xml";
	//cascade = (CvHaarClassifierCascade*)cvLoad(cascade_name, 0, 0, 0);

	char test_img_name[100];
	sprintf(test_img_name, "C:\\Users\\Administrator\\Desktop\\face.test\\test\\test\\%d.jpg", 1);
	IplImage* img = cvLoadImage(test_img_name, 0);

	ifstream myfile("e://feature_1.txt");
	float scalefstep = 1.25;
	float slidefstep = 0.1;
	StrongClassifier *sc = new StrongClassifier();
	CvMemStorage *storage = 0;
	CvCapture *capture = 0;//初始化从摄像头中获取视频
	IplImage *frame, *frame_copy = 0;
	const char *input_name;
	storage = cvCreateMemStorage(0);//创建内存块
	//capture = cvCaptureFromCAM(0);//获取摄像头
	//cvNamedWindow("人脸识别", 1);//创建格式化窗口
	char buffer[256];
	int a, b, c, d, e, f, g;
	double h;
	char str[100];
	int i = 0, j = 0;
	WeakClassifier *wc;
	Feature *fea;
	double weight_set[201];
	vector<WeakClassifier*> sc_WeakClassifier_set;
	vector<WeakClassifier*, double> sc_set;
	double Threshold = 0;
	while (!myfile.eof())
	{
		i++;
		myfile.getline(str, 100);
		sscanf(str, "%d,%d,%d,%d,%d,%d,%d,%lf", &a, &b, &c, &d, &e, &f, &g, &h);
		cout << a << " " << b << " " << c << " " << d << " " << e << " " << f << " " << g << " " << h << endl;
		fea = new Feature(a, b, c, d, e);
		wc = new WeakClassifier(fea, f, g);
		sc_WeakClassifier_set.push_back(wc);
		weight_set[i - 1] = h;
		//sc_set.push_back(wc, h);
		Threshold += h;
		if (i != 201)
			sc->add(wc, h);
	}
	myfile.close();
	detect_and_draw(img, sc_WeakClassifier_set, weight_set, Threshold, scalefstep, slidefstep); // 检测并且标识人脸
	/*
	if (capture)
	{
	//循环从摄像头读出图片进行检测
	while (1)
	{
	//从摄像头或者视频文件中抓取帧
	//函数cvQueryFrame从摄像头或者文件中抓取一帧，然后解压并返回这一帧。
	//这个函数仅仅是函数cvGrabFrame和函数cvRetrieveFrame在一起调用的组合。返回的图像不可以被用户释放或者修改。
	if (!cvGrabFrame(capture)){
	break;
	}
	frame = cvRetrieveFrame(capture); //获得由cvGrabFrame函数抓取的图片
	if (!frame){ break; }
	if (!frame_copy){
	frame_copy = cvCreateImage(cvSize(frame->width, frame->height), IPL_DEPTH_8U, frame->nChannels);
	}
	//图像原点或者是左上角 (img->origin=IPL_ORIGIN_TL)或者是左下角(img->origin=IPL_ORIGIN_BL)
	if (frame->origin == IPL_ORIGIN_TL){
	cvCopy(frame, frame_copy, 0);
	}
	else{
	//flip_mode = 0 沿X-轴翻转, flip_mode > 0 (如 1) 沿Y-轴翻转， flip_mode < 0 (如 -1) 沿X-轴和Y-轴翻转.见下面的公式
	//函数cvFlip 以三种方式之一翻转数组 (行和列下标是以0为基点的):
	cvFlip(frame, frame_copy, 0);//反转图像
	}
	detect_and_draw(frame_copy, sc, scalefstep, slidefstep); // 检测并且标识人脸
	if (cvWaitKey(10) >= 0)
	break;
	}

	//释放指针
	cvReleaseImage(&frame_copy);
	cvReleaseCapture(&capture);
	}*/

	cvDestroyWindow("人脸识别");
	return 0;
}



