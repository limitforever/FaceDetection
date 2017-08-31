
/*2016.4.13
����ѡȡ�ͼ��㺯��
����ͼ����󣬳����Լ����꣬���û���ͼ�����ټ������5������������ֵ*/

#include <string>
#include <iostream>
#include "Feature.h"


Feature::Feature(int t, int x, int y, int w, int h) {    //x,y��ʾλ�ã�w,h��ʾ��С��t��ʾ����
	type = t;
	xc = x;
	yc = y;
	width = w;
	height = h;
}

int Feature::getType() {
	return this->type;
}

int Feature::getWidth() {
	return this->width;
}

int Feature::getHeight() {
	return this->height;
}

int Feature::getxc() {
	return this->xc;
}

int Feature::getyc() {
	return this->yc;
}

/*���û���ͼ�����������ֵ*/
float Feature::rectangleValue(float *ii, int iiwidth, int ix, int iy, int rx, int ry, int rw, int rh) {
	float value = ii[((iy + ry + rh - 1)*iiwidth) + (ix + rx + rw - 1)];         //A+B+C+D
	if ((ix + rx) > 0) value -= ii[((iy + ry + rh - 1)*iiwidth) + (ix + rx - 1)];  //B+D(x>0)
	if ((iy + ry) > 0) value -= ii[((iy + ry - 1)*iiwidth) + (ix + rx + rw - 1)];  //C+D(y>0)
	if ((ix + rx) > 0 && (iy + ry) > 0) value += ii[((iy + ry - 1)*iiwidth) + (ix + rx - 1)];   //D(x>0��y>0)
	return value;
}
/*�趨��������*/
float Feature::getValue(float *ii, int iiwidth, int x, int y) {
	// 5 types of feature (A, B, C, C^t, D)
	float t = 0;
	switch (this->type) {
		//switch (type) {  //xc,ycΪ����ͼ��ʼλ�õ����꣨���Ͻǣ�
	case 0:
		t = (rectangleValue(ii, iiwidth, x, y, this->xc + (this->width / 2), this->yc, this->width / 2, this->height) - rectangleValue(ii, iiwidth, x, y, this->xc, this->yc, this->width / 2, this->height));
		//std::cout << "����1������ֵΪ��" << t << std::endl;
		return t;
		break;
	case 1:
		t = (rectangleValue(ii, iiwidth, x, y, this->xc, this->yc, this->width, this->height / 2) - rectangleValue(ii, iiwidth, x, y, this->xc, this->yc + (this->height / 2), this->width, this->height / 2));
		//std::cout << "����2������ֵΪ��" << t << std::endl;
		return t;
		break;
	case 2:
		t = (rectangleValue(ii, iiwidth, x, y, this->xc, this->yc, this->width / 3, this->height) + rectangleValue(ii, iiwidth, x, y, this->xc + (width * 2 / 3), this->yc, this->width / 3, this->height) - 2 * rectangleValue(ii, iiwidth, x, y, this->xc + (width / 3), this->yc, this->width / 3, this->height)) / 2;
		//std::cout << "����3������ֵΪ��" << t << std::endl;
		return t;
		break;
	case 3:
		t = (rectangleValue(ii, iiwidth, x, y, this->xc, this->yc, this->width, this->height / 3) + rectangleValue(ii, iiwidth, x, y, this->xc, this->yc + (this->height * 2 / 3), this->width, this->height / 3) - 2 * rectangleValue(ii, iiwidth, x, y, this->xc, this->yc + (this->height / 3), this->width, this->height / 3)) / 2;
		//std::cout << "����4������ֵΪ��" << t << std::endl;
		return t;
		break;
	case 4:
		t = (rectangleValue(ii, iiwidth, x, y, this->xc, this->yc, this->width * 2 / 5, this->height) + rectangleValue(ii, iiwidth, x, y, this->xc + (width * 3 / 5), this->yc, this->width * 2 / 5, this->height) - 4 * rectangleValue(ii, iiwidth, x, y, this->xc + (width * 2 / 5), this->yc, this->width / 5, this->height)) / 4;
		//std::cout << "����3������ֵΪ��" << t << std::endl;
		return t;
		break;
	case 5:
		t = (rectangleValue(ii, iiwidth, x, y, this->xc, this->yc, this->width, this->height * 2 / 5) - rectangleValue(ii, iiwidth, x, y, this->xc, this->yc + (this->height * 3 / 5), this->width, this->height * 2 / 5) - 4 * rectangleValue(ii, iiwidth, x, y, this->xc, this->yc + (this->height * 2 / 5), this->width, this->height / 5)) / 4;
		//std::cout << "����4������ֵΪ��" << t << std::endl;
		return t;
		break;

	case 6:
		t = (rectangleValue(ii, iiwidth, x, y, this->xc, this->yc, this->width / 5, this->height) + rectangleValue(ii, iiwidth, x, y, this->xc + (width * 4 / 5), this->yc, this->width / 5, this->height) - 2 / 3 * rectangleValue(ii, iiwidth, x, y, this->xc + (width / 5), this->yc, this->width * 3 / 5, this->height))*1.5;
		//std::cout << "����3������ֵΪ��" << t << std::endl;
		return t;
		break;
	case 7:
		t = (rectangleValue(ii, iiwidth, x, y, this->xc, this->yc, this->width, this->height / 5) + rectangleValue(ii, iiwidth, x, y, this->xc, this->yc + (this->height * 4 / 5), this->width, this->height / 5) - 2 / 3 * rectangleValue(ii, iiwidth, x, y, this->xc, this->yc + (this->height / 5), this->width, this->height * 3 / 5))*1.5;
		//std::cout << "����4������ֵΪ��" << t << std::endl;
		return t;
		break;

	case 8:
		t = (rectangleValue(ii, iiwidth, x, y, this->xc + (width / 2), this->yc, this->width / 2, this->height / 2) + rectangleValue(ii, iiwidth, x, y, this->xc, this->yc + (height / 2), this->width / 2, this->height / 2) - rectangleValue(ii, iiwidth, x, y, this->xc + (width / 2), this->yc + (height / 2), this->width / 2, this->height / 2) - rectangleValue(ii, iiwidth, x, y, this->xc, this->yc, this->width / 2, this->height / 2)) / 2;
		//std::cout << "����5������ֵΪ��" << t << std::endl;
		return t;
		break;
	case 9:
		t = (rectangleValue(ii, iiwidth, x, y, this->xc, this->yc, this->width, this->height) - 9 * rectangleValue(ii, iiwidth, x, y, this->xc + width / 3, this->yc + (height / 3), this->width / 3, this->height / 3)) / 8;
		//std::cout << "����5������ֵΪ��" << t << std::endl;
		return t;
		break;
	default:
		std::cout << "�����������" << this->type << " �������ڣ�" << std::endl;
		exit(-1);
		break;
	}

}
/*��������*/
void Feature::scale(float s) {
	this->width = (this->width)*s;
	this->height = (this->height)*s;
	this->xc = (this->xc)*s;
	this->yc = (this->yc)*s;
}

/*���ÿ��ٻ��ַ�*/
void fastIntegral(unsigned char* inputMatrix, unsigned long* outputMatrix, int width, int height){
	unsigned long *columnSum = new unsigned long[width]; // sum of each column  
	// calculate integral of the first line  
	for (int i = 0; i<width; i++){
		columnSum[i] = inputMatrix[i];
		outputMatrix[i] = inputMatrix[i];
		if (i>0){
			outputMatrix[i] += outputMatrix[i - 1];
		}
	}
	for (int i = 1; i<height; i++){
		int offset = i*width;
		// first column of each line  
		columnSum[0] += inputMatrix[offset];
		outputMatrix[offset] = columnSum[0];
		// other columns   
		for (int j = 1; j<width; j++){
			columnSum[j] += inputMatrix[offset + j];
			outputMatrix[offset + j] = outputMatrix[offset + j - 1] + columnSum[j];
		}
	}
	return;
}
