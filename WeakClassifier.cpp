/*2016.4.13
每个弱分类器相当于一种特性的划分标准。对于每一个弱分类器来说，需要是什么类别，单个样本的在此划分标准下值是多少，以及最终分到哪一类了
对于每一个样本图像，训练的时候我们需要对正负样本进行遍历，用我们已经设定好的特征类型去计算样本其每个位置、大小、形状的特征值，
然后根据正负样本的规模，计算弱分类器（这个指标）下的分布权重、阈值、样本的特征值。这里首先根据正负样本的权值进行排序，然后依据排序后的权重计算
误差阈值，再根据e的计算公式得到一系列的正负样本的权重和分类器阈值。最后，传入样本特征值，依据阈值对样本进行正负分类。*/
#include "WeakClassifier.h"
#include <math.h>
#include <algorithm>
#include <string>
#include <sstream>
#include <vector>
#include <iostream>
/*弱分类器，用权值、特征值、极性组成*/
struct h_f_p_t {
	float value;
	bool polarity;
	float weight;
};
/*定义弱分类器参数*/
struct weakClassifier
{
	int indexF;
	float threshold;
};

/*定义排序方法*/
static bool compare(const h_f_p_t &a, const h_f_p_t &b) {
	return a.value < b.value; //如果a分类器特征值比b的小，则返回true
}


WeakClassifier::WeakClassifier(Feature *f) {
	this->f = f;
}

WeakClassifier::WeakClassifier(Feature *f, float threshold, bool polarity) {
	this->f = f;
	this->threshold = threshold;
	this->polarity = polarity;
}

/*寻找最优阈值*/
float WeakClassifier::find_optimum_threshold(float *fvalues, int fsize, int nfsize, float *weights) {
	h_f_p_t *score = new h_f_p_t[fsize + nfsize];
	for (int i = 0; i<fsize; i++) {      //正样本特征值、标签、权值
		score[i].value = fvalues[i];
		score[i].polarity = true;
		score[i].weight = weights[i];
	}
	for (int j = 0; j<nfsize; j++) {    //负样本特征值、标签、权值
		score[j + fsize].value = fvalues[j + fsize];
		score[j + fsize].polarity = false;
		score[j + fsize].weight = weights[j + fsize];
	}
	std::sort(score, score + fsize + nfsize, compare);     //将特征值排序,第一个参数是排序起始地址，第二个参数是结束地址，第三个参数是排序类型
	/*for (int i = 0; i < fsize + nfsize; i++)
	{
	std::cout << score[i].value << "  ";
	}
	system("pause");*/
	wsum *ws = new wsum[fsize + nfsize];
	float tp = 0;
	float tn = 0;
	if (score[0].polarity == false) {   //划分负样本的权重
		tn = score[0].weight;           //保存当前权重值
		ws[0].sn = score[0].weight;   //赋值负样本权重和S-
		ws[0].sp = 0;
	}
	else {                           //划分正样本的权重
		tp = score[0].weight;
		ws[0].sp = score[0].weight;
		ws[0].sn = 0;
	}
	for (int k = 1; k<(fsize + nfsize); k++) {
		if (score[k].polarity == false) {
			tn += score[k].weight;                   //计算全部负样本权重和T-
			ws[k].sn = ws[k - 1].sn + score[k].weight;   //计算当前负样本权重和S-
			ws[k].sp = ws[k - 1].sp;
		}
		else {
			tp += score[k].weight;                  //计算全部正样本权重和T+
			ws[k].sp = ws[k - 1].sp + score[k].weight;   //计算当前正样本权重和S+
			ws[k].sn = ws[k - 1].sn;
		}
	}
	float minerror = 1;
	float errorp;
	float errorm;
	for (int l = 0; l<(fsize + nfsize); l++) {    // e=min{(S+)+[(T-)-(S-)],(S-)+[(T+)-(S+)])，并且定义一个最小误差
		errorp = ws[l].sp + tn - ws[l].sn;
		errorm = ws[l].sn + tp - ws[l].sp;
		//	if (errorm >= 0.1 && errorp >= 0.1){
		if (errorp < errorm) {
			if (errorp < minerror) {
				minerror = errorp;               //找到最小的分类误差阈值，并判断极性
				this->threshold = score[l].value;
				this->polarity = false;
			}
		}
		else {
			if (errorm < minerror) {
				minerror = errorm;
				this->threshold = score[l].value;
				this->polarity = true;
			}
		}
		//	}
	}
	delete[] score;
	delete[] ws;
	return minerror;
}

Feature* WeakClassifier::getFeature() {
	return this->f;
}

float WeakClassifier::getthreshold() {
	return this->threshold;
}
int WeakClassifier::getPolarity() {
	return this->polarity;
}
/*利用弱分类器对正负样本分类，返回值为1/-1*/
int WeakClassifier::classify(float *img, int imwidth, int x, int y, float mean, float stdev) {
	float fval = this->f->getValue(img, imwidth, x, y);
	// 若是第二种或者第三种特征
	if (this->f->getType() == 2 || this->f->getType() == 3)
		fval = (fval + (this->f->getWidth()*this->f->getHeight()*mean / 3)); //样本特征值=  累加求和(宽*高*平均值/3)
	if (stdev != 0) fval = fval / stdev; //调整特征值，除以一个系数
	//std::cout << "fval:" << fval << std::endl;
	if (fval < this->threshold) {        //如果特征值小于阈值，且是正样本返回1
		if (this->polarity) return 1;
		else return -1;
	}
	else {                               //如果特征值大于阈值，且是负样本返回1
		if (this->polarity) return -1;
		else return 1;
	}
}

void WeakClassifier::scale(float s) {
	f->scale(s);
	this->threshold = (this->threshold)*pow(s, 2);
}

