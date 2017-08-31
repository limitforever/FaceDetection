/*2016.4.13
ÿ�����������൱��һ�����ԵĻ��ֱ�׼������ÿһ������������˵����Ҫ��ʲô��𣬵����������ڴ˻��ֱ�׼��ֵ�Ƕ��٣��Լ����շֵ���һ����
����ÿһ������ͼ��ѵ����ʱ��������Ҫ�������������б������������Ѿ��趨�õ���������ȥ����������ÿ��λ�á���С����״������ֵ��
Ȼ��������������Ĺ�ģ�������������������ָ�꣩�µķֲ�Ȩ�ء���ֵ������������ֵ���������ȸ�������������Ȩֵ��������Ȼ������������Ȩ�ؼ���
�����ֵ���ٸ���e�ļ��㹫ʽ�õ�һϵ�е�����������Ȩ�غͷ�������ֵ����󣬴�����������ֵ��������ֵ�����������������ࡣ*/
#include "WeakClassifier.h"
#include <math.h>
#include <algorithm>
#include <string>
#include <sstream>
#include <vector>
#include <iostream>
/*������������Ȩֵ������ֵ���������*/
struct h_f_p_t {
	float value;
	bool polarity;
	float weight;
};
/*����������������*/
struct weakClassifier
{
	int indexF;
	float threshold;
};

/*�������򷽷�*/
static bool compare(const h_f_p_t &a, const h_f_p_t &b) {
	return a.value < b.value; //���a����������ֵ��b��С���򷵻�true
}


WeakClassifier::WeakClassifier(Feature *f) {
	this->f = f;
}

WeakClassifier::WeakClassifier(Feature *f, float threshold, bool polarity) {
	this->f = f;
	this->threshold = threshold;
	this->polarity = polarity;
}

/*Ѱ��������ֵ*/
float WeakClassifier::find_optimum_threshold(float *fvalues, int fsize, int nfsize, float *weights) {
	h_f_p_t *score = new h_f_p_t[fsize + nfsize];
	for (int i = 0; i<fsize; i++) {      //����������ֵ����ǩ��Ȩֵ
		score[i].value = fvalues[i];
		score[i].polarity = true;
		score[i].weight = weights[i];
	}
	for (int j = 0; j<nfsize; j++) {    //����������ֵ����ǩ��Ȩֵ
		score[j + fsize].value = fvalues[j + fsize];
		score[j + fsize].polarity = false;
		score[j + fsize].weight = weights[j + fsize];
	}
	std::sort(score, score + fsize + nfsize, compare);     //������ֵ����,��һ��������������ʼ��ַ���ڶ��������ǽ�����ַ����������������������
	/*for (int i = 0; i < fsize + nfsize; i++)
	{
	std::cout << score[i].value << "  ";
	}
	system("pause");*/
	wsum *ws = new wsum[fsize + nfsize];
	float tp = 0;
	float tn = 0;
	if (score[0].polarity == false) {   //���ָ�������Ȩ��
		tn = score[0].weight;           //���浱ǰȨ��ֵ
		ws[0].sn = score[0].weight;   //��ֵ������Ȩ�غ�S-
		ws[0].sp = 0;
	}
	else {                           //������������Ȩ��
		tp = score[0].weight;
		ws[0].sp = score[0].weight;
		ws[0].sn = 0;
	}
	for (int k = 1; k<(fsize + nfsize); k++) {
		if (score[k].polarity == false) {
			tn += score[k].weight;                   //����ȫ��������Ȩ�غ�T-
			ws[k].sn = ws[k - 1].sn + score[k].weight;   //���㵱ǰ������Ȩ�غ�S-
			ws[k].sp = ws[k - 1].sp;
		}
		else {
			tp += score[k].weight;                  //����ȫ��������Ȩ�غ�T+
			ws[k].sp = ws[k - 1].sp + score[k].weight;   //���㵱ǰ������Ȩ�غ�S+
			ws[k].sn = ws[k - 1].sn;
		}
	}
	float minerror = 1;
	float errorp;
	float errorm;
	for (int l = 0; l<(fsize + nfsize); l++) {    // e=min{(S+)+[(T-)-(S-)],(S-)+[(T+)-(S+)])�����Ҷ���һ����С���
		errorp = ws[l].sp + tn - ws[l].sn;
		errorm = ws[l].sn + tp - ws[l].sp;
		//	if (errorm >= 0.1 && errorp >= 0.1){
		if (errorp < errorm) {
			if (errorp < minerror) {
				minerror = errorp;               //�ҵ���С�ķ��������ֵ�����жϼ���
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
/*�������������������������࣬����ֵΪ1/-1*/
int WeakClassifier::classify(float *img, int imwidth, int x, int y, float mean, float stdev) {
	float fval = this->f->getValue(img, imwidth, x, y);
	// ���ǵڶ��ֻ��ߵ���������
	if (this->f->getType() == 2 || this->f->getType() == 3)
		fval = (fval + (this->f->getWidth()*this->f->getHeight()*mean / 3)); //��������ֵ=  �ۼ����(��*��*ƽ��ֵ/3)
	if (stdev != 0) fval = fval / stdev; //��������ֵ������һ��ϵ��
	//std::cout << "fval:" << fval << std::endl;
	if (fval < this->threshold) {        //�������ֵС����ֵ����������������1
		if (this->polarity) return 1;
		else return -1;
	}
	else {                               //�������ֵ������ֵ�����Ǹ���������1
		if (this->polarity) return -1;
		else return 1;
	}
}

void WeakClassifier::scale(float s) {
	f->scale(s);
	this->threshold = (this->threshold)*pow(s, 2);
}

