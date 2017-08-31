/*2016.4.13
ǿ����������
��ǿ����������ʱ��Ҫ�������������з��࣬����ǿ����������ֵ��Ȩ�أ���Ȼ��ÿ��ǿ�������Լ�Ҳ��һ����ֵ
ǿ����������ʱ���ֵ�������õ�����������Ȩֵ���Լ����������Ľ������ͶƱ�������ǿ��������ֵ���бȽϣ��������ֵ�󣬾���ǿ����������֮����
ǿ��������ֵ�����Ǹ��ѵ㣬��Ҫ���õ�fnr��fpr����ָ�꣬fnr�����������ʼ��������ֵ��������ı��ʣ�fpr�Ǹ��������ʼ��������ֵ��������ı���
maxfnr�����Ϊ��������Ϊ����ִ���̫��Ҫ�и����ޣ�minfpr���ڸ���������ִ���Ҫ�и���Сֵ
�ڼ���ǿ��������ֵʱ����Ҫ������������������������ǿ��������ֵ�������õ��˻�����ʾ������Ϊʲô������Ȼ��Է�ֵ���������ټ�����������������
��ô��ֵ����Ϊ�������������󱨸�������΢Сһ���ֵ��
*/
#include "StrongClassifier.h"
#include <algorithm>
#include <string>
#include <sstream>
#include <vector>
#include <iostream>

struct score {
	float value;
	bool label;
	float weight;
};

static bool compare(const score &a, const score &b) {
	return a.value < b.value;
}

StrongClassifier::StrongClassifier() {
	this->threshold = 0;
}

StrongClassifier::StrongClassifier(std::vector<WeakClassifier*> wc, float *weight) {
	this->wc = wc;
	this->threshold = 0;
	for (int i = 0; i<wc.size(); i++) this->weight.push_back(weight[i]);
}
/*ǿ����������=������������+����Ȩ��+ǿ��������ֵ*/
StrongClassifier::StrongClassifier(std::vector<WeakClassifier*> wc, float *weight, float threshold) {
	this->wc = wc;
	for (int i = 0; i<wc.size(); i++) this->weight.push_back(weight[i]);
	this->threshold = threshold;
}


/*ǿ�������ķ��ࣺ��һ��ǿ��������ֵ�����ݼ�������ķ�ֵ���ж��Ƿ���ǿ������*/
bool StrongClassifier::classify(float *img, int imwidth, int x, int y, float mean, float stdev) {
	float strongscore = 0, Threshold = 0;
	int i = 0;
	for (std::vector<WeakClassifier*>::iterator it = wc.begin(); it != wc.end(); ++it) {
		strongscore += weight[i] * ((*it)->classify(img, imwidth, x, y, mean, stdev));    // ǿ��������ֵ=���(Ȩ��*��������������)
		Threshold = weight[i] + Threshold;
		i++;
		//strongscore += alphat * ((*it)->classify(img, imwidth, x, y, mean, stdev));    // ǿ��������ֵ=���(Ȩ��*��������������)
		//Threshold += alphat;
	}
	//if (strongscore >= this->threshold) return true;  //���������ֵ��ǿ��������С����ֵ���ǵ�
	if (strongscore >= Threshold / 2) return true;  //���������ֵ��ǿ��������С����ֵ���ǵ�
	else return false;
}

void StrongClassifier::add(WeakClassifier* wc, float weight) {
	this->wc.push_back(wc);
	this->weight.push_back(weight);
}
/*������������*/
float StrongClassifier::fnr(std::vector<float*> &positive_set, int base_resolution) {
	int fn = 0;
	for (int i = 0; i<positive_set.size(); i++)
	if (this->classify(positive_set[i], base_resolution, 0, 0, 0, 1) == false)
		fn++;
	return float(fn) / float(positive_set.size());
}
/*������������*/
float StrongClassifier::fpr(std::vector<float*> &negative_set, int base_resolution) {
	int fp = 0;
	for (int i = 0; i<negative_set.size(); i++) //����ÿ����������������������������С���Ҳ��2����ԭ�㿪ʼɨ��������Ȩֵ������ôfp++
	if (this->classify(negative_set[i], base_resolution, 0, 0, 0, 1) == true)
		fp++;
	return float(fp) / float(negative_set.size());  //�����������ʣ��������б��ж�Ϊ�������ĸ���=���������жϴ��/������������
}

void StrongClassifier::scale(float s) {
	for (std::vector<WeakClassifier*>::iterator it = this->wc.begin(); it != this->wc.end(); ++it) {
		(*it)->scale(s);
	}
}
/*ǿ��������������ֵ*/
void StrongClassifier::optimise_threshold(std::vector<float*> &positive_set, int base_resolution, float maxfnr) {//maxfnr�����������������ֵ����ֵ����������ѵ���õ�
	int wf;
	float thr;
	float *strongscores = new float[positive_set.size()];
	for (int i = 0; i<positive_set.size(); i++) {   //������������С
		strongscores[i] = 0;
		wf = 0;
		for (std::vector<WeakClassifier*>::iterator it = wc.begin(); it != wc.end(); ++it, wf++)
			strongscores[i] += (this->weight[wf])*((*it)->classify(positive_set[i], base_resolution, 0, 0, 0, 1));//ǿ��������ֵ=���(Ȩ��*������������)
	}
	std::sort(strongscores, strongscores + positive_set.size());   //��������������
	int maxfnrind = maxfnr*positive_set.size();                 //�����������󱨵����������������������*����������
	if (maxfnrind >= 0 && maxfnrind < positive_set.size()) {
		thr = strongscores[maxfnrind];
		while (maxfnrind > 0 && strongscores[maxfnrind] == thr)
			maxfnrind--;
		this->threshold = strongscores[maxfnrind];    //ǿ��������ֵ�趨Ϊ��ǿ�����������󱨸�������΢Сһ���ֵ
	}
	delete[] strongscores;
}

std::vector<WeakClassifier*> StrongClassifier::getWeakClassifiers() {
	return this->wc;
}
/*std::vector<float*> StrongClassifier::getWeight(int i) {
return this->weight.at[i];
}*/


void StrongClassifier::strictness(float p) {
	this->threshold = (this->threshold)*p;
}
