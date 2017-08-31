
#ifndef STRONGCLASSIFIER_H
#define STRONGCLASSIFIER_H

#include "WeakClassifier.h"
#include <vector>
#include <string>

class StrongClassifier {
public:
	StrongClassifier();
	StrongClassifier(std::vector<WeakClassifier*> wc, float *weight);  //����Ϊ������������Ȩֵ�Ĺ��캯��
	StrongClassifier(std::vector<WeakClassifier*> wc, float *weight, float threshold);    //����Ϊ������������Ȩֵ����ֵ�Ĺ��캯��
	bool classify(float *img, int imwidth, int x, int y, float mean, float stdev);    //��������ͼ�񣬿�ȣ�����xy��ƽ��ֵ��stdev��
	std::string toString();
	void add(WeakClassifier* wc, float weight);         //������������ĺ���
	void scale(float s);                            //�����ߺ������Ŵ������
	void optimise_threshold(std::vector<float*> &positive_set, int base_resolution, float maxfnr); //������ֵ�������������������ֱ��ʣ�������ֵ���ޣ�
	float fnr(std::vector<float*> &positive_set, int base_resolution);   //������������ֵ(�������������ֱ���)
	float fpr(std::vector<float*> &negative_set, int base_resolution);   //������������ֵ(�������������ֱ���)
	void strictness(float p);
	std::vector<WeakClassifier*> getWeakClassifiers();
	std::vector<float*> getWeight(int i);
protected:
	std::vector<WeakClassifier*> wc; //��������
	std::vector<float> weight;     //Ȩֵ
	float threshold;     //��ֵ
};

#endif
