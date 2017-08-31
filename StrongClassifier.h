
#ifndef STRONGCLASSIFIER_H
#define STRONGCLASSIFIER_H

#include "WeakClassifier.h"
#include <vector>
#include <string>

class StrongClassifier {
public:
	StrongClassifier();
	StrongClassifier(std::vector<WeakClassifier*> wc, float *weight);  //参数为弱分类器集和权值的构造函数
	StrongClassifier(std::vector<WeakClassifier*> wc, float *weight, float threshold);    //参数为弱分类器集和权值和阈值的构造函数
	bool classify(float *img, int imwidth, int x, int y, float mean, float stdev);    //分类器（图像，宽度，坐标xy，平均值和stdev）
	std::string toString();
	void add(WeakClassifier* wc, float weight);         //添加弱分类器的函数
	void scale(float s);                            //比例尺函数（放大比例）
	void optimise_threshold(std::vector<float*> &positive_set, int base_resolution, float maxfnr); //最优阈值函数（正样本集，基分辨率，搜索阈值上限）
	float fnr(std::vector<float*> &positive_set, int base_resolution);   //负样本特征阈值(正样本集，基分辨率)
	float fpr(std::vector<float*> &negative_set, int base_resolution);   //正样本特征阈值(负样本集，基分辨率)
	void strictness(float p);
	std::vector<WeakClassifier*> getWeakClassifiers();
	std::vector<float*> getWeight(int i);
protected:
	std::vector<WeakClassifier*> wc; //弱分类器
	std::vector<float> weight;     //权值
	float threshold;     //阈值
};

#endif
