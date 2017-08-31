/*对于每一个特征，样本会根据特征值排序，用Adaboost得到的特征最优阈值可以用列表排序的结果得到，对于其中的每一个元素，:四个和T和S维持和评估*/
#ifndef WEAKCLASSIFIER_H
#define WEAKCLASSIFIER_H

#include "Feature.h"
#include <string>
#include <vector>

struct wsum {
	// 当前样本正权重和 S+
	float sp;
	// 当前样本负权重和 S−
	float sn;
};

class WeakClassifier {
public:
	WeakClassifier(Feature *f);
	WeakClassifier(Feature *f, float threshold, bool polarity);
	float find_optimum_threshold(float *fvalues, int fsize, int nfsize, float *weights);
	Feature* getFeature();
	float getthreshold();
	int getPolarity();
	int classify(float *img, int imwidth, int x, int y, float mean, float stdev);
	void scale(float s);
	std::string toString();
protected:
	Feature *f;
	float threshold;
	bool polarity;
};

#endif
