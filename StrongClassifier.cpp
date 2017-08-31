/*2016.4.13
强分类器分类
对强分类器分类时需要用弱分类器进行分类，包括强分类器的阈值和权重，当然，每个强分类器自己也有一个阈值
强分类器分类时其分值计算利用到弱分类器的权值，以及弱分类器的结果进行投票，最后与强分类器阈值进行比较，如果比阈值大，就是强分类器，反之不是
强分类器阈值计算是个难点，需要利用到fnr和fpr两个指标，fnr是正样本误报率即正样本分到负样本的比率，fpr是负样本误报率即负样本分到正样本的比率
maxfnr我理解为正样本因为少其分错不能太大，要有个上限，minfpr由于负样本多其分错需要有个最小值
在计算强分类器阈值时，需要利用正样本和正分类结果计算强分类器分值（这里用到了基础表示，但是为什么？），然后对分值进行排序，再计算正样本误报最大个数
那么阈值最终为正样本里所有误报个数中稍微小一点的值。
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
/*强分类器属性=若干弱分类器+若干权重+强分类器阈值*/
StrongClassifier::StrongClassifier(std::vector<WeakClassifier*> wc, float *weight, float threshold) {
	this->wc = wc;
	for (int i = 0; i<wc.size(); i++) this->weight.push_back(weight[i]);
	this->threshold = threshold;
}


/*强分类器的分类：用一个强分类器阈值，根据计算出来的分值来判断是否是强分类器*/
bool StrongClassifier::classify(float *img, int imwidth, int x, int y, float mean, float stdev) {
	float strongscore = 0, Threshold = 0;
	int i = 0;
	for (std::vector<WeakClassifier*>::iterator it = wc.begin(); it != wc.end(); ++it) {
		strongscore += weight[i] * ((*it)->classify(img, imwidth, x, y, mean, stdev));    // 强分类器分值=求和(权重*弱分类器分类结果)
		Threshold = weight[i] + Threshold;
		i++;
		//strongscore += alphat * ((*it)->classify(img, imwidth, x, y, mean, stdev));    // 强分类器分值=求和(权重*弱分类器分类结果)
		//Threshold += alphat;
	}
	//if (strongscore >= this->threshold) return true;  //如果大于阈值是强分类器，小于阈值不是的
	if (strongscore >= Threshold / 2) return true;  //如果大于阈值是强分类器，小于阈值不是的
	else return false;
}

void StrongClassifier::add(WeakClassifier* wc, float weight) {
	this->wc.push_back(wc);
	this->weight.push_back(weight);
}
/*正样本的误报率*/
float StrongClassifier::fnr(std::vector<float*> &positive_set, int base_resolution) {
	int fn = 0;
	for (int i = 0; i<positive_set.size(); i++)
	if (this->classify(positive_set[i], base_resolution, 0, 0, 0, 1) == false)
		fn++;
	return float(fn) / float(positive_set.size());
}
/*负样本的误报率*/
float StrongClassifier::fpr(std::vector<float*> &negative_set, int base_resolution) {
	int fp = 0;
	for (int i = 0; i<negative_set.size(); i++) //对于每个负样本，参数（负样本集，最小宽度也是2，从原点开始扫，不更新权值），那么fp++
	if (this->classify(negative_set[i], base_resolution, 0, 0, 0, 1) == true)
		fp++;
	return float(fp) / float(negative_set.size());  //负样本的误报率（负样本中被判断为正样本的概率=负样本被判断错的/负样本个数）
}

void StrongClassifier::scale(float s) {
	for (std::vector<WeakClassifier*>::iterator it = this->wc.begin(); it != this->wc.end(); ++it) {
		(*it)->scale(s);
	}
}
/*强分类器的最优阈值*/
void StrongClassifier::optimise_threshold(std::vector<float*> &positive_set, int base_resolution, float maxfnr) {//maxfnr是正样本的误报率最大值，阈值根据正样本训练得到
	int wf;
	float thr;
	float *strongscores = new float[positive_set.size()];
	for (int i = 0; i<positive_set.size(); i++) {   //根据正样本大小
		strongscores[i] = 0;
		wf = 0;
		for (std::vector<WeakClassifier*>::iterator it = wc.begin(); it != wc.end(); ++it, wf++)
			strongscores[i] += (this->weight[wf])*((*it)->classify(positive_set[i], base_resolution, 0, 0, 0, 1));//强分类器分值=求和(权重*正样本分类结果)
	}
	std::sort(strongscores, strongscores + positive_set.size());   //对正样本集排序
	int maxfnrind = maxfnr*positive_set.size();                 //计算正样本误报的最大个数：最大正样本误报率*正样本个数
	if (maxfnrind >= 0 && maxfnrind < positive_set.size()) {
		thr = strongscores[maxfnrind];
		while (maxfnrind > 0 && strongscores[maxfnrind] == thr)
			maxfnrind--;
		this->threshold = strongscores[maxfnrind];    //强分类器阈值设定为比强分类器所有误报个数中稍微小一点的值
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
