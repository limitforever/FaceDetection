#ifndef FEATURE_H
#define FEATURE_H

#include <string>
#include <stdlib.h>

class Feature {
public:
	Feature();
	Feature(int type, int xc, int yc, int width, int height);
	float getValue(float *ii, int iiwidth, int x, int y);
	int getType();
	int getWidth();
	int getHeight();
	int Feature::getxc();
	int Feature::getyc();
	void scale(float s);
	std::string toString();
protected:
	int type; // five feature types available (corresponding to A, B, C, C^t and D types on V&J 2001 paper)
	int width, height; // width and height of the feature
	int xc, yc; // x and y coords of top-left feature corner within the detection sub-window  ×óÉÏ½Çx,yµÄÌØÕ÷
	float rectangleValue(float *ii, int iiwidth, int ix, int iy, int rx, int ry, int rw, int rh);
};

#endif
