#include "utils.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>

using namespace parsph;

#define max(a,b)    (((a) > (b)) ? (a) : (b))
#define min(a,b)    (((a) < (b)) ? (a) : (b))

std::string utils::integerString(int number)
{
	std::stringstream str;
	str << number;
	return str.str();
}

bool utils::circleLineIntersect(const vector3<double>& startLine, const vector3<double>& endLine, const vector3<double>& sphereCenter, double radius)
{
	vector2<double> v = vector2<double>(endLine.x - startLine.x, endLine.y - startLine.y);
	vector2<double> w = vector2<double>(sphereCenter.x - startLine.x, sphereCenter.y - startLine.y);
	double t = w.dot(v) / v.dot(v);
	t = max(min(t, 1.0), 0.0);
	vector2<double> closestLine = startLine.toVector2() + (v * t) - sphereCenter.toVector2();
	return closestLine.lengthSq() < radius * radius;
}

int utils::packIntegerPair( int z1, int z2 )
{
	z1 = (z1 >= 0) ? z1 * 2 : -z1 * 2 - 1;
	z2 = (z2 >= 0) ? z2 * 2 : -z2 * 2 - 1;

	return ((z1 + z2) * (z1 + z2 + 1)) / 2 + z2;
}