#include "artificialCoordinate.h"

int artificialCoordinate::count = 0;

artificialCoordinate::artificialCoordinate(QString _name)
	: name(_name)
	, id(count)
	, matrix_location(0)
{
	count++;
}

artificialCoordinate::~artificialCoordinate()
{
	count--;
}