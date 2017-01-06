#include "CementElement.h"

double CementElement::brmul = 0; // bond radius multiplier
double CementElement::cyoungsModulus = 0;
double CementElement::cstiffnessRatio = 0;
double CementElement::maxTensileStress = 0;
double CementElement::maxShearStress = 0;
double CementElement::tensileStdDeviation = 0;
double CementElement::shearStdDeviation = 0;

CementElement::CementElement()
{

}

CementElement::~CementElement()
{

}