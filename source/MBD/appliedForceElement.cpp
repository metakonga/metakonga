#include "appliedForceElement.h"
using namespace parSIM;

appliedForceElement::appliedForceElement()
	: targetBody(0)
{

}

appliedForceElement::appliedForceElement(int tBody, char dir, double (*_aForce)(double))
	: targetBody(tBody)
{
	for(int i(0); i < 3; i++) direction[i]=0.0;
	switch(dir){
	case 'x': direction[0] = 1.0; break;
	case 'y': direction[1] = 1.0; break;
	case 'z': direction[2] = 1.0; break;
	}
	aForce = _aForce;
}

appliedForceElement::~appliedForceElement()
{

}