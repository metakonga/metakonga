#ifndef CONTACTPAIR_H
#define CONTACTPAIR_H

#include <QString>

class pointMass;

class contactPair
{
public:
	contactPair();
	contactPair(QString name);
	~contactPair();

	void setFirstBody(pointMass* _ib) { ib = _ib; }
	void setSecondBody(pointMass* _jb) { jb = _jb; }

private:
	pointMass* ib;
	pointMass* jb;
};

#endif