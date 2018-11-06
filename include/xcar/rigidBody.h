#ifndef RIGIDBODY_H
#define RIGIDBODY_H

#include "pointMass.h"

class rigidBody : public pointMass
{
public:
	rigidBody();
	rigidBody(QString _name);
	~rigidBody();
};

#endif