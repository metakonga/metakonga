#include "forceElement.h"

forceElement::forceElement()
{

}

forceElement::forceElement(QString _name, mbd_model* _md, Type tp, pointMass* _b, pointMass* _a)
	: name(_name)
	, md(_md)
	, type(tp)
	, base(_b)
	, action(_a)
{

}

forceElement::~forceElement()
{

}

QString forceElement::Name()
{
	return name;
}

pointMass* forceElement::BaseBody()
{
	return base;
}

pointMass* forceElement::ActionBody()
{
	return action;
}

forceElement::Type forceElement::ForceType()
{
	return type;
}
