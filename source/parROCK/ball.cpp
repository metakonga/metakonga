#include "ball.h"
#include "contact.h"
#include "Geometry.h"

ball* ball::beginBallPointer = NULL;
unsigned int ball::count = 0;
unsigned int ball::nballs = 0;

ball::ball()
	: id(nballs)
{
	if (!nballs)
		beginBallPointer = this;
	nballs++;
}

ball::~ball()
{
	//for (std::map<ball*, ccontact*>::iterator it = pcmap.begin(); it != pcmap.end(); it++){
	//	delete it->second;
	//}
	//for (std::map<Geometry*, ccontact*>::iterator it = wcmap.begin(); it != wcmap.end(); it++){
	//	delete it->second;
	//}
}

ball* ball::BeginBall()
{
	count = 0;
	return beginBallPointer;
}

ball* ball::NextBall()
{
	return ++count == nballs ? NULL : this + 1;
}

bool ball::Collision(ball *jb, double cdist, vector3<double> nor)
{
	//ccontact *c;
	if (cdist <= 0) {
		std::map<ball*, ccontact>::iterator it = pcmap.find(jb);
		if (it != pcmap.end()){
			pcmap.erase(jb);
		}
		return false;
	}
	std::map<ball*, ccontact>::iterator it = pcmap.find(jb);
	if (it != pcmap.end()){
		ccontact *c = &(it->second);
		c->CalculateContactForces(cdist, nor);
	}
	else{
		ccontact c;
		c.SetIBall(this);
		c.SetJBall(jb);
		c.CalculateContactForces(cdist, nor);
		InsertPContact(jb, c);
	}
	return true;
}

void ball::InsertPContact(ball *jb, ccontact& ct)
{
	std::map<ball*, ccontact>::iterator it = pcmap.find(jb);
	if (it == pcmap.end())
		pcmap[jb] = ct;
}

void ball::InsertWContact(Geometry *wall, ccontact& ct)
{
	std::map<Geometry*, ccontact>::iterator it = wcmap.find(wall);
	if (it == wcmap.end())
		wcmap[wall] = ct;
}

void ball::InsertSContact(Geometry *shape, ccontact& ct)
{
	std::map<Geometry*, ccontact>::iterator it = scmap.find(shape);
	if (it == scmap.end())
		scmap[shape] = ct;
}

void ball::InsertOContact(Object *object, ccontact& ct)
{
	std::map<Object*, ccontact>::iterator it = ocmap.find(object);
	if (it == ocmap.end())
		ocmap[object] = ct;
}

double ball::OnlyNormalForceBySumation()
{
	double sumFn = 0;
	std::map<ball*, ccontact>::iterator pit = pcmap.begin();
	std::map<Geometry*, ccontact>::iterator wit = wcmap.begin();
	for (; pit != pcmap.end(); pit++){
		sumFn += pit->second.NormalForce().length();
	}

	for (; wit != wcmap.end(); wit++){
		sumFn += wit->second.NormalForce().length();
	}
	return sumFn;
}

double ball::GetNormalForceBySumation()
{
	double sumFn = 0;
	double rcp = 0;
	std::map<ball*, ccontact>::iterator pit = pcmap.begin();
	std::map<Geometry*, ccontact>::iterator wit = wcmap.begin();
	for (; pit != pcmap.end(); pit++){
		rcp = (pit->second.ContactPoint() - pos).length();
		sumFn += rcp * pit->second.NormalForce().length();
	}

	for (; wit != wcmap.end(); wit++){
		rcp = (wit->second.ContactPoint() - pos).length();
		sumFn += rcp * wit->second.NormalForce().length();
	}

	return sumFn;
}

double ball::DeltaIsotropicStress()
{
	double sumFd = 0;
	std::map<ball*, ccontact>::iterator pit = pcmap.begin();
	std::map<Geometry*, ccontact>::iterator wit = wcmap.begin();
	for (; pit != pcmap.end(); pit++)
	{
		double rcp = (pit->second.ContactPoint() - pos).length();
		double delta = radius + pit->second.JBall()->Radius();
		sumFd += rcp * pit->second.eKn() * delta;
	}
	for (; wit != wcmap.end(); wit++)
	{
		double rcp = (wit->second.ContactPoint() - pos).length();
		double delta = radius;
		sumFd += rcp * wit->second.eKn() * delta;
	}
	return sumFd;
}

