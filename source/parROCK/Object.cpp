#include "Object.h"
#include "Simulation.h"
#include "contact.h"

Object::Object(Simulation *_sim)
	: sim(_sim)
	, func(NULL)
	, updateProcess(false)
{

}

Object::~Object()
{

}

void Object::addLine(vector3<double>& sp, vector3<double>& ep, vector3<double>& nor)
{
	line l;
	l.sp = sp;
	l.ep = ep;
	l.nor = nor;
	lines.push(l);
}

void Object::define(std::string _name)
{
	if (lines.sizes()){
		lines.adjustment();
	}
	name = _name;
	vector3<double> point = vector3<double>(-0.001, 0.016, 0.0);
	points.push(point);
	points.adjustment();
	sim->Objects()[name] = this;
}

void Object::Collision(ball * ib)
{
	if (ib->ID() == 19 || ib->ID() == 20)
	{
		bool pause = true;
	}
	for (unsigned int i = 0; i < points.sizes(); i++){
		vector3<double> p = points(i);
		vector3<double> rp = ib->Position() - p;
		double dist = rp.length();
		double cdist = ib->Radius() - dist;
		vector3<double> nor = rp / dist;
		if (cdist > 0){
			std::map<Object*, ccontact>::iterator it = ib->ContactOMap().find(this);
			if (it != ib->ContactOMap().end()){
				ccontact* c = &(it->second);
				c->CalculateContactForces(cdist, nor, kn, ks, fric);
				return;
			}
			else{
				ccontact c;
				c.SetIBall(ib);
				c.SetJBall(NULL);
				c.SetWall(NULL);
				c.SetObject(this);
				c.CalculateContactForces(cdist, nor, kn, ks, fric);
				ib->InsertOContact(this, c);
				return;
			}
		}
	}
	for (unsigned int i = 0; i < lines.sizes(); i++){
		line l = lines(i);
		vector3<double> ab = l.ep - l.sp;
		double t = (ib->Position() - l.sp).dot(ab) / ab.dot();
		if (t < 0.0) t = 0.0;
		if (t > 1.0) t = 1.0;
		vector3<double> d = l.sp + t * ab;
		vector3<double> rp = ib->Position() - d;
		double dist = rp.length();
		double cdist = ib->Radius() - dist;
		if (cdist > 0){
			std::map<Object*, ccontact>::iterator it = ib->ContactOMap().find(this);
			if (it != ib->ContactOMap().end()){
				ccontact* c = &(it->second);
				c->CalculateContactForces(cdist, -l.nor, kn, ks, fric);
				return;
			}
			else{
				ccontact c;
				c.SetIBall(ib);
				c.SetJBall(NULL);
				c.SetWall(NULL);
				c.SetObject(this);
				c.CalculateContactForces(cdist, -l.nor, kn, ks, fric);
				ib->InsertOContact(this, c);
				return;
			}
		}
	}
	std::map<Object*, ccontact>::iterator it = ib->ContactOMap().find(this);
	if (it != ib->ContactOMap().end()){
		ib->ContactOMap().erase(this);
	}
}

void Object::Update(double time)
{
	if (!updateProcess)
		return;
	vector3<double> newCenter = func(time);
	vector3<double> diffCenter = newCenter - center;
	for (unsigned int i = 0; i < lines.sizes(); i++){
		line *l = &lines(i);
		l->sp += diffCenter;
		l->ep += diffCenter;
	}
	for (unsigned int i = 0; i < points.sizes(); i++){
		vector3<double> *pp = &points(i);
		*pp += diffCenter;
	}
	center = newCenter;
}

void Object::save2file(std::fstream& of, char ft)
{
	int type = OBJECT;
	if (ft == 'b'){
		of.write((char*)&type, sizeof(int));
		int name_size = name.size();
		of.write((char*)&name_size, sizeof(int));
		of.write((char*)name.c_str(), sizeof(char) * name_size);
		of.write((char*)&center.x, sizeof(double) * 3);
		of.write((char*)&lines.sizes(), sizeof(unsigned int));
		of.write((char*)lines.get_ptr(), sizeof(line)*lines.sizes());
		of.write((char*)&points.sizes(), sizeof(unsigned int));
		of.write((char*)points.get_ptr(), sizeof(vector3<double>) * points.sizes());
	}
	else if (ft == 'a'){

	}
	else{

	}
}