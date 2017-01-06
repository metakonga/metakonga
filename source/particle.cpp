#include "particle.h"

using namespace parSIM;

particle::particle()
	: cfric(0)
	, isFloater(false)
	, xfix(false)
	, yfix(false)
	, rfix(false)
{

}

particle::~particle()
{

}

void particle::AppendContactRelation(contact_data& cdata)
{
	contacts.push_back(cdata);
}

void particle::CalculateRockForce(double dt)
{
	double Kn = 0, Ks = 0;
	if(!contacts.size())
	{
		moment = 0.0;
		return;
	}
	std::list<contact_data>::iterator cdata = contacts.begin();
	vector3<double> Fn_i, dFs_i, V_i, Xc_i, V_is, dU_is;
	vector3<double> tFs;
	for(; cdata != contacts.end(); cdata++){
		particle *parJ = cdata->j;
		Kn = parJ == NULL ? kn * 1.1 : kn * parJ->Kn() / (kn + parJ->Kn());
		Ks = parJ == NULL ? ks * 1.1 : ks * parJ->Ks() / (ks + parJ->Ks());
		Xc_i = pos + (radius - 0.5 * cdata->cdist) * cdata->normal;
		V_i = parJ ? (parJ->Velocity() + parJ->Omega().cross(Xc_i - parJ->Position())) - (vel + omega.cross(Xc_i - pos)) : - (vel + omega.cross(Xc_i - pos));
		V_is = V_i - V_i.dot(cdata->normal) * cdata->normal;
		dU_is = dt * V_is;
		dFs_i = -Ks * dU_is;
		//force_s = force_s + dFs_i;
		Fn_i = Kn * cdata->cdist * cdata->normal;
		force -= (Fn_i + dFs_i);
		moment -= (Xc_i - pos).cross(dFs_i);
	}
}

void particle::setInhibitFlageForContact()
{
	std::list<contact_data>::iterator cdata = contacts.begin();
	for(; cdata != contacts.end(); cdata++){
		cdata->inhibit = false;
		if(cdata->j){
			if(rfix){
				if(cdata->j->rfix){
					cdata->inhibit = true;
				}
			}
		}
	}
}

double particle::shrink(double target)
{
// 	unsigned int count = 0;
// 	double sum = 0.0;
// 	std::list<contact_data>::iterator cdata = contacts.begin();
// 	for(; cdata != contacts.end(); cdata++){
// 		sum += CalculateNormalForce();
// 	}
// 	if(count > 1){
// 		sum /= (double)count;
// 		if(sum > target){
// 
// 		}
// 	}
	return 0.0;
}

void particle::removeInhibitFlags()
{
	std::list<contact_data>::iterator cdata = contacts.begin();
	for(; cdata != contacts.end(); cdata++){
		cdata->inhibit = 0;
	}
}

double particle::CalculateNormalForce()
{
	double sum=0.0;
	std::list<contact_data>::iterator cdata = contacts.begin();
	for(; cdata != contacts.end(); cdata++){
		particle *op = cdata->j;
		double KN = op == NULL ? kn * 1.1 : kn * op->Kn() / (kn + op->Kn());
		vector3<double> Xc = pos + (radius - 0.5 * cdata->cdist) * cdata->normal;
		double rcp = (Xc - pos).length();
		double Fn = KN * cdata->cdist;
		sum += rcp * Fn;
	}
	return sum;
}

double particle::PreCalculateIsotropicStress()
{
	double sum=0.0;
	std::list<contact_data>::iterator cdata = contacts.begin();
	for(; cdata != contacts.end(); cdata++){
		particle *op = cdata->j;
		vector3<double> Xc = pos + (radius - 0.5 * cdata->cdist) * cdata->normal;
		double rcp = (Xc - pos).length();
		double delta = op == NULL ? radius : radius + op->Radius();
		sum += rcp * kn * delta;
	}
	return sum;
}