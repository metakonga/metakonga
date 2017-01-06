#ifndef CONTACT_HPP
#define CONTACT_HPP

#include "algebra/vector3.hpp"
#include "parRock_v2/ParallelBondProperty.hpp"

using namespace algebra;

namespace Object{
	template<typename base_type>
	class particle;

	template<typename base_type>
	class Geometry;
}

template<typename base_type>
class Contact
{
public:
	enum WITH{ WITH_PARTICLE, WITH_WALL, WITH_SHAPE };
	
public:
	Contact() 
		: ipar(0)
		, jpar(0)
		, ekn(0)
		, eks(0)
		, inhibit(false)
		//, wall(NULL)
		//, shape(NULL)
	{}
	Contact(const Contact& c)
		: ipar(c.ip)
		, jpar(c.jp)
		, ekn(c.ekn)
		, eks(c.eks)
		//, wall(c.wall)
		//, shape(c.shape)
		, inhibit(false)
		, cp(c.cp)
		, nforce(c.nforce)
		, sforce(c.sforce)
		, nmoment(c.nmoment)
		, smoment(c.smoment)
		, oldNormal(c.oldNormal)
		, newNormal(c.newNormal)
	{}
	~Contact() 
	{
		nforce = 0.0f;
		sforce = 0.0f;
		nmoment = 0.0f;
		smoment = 0.0f;
		oldNormal = 0.0f;
		newNormal = 0.0;
	}

	Contact* This() { return this; }
	Object::particle<base_type>* IParticle() { return ipar; }
	Object::particle<base_type>* JParticle() { return jpar; }

	WITH OtherContactElement() { return with; }

	//vector3<base_type> CalculateDistance()
	//{
	//	vector3<base_type> distance;
	//	switch (with)
	//	{
	//	case WITH_PARTICLE:
	//		distance = jpar->Position() - ipar->Position();
	//		break;
	//	case WITH_WALL:
	//		geometry->CalculateDistance(ipar->Position());
	//		break;
	//	}
	//	return distance;
	//}
	void CollisionP2P(Object::particle<base_type>* ip, Object::particle<base_type>* jp, base_type cdist, vector3<base_type>& nor)
	{
		with = WITH_PARTICLE;
		overlap = cdist;
		oldNormal = newNormal;
		newNormal = nor;
		ipar = ip;
		jpar = jp;
		cp = ipar->Position() + (ipar->Radius() - 0.5f * cdist) * newNormal;
		vector3<base_type> rvel = ( jpar->Velocity() + jpar->Omega().cross((cp - jpar->Position())) ) - ( ipar->Velocity() + ipar->Omega().cross((cp - ipar->Position())) );
		rvel_s = rvel - rvel.dot(newNormal) * newNormal;
		if (contact_stiffness_model == 'l'){
			ekn = ipar->Kn() * jpar->Kn() / (ipar->Kn() + jpar->Kn());
			eks = ipar->Ks() * jpar->Ks() / (ipar->Ks() + jpar->Ks());
			nforce = ekn * pow(overlap, 1.5f)  * newNormal;
			sforce += -eks * dt * rvel_s;
		}
		else{
			base_type _R = 2 * ipar->Radius() * jpar->Radius() / (ipar->Radius() + jpar->Radius());
			base_type _v = RockElement<base_type>::poissonRatio;
			base_type _G = RockElement<base_type>::shearModulus;
			ekn = (2 * _G * sqrt(2 * _R) / (3 * (1 - _v))) * sqrt(overlap);
			nforce += ekn * overlap * newNormal;
			eks = (pow(2 * _G * _G * 3 * (1 - _v) * _R, 1.0f / 3.0f) / (2 - _v)) * pow(nforce.length(), 1.0f / 3.0f);
			sforce += -eks * dt * rvel_s;
		}
	}

	void CollisionP2G(Object::particle<base_type>* ip, Object::Geometry<base_type>* og, base_type cdist, vector3<base_type>& nor)
	{
		with = WITH_WALL;
		overlap = cdist;
		oldNormal = newNormal;
		newNormal = nor;
		ipar = ip;
		jpar = NULL;
		geometry = og;
		cp = ipar->Position() + (ipar->Radius() - 0.5f * cdist) * newNormal;
		vector3<base_type> rvel = - (ipar->Velocity() + ipar->Omega().cross((cp - ipar->Position())));
		rvel_s = rvel - rvel.dot(newNormal) * newNormal;
		if (contact_stiffness_model == 'l'){
			ekn = ipar->Kn() * geometry->Kn() / (ipar->Kn() + geometry->Kn());
			eks = ipar->Ks() * geometry->Ks() / (ipar->Ks() + geometry->Ks());
			nforce = ekn * pow(overlap, 1.5f) * newNormal;
			sforce += -eks * dt * rvel_s;
		}
		else{
			base_type _R = ipar->Radius();
			base_type _v = RockElement<base_type>::poissonRatio;
			base_type _G = RockElement<base_type>::shearModulus;
			ekn = (2 * _G * sqrt(2 * _R) / (3 * (1 - _v))) * sqrt(overlap);
			nforce += ekn * overlap * newNormal;
			eks = (pow(2 * _G * _G * 3 * (1 - _v) * _R, 1.0f / 3.0f) / (2 - _v)) * pow(nforce.length(), 1.0f / 3.0f);
			sforce += -eks * dt * rvel_s;
		}
	}

public:
	WITH with;
	static base_type dt;
	static char contact_stiffness_model;

	bool inhibit;

	Object::particle<base_type>* ipar;
	Object::particle<base_type>* jpar;
	Object::Geometry<base_type>* geometry;

	base_type overlap;
	base_type ekn;
	base_type eks;
	vector3<base_type> rvel_s;
	vector3<base_type> cp;
	vector3<base_type> nforce;
	vector3<base_type> sforce;
	vector3<base_type> nmoment;
	vector3<base_type> smoment;
	vector3<base_type> oldNormal;
	vector3<base_type> newNormal;

	ParallelBondProperty<base_type> pbprop;
};

template<typename base_type> char Contact<base_type>::contact_stiffness_model = 'h';
template<typename base_type> base_type Contact<base_type>::dt = 0.0f;

#endif