#ifndef PARTICLE_H
#define PARTICLE_H

#include "algebra.h"
#include "geometry.h"
#include <list>

namespace parSIM
{
	class particle
	{
	public:
		typedef struct  
		{
			//contact_data();
			geometry* geo;
			particle *j;
			double cdist;
			vector3<double> normal;
			bool inhibit;
			double _R;
			double A;
			double I;
			double J;
		}contact_data;
	public:
		particle();
		~particle();

		void setKn(double _kn) { kn = _kn; }
		void setKs(double _ks) { ks = _ks; }
		void setPosition(vector3<double> _pos) { pos = _pos; }
		void setVelocity(vector3<double> _vel) { vel = _vel; }
		void setAcceleration(vector3<double> _acc) { acc = _acc; }
		void setOmega(vector3<double> _omega) { omega = _omega; }
		void setAlpha(vector3<double> _alpha) { alpha = _alpha; }
		void setForce(vector3<double> _force) { force = _force; }
		void setMoment(vector3<double> _moment) { moment = _moment; }
		void setRadius(double _radius) { radius = _radius; }
		void setMass(double _mass) { mass = _mass; }
		void setInertia(double _inertia) { inertia = _inertia; }

		vector3<double>& Position() { return pos; }
		vector3<double>& Velocity() { return vel; }
		vector3<double>& Acceleration() { return acc; }
		vector3<double>& Omega() { return omega; }
		vector3<double>& Alpha() { return alpha; }
		vector3<double>& Force() { return force; }
		vector3<double>& Moment() { return moment; }
		double& Radius() { return radius; }
		double& Mass() { return mass; }
		double& Inertia() { return inertia; }
		double& Kn() { return kn; }
		double& Ks() { return ks; }

		void AppendContactRelation(contact_data& cdata);
		double PreCalculateIsotropicStress();
		double CalculateNormalForce();
		void CalculateRockForce(double dt);
		void setInhibitFlageForContact();
		double shrink(double target);
		void removeInhibitFlags();

		std::list<contact_data> contacts;
		bool xfix;
		bool yfix;
		bool rfix;
		bool isFloater;
		double cfric;

	private:
		// particle information
		
		double radius;
		double mass;
		double inertia;
		//double cfric;
		vector3<double> pos;
		vector3<double> vel;
		vector3<double> acc;
		vector3<double> omega;
		vector3<double> alpha;
		vector3<double> force;
		vector3<double> force_s;
		vector3<double> moment;

		// contact parameters
		double kn;
		double ks;

		
	};
}

#endif