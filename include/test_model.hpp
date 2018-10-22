#ifndef PARTICLE_TIRE_ADDITION_MODEL_HPP
#define PARTICLE_TIRE_ADDITION_MODEL_HPP

#include "mbd_model.h"

class particle_tire_addition_model// : public mbd_model
{
public:
	particle_tire_addition_model(mbd_model* _mbd)
		: mbd(_mbd)
	{

	}
	~particle_tire_addition_model()
	{

	}

	bool setUp()
	{
	//	model::setGravity(9.80665, MINUS_Y);

		pointMass* h_body = mbd->createPointMass(
			"h_body", 1.0,
			VEC3D(1.0, 1.0, 1.0),
			VEC3D(0.0, 0.0, 0.0),
			VEC3D(0.0, 0.0, 0.8));

		pointMass* v_body = mbd->createPointMass(
			"v_body", 1.0,
			VEC3D(1.0, 1.0, 1.0),
			VEC3D(0.0, 0.0, 0.0),
			VEC3D(0.0, 0.0, 0.3378));
		
		pointMass* ground = mbd->Ground();
		VEC3D spi = ground->toLocal(VEC3D(0.0, 0.0, 0.8) - ground->Position());
		VEC3D spj = h_body->toLocal(VEC3D(0.0, 0.0, 0.8) - h_body->Position());
		kinematicConstraint* h_trans = mbd->createKinematicConstraint(
			"h_trans", kinematicConstraint::TRANSLATIONAL,
			ground, spi, VEC3D(0, 1, 0), VEC3D(0, 0, -1),
			h_body, spj, VEC3D(0, 0, 1), VEC3D(0, 1, 0));

		spi = h_body->toLocal(h_body->Position() - h_body->Position());
		spj = v_body->toLocal(h_body->Position() - v_body->Position());
		kinematicConstraint* v_trans = mbd->createKinematicConstraint(
			"v_trans", kinematicConstraint::TRANSLATIONAL,
			h_body, spi, VEC3D(1, 0, 0), VEC3D(0, 1, 0),
			v_body, spj, VEC3D(0, 1, 0), VEC3D(-1, 0, 0));

		pointMass* wheel = mbd->PointMass("wheel_final");
		spi = v_body->toLocal(wheel->Position() - v_body->Position());
		spj = wheel->toLocal(wheel->Position() - wheel->Position());
		kinematicConstraint* rev = mbd->createKinematicConstraint(
			"rev", kinematicConstraint::REVOLUTE,
			v_body, spi, VEC3D(1, 0, 0), VEC3D(0, 0, 1),
			wheel, spj, VEC3D(0, 0, 1), VEC3D(-1, 0, 0));
//    		drivingConstraint* h_dc = mbd->createDrivingConstraint(
//    			"h_dc", h_trans, drivingConstraint::DRIVING_TRANSLATION, 0.0, 1.0);
// 		drivingConstraint* v_dc = mbd->createDrivingConstraint(
// 			"v_dc", v_trans, drivingConstraint::DRIVING_TRANSLATION, 0.0, -0.2);// drivingConstraint("h_dc");
// // 		//h_dc->define(h_trans, drivingConstraint::DRIVING_TRANSLATION, 0.0, 1.0);
// // 		drivingConstraint* r_dc = mbd->createDrivingConstraint(
// // 			"r_dc", rev, drivingConstraint::DRIVING_ROTATION, 0.0, 6.28);
//   		v_dc->setStartTime(0.2);
//  		r_dc->setStartTime(0.8);
		return true;
	}

private:
	mbd_model* mbd;
};

#endif