#ifndef FULLCARMODEL_HPP
#define FULLCARMODEL_HPP

#include "mbd_model.h"

class FullCarModel : public mbd_model
{
public:
	FullCarModel()
		: mbd_model("FullCarModel")
	{

	}
	~FullCarModel()
	{

	}

	bool setUp()
	{
		model::setGravity(9.80665, MINUS_Z);
		VEC3D wheel_position(-0.98833311, -0.67800231, -0.16953447);
		VEC3D LF_SDW_position(-1.002881233, -0.3638407391, -0.28434301585);
		VEC3D LF_SUW_position(-0.988333491315223, -0.3659467684, 0.001803237321);
		VEC3D LF_SWN_position(-0.988724341652014, -0.53965043107446, -0.15682570920124);

		VEC3D N_SDW_SPH_location(-1.004045, -0.500955, -0.2863);
		VEC3D N_SUW_SPH_location(-0.98833, -0.49865, 0.01185);
		VEC3D N_W_REV_location(-0.98833311, -0.67800231, -0.16953447);
		VEC3D G_SUW_REV_location(-0.98833, -0.26923, -0.00453);
		VEC3D G_SDW_REV_location(-0.988455, -0.20423, -0.28453);

		VEC3D LF_SUS_UPPER_location(-0.98835, -0.30472, 0.27691);
		VEC3D LF_SUS_LOWER_location(-0.98833, -0.49423, -0.16953);

		rigidBody* wheel = createRigidBody(
			"wheel", 28.05721, 
			VEC3D(0.51826379711, 0.79352386286, 0.51826381642),
			VEC3D(0.0, 0.0, 0.0),
			wheel_position);

		rigidBody* LF_SDW = createRigidBody(
			"Suspension_Left_Front_DownWishbon", 5.749217770295790,
			VEC3D(0.04446857945, 0.03732034159, 0.08030433475),
			VEC3D(0.0, 0.0, 0.0), 
			LF_SDW_position);

		rigidBody* LF_SUW = createRigidBody(
			"Suspension_Left_Front_Upwishbon", 5.75592915686434,
			VEC3D(0.04446857945, 0.03732034159, 0.08030433475),
			VEC3D(0.0, 0.0, 0.0),
			LF_SUW_position);

		rigidBody* LF_SWN = createRigidBody(
			"Suspension_Left_Front_Wheel_Nuckle", 4.15873758871336,
			VEC3D(0.02145225448, 0.02247639296, 0.00463949401),
			VEC3D(0.0, 0.0, 0.0),
			LF_SWN_position);

		VEC3D spi = wheel->toLocal(N_W_REV_location - wheel_position);
		VEC3D spj = LF_SWN->toLocal(N_W_REV_location - LF_SWN_position);
		kinematicConstraint* WHEEL_NUCKLE_REV = createKinematicConstraint(
			"Wheel_Nuckle_Revolution", kinematicConstraint::REVOLUTE,
			wheel, spi, VEC3D(1, 0, 0), VEC3D(0, 0, 1),
			LF_SWN, spj, VEC3D(-1, 0, 0), VEC3D(0, 0, 1));

		spi = LF_SWN->toLocal(N_SUW_SPH_location - LF_SWN_position);
		spj = LF_SUW->toLocal(N_SUW_SPH_location - LF_SUW_position);
		kinematicConstraint* N_SUW_SPH_R = createKinematicConstraint(
			"Nuckle_SUW_Spherical", kinematicConstraint::SPHERICAL,
			LF_SWN, spi, VEC3D(1, 0, 0), VEC3D(0, 1, 0),
			LF_SUW, spj, VEC3D(-1, 0, 0), VEC3D(0, 1, 0));

		spi = LF_SWN->toLocal(N_SDW_SPH_location - LF_SWN_position);
		spj = LF_SDW->toLocal(N_SDW_SPH_location - LF_SDW_position);
		kinematicConstraint* N_SDW_SPH_R = createKinematicConstraint(
			"Nuckle_SDW_Spherical", kinematicConstraint::SPHERICAL,
			LF_SWN, spi, VEC3D(1, 0, 0), VEC3D(0, 1, 0),
			LF_SDW, spj, VEC3D(-1, 0, 0), VEC3D(0, 1, 0));

		spi = ground->toLocal(G_SUW_REV_location - ground->getPosition());
		spj = LF_SUW->toLocal(G_SUW_REV_location - LF_SUW_position);
		kinematicConstraint* G_SUW_REV = createKinematicConstraint(
			"Ground_SUW_Revolution", kinematicConstraint::REVOLUTE,
			ground, spi, VEC3D(0, 1, 0), VEC3D(0, 0, 1),
			LF_SUW, spj, VEC3D(0, 1, 0), VEC3D(0, 0, -1));

		spi = ground->toLocal(G_SDW_REV_location - ground->getPosition());
		spj = LF_SDW->toLocal(G_SDW_REV_location - LF_SDW_position);
		kinematicConstraint* G_SDW_F = createKinematicConstraint(
			"Ground_SDW_Revolution", kinematicConstraint::REVOLUTE,
			ground, spi, VEC3D(0, 1, 0), VEC3D(0, 0, 1),
			LF_SDW, spj, VEC3D(0, 1, 0), VEC3D(0, 0, -1));

		springDamperModel* LF_SUS = createSpringDamperElement(
			"Left_Front_Suspension", LF_SWN, LF_SUS_LOWER_location,
			ground, LF_SUS_UPPER_location, 10000.0, 40);
		return true;
	}

private:

};

#endif