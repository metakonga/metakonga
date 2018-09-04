#ifndef SLIDERCRANK3D_HPP
#define SLIDERCRANK3D_HPP

#include "mbd_model.h"

class SliderCrank3D : public mbd_model
{
public:
	SliderCrank3D()
		: mbd_model("SliderCrank3D")
	{

	}
	~SliderCrank3D()
	{

	}

	void setUp()
	{
	//	model::setGravity(9.80665, MINUS_Z);
// 		VEC3D crank_position(0.0, 1.0, 0.25);
// 		VEC3D rod_position(1.0, 0.5, 0.5);
// 		VEC3D slider_position(2.0, 0.0, 0.5);
// 		
// 		VEC3D G_CRANK_REV_Loc(0.0, 1.0, 0.0);
// 		VEC3D CRANK_ROD_SPH_Loc(0.0, 1.0, 0.5);
// 		VEC3D G_SLIDER_TRANS_Loc(2.0, 0.0, 0.5);
// 		VEC3D ROD_SLIDER_UNIV_Loc(2.0, 0.0, 0.5);
// 
// 		rigidBody* crank = createRigidBody("crank");
// 		crank->setPosition(crank_position);
// 		crank->setMass(7.7);
// 		crank->setDiagonalInertia(0.162, 0.162, 0.162);
// 
// 		rigidBody* rod = createRigidBody("rod");
// 		rod->setPosition(rod_position);
// 		rod->setMass(22);
// 		rod->setDiagonalInertia(1.842, 7.355,9.193);
// 
// 		rigidBody* slider = createRigidBody("slider");
// 		slider->setPosition(slider_position);
// 		slider->setMass(4.11);
// 		slider->setDiagonalInertia(0.00411, 0.00411, 0.00411);
// 
// 
// 		VEC3D spi = ground->toLocal(G_CRANK_REV_Loc - ground->getPosition());
// 		VEC3D spj = ground->toLocal(G_CRANK_REV_Loc - crank_position);
// 		kinematicConstraint* G_CRANK_REV = createKinematicConstraint(
// 			"G_CRANK_REV", kinematicConstraint::REVOLUTE,
// 			ground, spi, VEC3D(0, 1, 0), VEC3D(0, 0, 1),
// 			crank, spj, VEC3D(0, 1, 0), VEC3D(0, 0, -1));
// 
// 		spi = crank->toLocal(CRANK_ROD_SPH_Loc - crank_position);
// 		spj = rod->toLocal(CRANK_ROD_SPH_Loc - rod_position);
// 		kinematicConstraint* CRANK_ROD_SPH = createKinematicConstraint(
// 			"CRANK_ROD_SPH", kinematicConstraint::SPHERICAL,
// 			crank, spi, VEC3D(0, 1, 0), VEC3D(1, 0, 0),
// 			rod, spj, VEC3D(0, 1, 0), VEC3D(-1, 0, 0));
// 
// 		spi = ground->toLocal(G_SLIDER_TRANS_Loc - ground->getPosition());
// 		spj = slider->toLocal(G_SLIDER_TRANS_Loc - slider_position);
// 		kinematicConstraint* G_SLIDER_TRANS = createKinematicConstraint(
// 			"G_SLIDER_TRANS", kinematicConstraint::TRANSLATIONAL,
// 			ground, spi, VEC3D(0, 1, 0), VEC3D(0, 0, 1),
// 			slider, spj, VEC3D(0, 0, -1), VEC3D(0, 1, 0));
// 
// 		spi = rod->toLocal(ROD_SLIDER_UNIV_Loc - rod_position);
// 		spj = slider->toLocal(ROD_SLIDER_UNIV_Loc - slider_position);
// 		kinematicConstraint* ROD_SLIDER_UNIV = createKinematicConstraint(
// 			"ROD_SLIDER_UNIV", kinematicConstraint::UNIVERSAL,
// 			rod, spi, VEC3D(0, 1, 0), VEC3D(0, 0, 1),
// 			slider, spj, VEC3D(0, 1, 0), VEC3D(-1, 0, 0));
// 
// 		axialRotationForce* arf = createAxialRotationForce(
// 			"AFORCE0", ground, crank,
// 			VEC3D(0.0, 1.0, 0.0), VEC3D(1.0, 0.0, 0.0), 100.0);
// 
// 		springDamperModel* sdm = createSpringDamperElement(
// 			"TSDA1", slider, VEC3D(2.0, 0.0, 0.5),
// 			ground, VEC3D(0.0, 0.0, 0.5), 10000, 10);
// 
// 		// 		spi = LF_SWN->toLocal(N_SDW_SPH_R_location - LF_SWN_position);
// 		// 		spj = LF_SDW_R->toLocal(N_SDW_SPH_R_location - LF_SDW_R_position);
// 		// 		kinematicConstraint* N_SDW_SPH_R = createKinematicConstraint(
// 		// 			"Nuckle_SDW_R_Spherical", kinematicConstraint::SPHERICAL,
// 		// 			LF_SWN, spi, VEC3D(1, 0, 0), VEC3D(0, 1, 0),
// 		// 			LF_SDW_R, spj, VEC3D(-1, 0, 0), VEC3D(0, 1, 0));
// 		// 
// 		// 		spi = LF_SWN->toLocal(N_SUW_SPH_location - LF_SWN_position);
// 		// 		spj = LF_SUW->toLocal(N_SUW_SPH_location - LF_SUW_position);
// 		// 		kinematicConstraint* N_SUW_SPH_R = createKinematicConstraint(
// 		// 			"Nuckle_SUW_Spherical", kinematicConstraint::SPHERICAL,
// 		// 			LF_SWN, spi, VEC3D(1, 0, 0), VEC3D(0, 1, 0),
// 		// 			LF_SUW, spj, VEC3D(-1, 0, 0), VEC3D(0, 1, 0));
// 		// // 
// 		// 		spi = wheel->toLocal(N_W_REV_location - wheel_position);
// 		// 		spj = LF_SWN->toLocal(N_W_REV_location - LF_SWN_position);
// 		// 		kinematicConstraint* WHEEL_NUCKLE_REV = createKinematicConstraint(
// 		// 			"Wheel_Nuckle_Revolution", kinematicConstraint::REVOLUTE,
// 		// 			wheel, spi, VEC3D(1, 0, 0), VEC3D(0, 0, 1),
// 		// 			LF_SWN, spj, VEC3D(-1, 0, 0), VEC3D(0, 0, 1));
// 		// 
// 		// 		spi = ground->toLocal(G_SUW_REV_location - ground->getPosition());
// 		// 		spj = LF_SUW->toLocal(G_SUW_REV_location - LF_SUW_position);
// 		// 		kinematicConstraint* G_SUW_REV = createKinematicConstraint(
// 		// 			"Ground_SUW_Revolution", kinematicConstraint::REVOLUTE,
// 		// 			ground, spi, VEC3D(0, 1, 0), VEC3D(0, 0, 1),
// 		// 			LF_SUW, spj, VEC3D(0, 1, 0), VEC3D(0, 0, -1));
// 		// // 
// 		// 		spi = ground->toLocal(G_SDW_REV_F_location - ground->getPosition());
// 		// 		spj = LF_SDW_F->toLocal(G_SDW_REV_F_location - LF_SDW_F_position);
// 		// 		kinematicConstraint* G_SDW_F = createKinematicConstraint(
// 		// 			"Ground_SDW_F_Revolution", kinematicConstraint::REVOLUTE,
// 		// 			ground, spi, VEC3D(0, 1, 0), VEC3D(0, 0, 1),
// 		// 			LF_SDW_F, spj, VEC3D(0, 1, 0), VEC3D(0, 0, -1));
// 		// 
// 		// 		spi = ground->toLocal(G_SDW_REV_R_location - ground->getPosition());
// 		// 		spj = LF_SDW_R->toLocal(G_SDW_REV_R_location - LF_SDW_R_position);
// 		// 		kinematicConstraint* G_SDW_R = createKinematicConstraint(
// 		// 			"Ground_SDW_R_Revolution", kinematicConstraint::REVOLUTE,
// 		// 			ground, spi, VEC3D(0, 1, 0), VEC3D(0, 0, 1),
// 		// 			LF_SDW_R, spj, VEC3D(0, 1, 0), VEC3D(0, 0, -1));
// 
// 		// 		springDamperModel* LF_SUS = createSpringDamperElement(
// 		// 			"Left_Front_Suspension", LF_SWN, LF_SUS_LOWER_location, 
// 		// 			ground, LF_SUS_UPPER_location, 1000.0, 40);
	}

private:

};

#endif