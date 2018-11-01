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
		//--------------------------------------------------------------------
		VEC3D ppap(0, 0, 0);
		//VEC3D ppap(0, 0, -0.5);
		//VEC3D ppap(0, 0, 0);

		//--------------------------------------------------------------------
		//Car Body data
		VEC3D CarBody_position(0, 0, 0);//
		//	VEC3D G_CarBody_Fix_location(0, 0, 3);
		CarBody_position = CarBody_position - ppap;

		///// Left-front suspension data
		////--------------------------------------------------------------------
		VEC3D LF_wheel_position(-0.98833311, -0.67800231, -0.16953447);//
		//VEC3D LF_wheel_position(0.0,0.0, 0.30);
		////-----------------------------------------------------------------
		VEC3D LF_SDW_position(-1.002881233, -0.3638407391, -0.28434301585);//
		VEC3D LF_SUW_position(-0.988333491315223, -0.3659467684, 0.001803237321);//
		VEC3D LF_SWN_position(-0.988724341652014, -0.53965043107446, -0.15682570920124);//

		VEC3D LF_N_SDW_SPH_location(-1.004045, -0.500955, -0.2863);//
		VEC3D LF_N_SUW_SPH_location(-0.98833, -0.49865, 0.01185);//
		VEC3D LF_N_W_REV_location(-0.98833311, -0.61864, -0.16953447);//
		VEC3D LF_CarBody_SUW_REV_location(-0.98833, -0.26923, -0.00453);//
		VEC3D LF_CarBody_SDW_REV_location(-0.99000, -0.20000, -0.28000);//
		VEC3D LF_SUS_UPPER_location(-0.98835, -0.30472, 0.27691);//
		VEC3D LF_SUS_LOWER_location(-0.98833, -0.49423, -0.16953);//

		LF_wheel_position = LF_wheel_position - ppap;
		LF_SDW_position = LF_SDW_position - ppap;
		LF_SUW_position = LF_SUW_position - ppap;
		LF_SWN_position = LF_SWN_position - ppap;

		LF_N_SDW_SPH_location = LF_N_SDW_SPH_location - ppap;
		LF_N_SUW_SPH_location = LF_N_SUW_SPH_location - ppap;
		LF_N_W_REV_location = LF_N_W_REV_location - ppap;
		LF_CarBody_SUW_REV_location = LF_CarBody_SUW_REV_location - ppap;
		LF_CarBody_SDW_REV_location = LF_CarBody_SDW_REV_location - ppap;
		LF_SUS_UPPER_location = LF_SUS_UPPER_location - ppap;
		LF_SUS_LOWER_location = LF_SUS_LOWER_location - ppap;

		//// Right-front suspension data
		VEC3D RF_wheel_position(-0.98833358, 0.69254348, -0.16953447);//
		VEC3D RF_SDW_position(-0.970664582816707, 0.372662720737703, -0.284622371548781);//
		VEC3D RF_SUW_position(-0.98833320, 0.38856849, 0.00252935);//
		VEC3D RF_SWN_position(-0.98872480, 0.55119029, -0.15682639);//

		VEC3D RF_N_SDW_SPH_location(-1.00405, 0.51250, -0.28630);//
		VEC3D RF_N_SUW_SPH_location(-0.98833, 0.51019, 0.01185);//
		VEC3D RF_N_W_REV_location(-0.98833, 0.50577, -0.16953);//
		VEC3D RF_CarBody_SUW_REV_location(-0.98833, 0.28077, -0.00453);//
		VEC3D RF_CarBody_SDW_REV_location(-0.99000, 0.22000, -0.28000);//
		VEC3D RF_SUS_UPPER_location(-0.98835, 0.31627, 0.27691);//
		VEC3D RF_SUS_LOWER_location(-0.98833, 0.50577, -0.16953);//

		RF_wheel_position = RF_wheel_position - ppap;
		RF_SDW_position = RF_SDW_position - ppap;
		RF_SUW_position = RF_SUW_position - ppap;
		RF_SWN_position = RF_SWN_position - ppap;

		RF_N_SDW_SPH_location = RF_N_SDW_SPH_location - ppap;
		RF_N_SUW_SPH_location = RF_N_SUW_SPH_location - ppap;
		RF_N_W_REV_location = RF_N_W_REV_location - ppap;
		RF_CarBody_SUW_REV_location = RF_CarBody_SUW_REV_location - ppap;
		RF_CarBody_SDW_REV_location = RF_CarBody_SDW_REV_location - ppap;
		RF_SUS_UPPER_location = RF_SUS_UPPER_location - ppap;
		RF_SUS_LOWER_location = RF_SUS_LOWER_location - ppap;

		// Left_Rear suspension data
		VEC3D LR_wheel_position(0.68166656, -0.67550230, -0.16953490);//
		VEC3D LR_SDW_position(0.669974376, -0.363761133, -0.284343661);//
		VEC3D LR_SUW_position(0.681666492, -0.377027294, 0.002529342);//
		VEC3D LR_SWN_position(0.68127567, -0.53965043, -0.15682572);//

		VEC3D LR_N_SDW_SPH_location(0.665955, -0.500955, -0.2863);//
		VEC3D LR_N_SUW_SPH_location(0.68167, -0.49865, 0.01185);//
		VEC3D LR_N_W_REV_location(0.68167, -0.61864, -0.16953);//
		VEC3D LR_CarBody_SUW_REV_location(0.68167, -0.26923, -0.00453);//
		VEC3D LR_CarBody_SDW_REV_location(0.68000, -0.20000, -0.28000);//
		VEC3D LR_SUS_UPPER_location(0.68165, -0.30472, 0.27691);//
		VEC3D LR_SUS_LOWER_location(0.68167, -0.49423, -0.16953);//

		LR_wheel_position = LR_wheel_position - ppap;
		LR_SDW_position = LR_SDW_position - ppap;
		LR_SUW_position = LR_SUW_position - ppap;
		LR_SWN_position = LR_SWN_position - ppap;

		LR_N_SDW_SPH_location = LR_N_SDW_SPH_location - ppap;
		LR_N_SUW_SPH_location = LR_N_SUW_SPH_location - ppap;
		LR_N_W_REV_location = LR_N_W_REV_location - ppap;
		LR_CarBody_SUW_REV_location = LR_CarBody_SUW_REV_location - ppap;
		LR_CarBody_SDW_REV_location = LR_CarBody_SDW_REV_location - ppap;
		LR_SUS_UPPER_location = LR_SUS_UPPER_location - ppap;
		LR_SUS_LOWER_location = LR_SUS_LOWER_location - ppap;

		// Right_Rear suspension data
		VEC3D RR_wheel_position(0.68166656, 0.69254347, -0.16953437);//
		VEC3D RR_SDW_position(0.699335414, 0.372662718, -0.28462237);//
		VEC3D RR_SUW_position(0.681666799, 0.388568493, 0.002529347);//
		VEC3D RR_SWN_position(0.68127521, 0.55119029, -0.15682640);//

		VEC3D RR_N_SDW_SPH_location(0.665955, 0.51255, -0.2863);//
		VEC3D RR_N_SUW_SPH_location(0.68167, 0.51019, 0.01185);//
		VEC3D RR_N_W_REV_location(0.68167, 0.63019, -0.16953);//
		VEC3D RR_CarBody_SUW_REV_location(0.68167, 0.28077, -0.00453);//
		VEC3D RR_CarBody_SDW_REV_location(0.68000, 0.22000, -0.28000);//
		VEC3D RR_SUS_UPPER_location(0.6814, 0.31627, 0.27691);//
		VEC3D RR_SUS_LOWER_location(0.68167, 0.50577, -0.16953);//

		RR_wheel_position = RR_wheel_position - ppap;
		RR_SDW_position = RR_SDW_position - ppap;
		RR_SUW_position = RR_SUW_position - ppap;
		RR_SWN_position = RR_SWN_position - ppap;

		RR_N_SDW_SPH_location = RR_N_SDW_SPH_location - ppap;
		RR_N_SUW_SPH_location = RR_N_SUW_SPH_location - ppap;
		RR_N_W_REV_location = RR_N_W_REV_location - ppap;
		RR_CarBody_SUW_REV_location = RR_CarBody_SUW_REV_location - ppap;
		RR_CarBody_SDW_REV_location = RR_CarBody_SDW_REV_location - ppap;
		RR_SUS_UPPER_location = RR_SUS_UPPER_location - ppap;
		RR_SUS_LOWER_location = RR_SUS_LOWER_location - ppap;

		// Steer bar data
		VEC3D LF_Steer_Bar_position(-1.07849775, -0.475986948, -0.15153643);//
		VEC3D RF_Steer_Bar_position(-1.074865046, 0.496608617, -0.153817101);//
		VEC3D LR_Steer_Bar_position(0.588287987, -0.401067701, -0.147345361);//
		VEC3D RR_Steer_Bar_position(0.588865418, 0.389442659, -0.148057525);//

		VEC3D LF_Nuckle_Steer_Bar_SPH_location(-1.076679343, -0.566562414, -0.14704955);//
		VEC3D LF_CarBody_Steer_Bar_SPH_location(-1.088333343, -0.224229414, -0.139534634);//

		VEC3D RF_Nuckle_Steer_Bar_SPH_location(-1.076679343, 0.578103586, -0.147404955);//
		VEC3D RF_CarBody_Steer_Bar_SPH_location(-1.088333343, 0.235770586, -0.139534634);//

		VEC3D LR_Nuckle_Steer_Bar_SPH_location(0.593320657, -0.566562414, -0.147404955);//
		VEC3D LR_CarBody_Steer_Bar_SPH_location(0.581666657, -0.224229414, -0.139534634);//

		VEC3D RR_Nuckle_Steer_Bar_SPH_location(0.593320657, 0.578103586, -0.147404955);//
		VEC3D RR_CarBody_Steer_Bar_SPH_location(0.581666657, 0.235770586, -0.139534634);//

		LF_Steer_Bar_position = LF_Steer_Bar_position - ppap;
		RF_Steer_Bar_position = RF_Steer_Bar_position - ppap;
		LR_Steer_Bar_position = LR_Steer_Bar_position - ppap;
		RR_Steer_Bar_position = RR_Steer_Bar_position - ppap;

		LF_Nuckle_Steer_Bar_SPH_location = LF_Nuckle_Steer_Bar_SPH_location - ppap;
		LF_CarBody_Steer_Bar_SPH_location = LF_CarBody_Steer_Bar_SPH_location - ppap;

		RF_Nuckle_Steer_Bar_SPH_location = RF_Nuckle_Steer_Bar_SPH_location - ppap;
		RF_CarBody_Steer_Bar_SPH_location = RF_CarBody_Steer_Bar_SPH_location - ppap;

		LR_Nuckle_Steer_Bar_SPH_location = LR_Nuckle_Steer_Bar_SPH_location - ppap;
		LR_CarBody_Steer_Bar_SPH_location = LR_CarBody_Steer_Bar_SPH_location - ppap;

		RR_Nuckle_Steer_Bar_SPH_location = RR_Nuckle_Steer_Bar_SPH_location - ppap;
		RR_CarBody_Steer_Bar_SPH_location = RR_CarBody_Steer_Bar_SPH_location - ppap;

		//CarBody
		pointMass* CarBody = createPointMass("CarBody", 500,
			VEC3D(552.20658885, 869.75107357, 1141.92974516),
			VEC3D(0.0, 0.0, 0.0),
			CarBody_position);//246326396293451,552,869,1141

		//	//CarBody fixed Joint

		//Steer Bar
		pointMass* LF_Steer_Bar = createPointMass(
			"LF_Steer_Bar", 1.287169489,
			VEC3D(0.012128691, 0.001044893, 0.011885976),
			VEC3D(0.0, 0.0, 0.0),
			LF_Steer_Bar_position);//

		pointMass* RF_Steer_Bar = createPointMass(
			"RF_Steer_Bar", 0.91919316,
			VEC3D(0.010556064, 0.000609178, 0.010344812),
			VEC3D(0.0, 0.0, 0.0),
			RF_Steer_Bar_position);//

		pointMass* LR_Steer_Bar = createPointMass(
			"LR_Steer_Bar ", 2.718351924,
			VEC3D(0.046944765, 0.001381257, 0.046654043),
			VEC3D(0.0, 0.0, 0.0),
			LR_Steer_Bar_position);//

		pointMass* RR_Steer_Bar = createPointMass(
			"RR_Steer_Bar ", 2.143980851,
			VEC3D(0.038878853, 0.000961079, 0.038628115),
			VEC3D(0.0, 0.0, 0.0),
			RR_Steer_Bar_position);//


		//Left -front suspension
		// (name, mass, diag_inertia, sym_inertia, position)
		//---------------------------------------------------------------------------------------------
		pointMass* LF_wheel = createPointMass(
			"LF_wheel", 15,
			VEC3D(0.5, 0.5, 0.5),
			VEC3D(0.0, 0.0, 0.0),
			LF_wheel_position);
		//LF_wheel->setVelocity(VEC3D(10, 0, 0));
		/*	pointMass* LF_wheel = createPointMass(
		"LF_wheel", 28.05721,
		VEC3D(0.51826379711, 0.79352386286, 0.51826381642),
		VEC3D(0.0, 0.0, 0.0),
		LF_wheel_position);*/
		//LF_wheel->setVelocity(VEC3D(1,0,0));

		//	---------------------------------------------------------------------------------------------

		pointMass* LF_SDW = createPointMass(
			"LF_Suspension_DownWishbon", 5.749217770295790,
			VEC3D(0.0754236819, 0.065836213674189, 0.13970432230489),
			VEC3D(0.0, 0.0, 0.0),
			LF_SDW_position);//

		pointMass* LF_SUW = createPointMass(
			"LF_Suspension_Upwishbon", 5.75592915686434,
			VEC3D(0.04446857945, 0.03732034159, 0.08030433475),
			VEC3D(0.0, 0.0, 0.0),
			LF_SUW_position);

		pointMass* LF_SWN = createPointMass(
			"LF_Suspension_Wheel_Nuckle", 4.15873758871336,
			VEC3D(0.02145225448, 0.02247639296, 0.00463949401),
			VEC3D(0.0, 0.0, 0.0),
			LF_SWN_position);



		// Suspension joint and Spring_LF, RF, LR, RR
		VEC3D spi = LF_wheel->toLocal(LF_N_W_REV_location - LF_wheel_position);
		VEC3D spj = LF_SWN->toLocal(LF_N_W_REV_location - LF_SWN_position);

		// (name, type, ibody, local from i, P point of i, Q point of i, jbody, local from j,  P point of j, Q point of j)
		kinematicConstraint* LF_WHEEL_NUCKLE_REV = createKinematicConstraint(
			"LF_Wheel_Nuckle_Revolution", kinematicConstraint::REVOLUTE,
			LF_wheel, spi, VEC3D(1, 0, 0), VEC3D(0, 0, 1),
			LF_SWN, spj, VEC3D(-1, 0, 0), VEC3D(0, 0, 1));

		spi = LF_SWN->toLocal(LF_N_SUW_SPH_location - LF_SWN_position);
		spj = LF_SUW->toLocal(LF_N_SUW_SPH_location - LF_SUW_position);
		kinematicConstraint* LF_N_SUW_SPH_R = createKinematicConstraint(
			"LF_Nuckle_SUW_Spherical", kinematicConstraint::SPHERICAL,
			LF_SWN, spi, VEC3D(1, 0, 0), VEC3D(0, 1, 0),
			LF_SUW, spj, VEC3D(-1, 0, 0), VEC3D(0, 1, 0));

		spi = LF_SWN->toLocal(LF_N_SDW_SPH_location - LF_SWN_position);
		spj = LF_SDW->toLocal(LF_N_SDW_SPH_location - LF_SDW_position);
		kinematicConstraint* LF_N_SDW_SPH_R = createKinematicConstraint(
			"LF_Nuckle_SDW_Spherical", kinematicConstraint::SPHERICAL,
			LF_SWN, spi, VEC3D(1, 0, 0), VEC3D(0, 1, 0),
			LF_SDW, spj, VEC3D(-1, 0, 0), VEC3D(0, 1, 0));

		spi = CarBody->toLocal(LF_CarBody_SUW_REV_location - CarBody_position);
		spj = LF_SUW->toLocal(LF_CarBody_SUW_REV_location - LF_SUW_position);
		kinematicConstraint* LF_CarBody_SUW_REV = createKinematicConstraint(
			"LF_CarBody_SUW_Revolution", kinematicConstraint::REVOLUTE,
			CarBody, spi, VEC3D(0, 1, 0), VEC3D(0, 0, 1),
			LF_SUW, spj, VEC3D(0, 1, 0), VEC3D(0, 0, -1));

		spi = CarBody->toLocal(LF_CarBody_SDW_REV_location - CarBody_position);
		spj = LF_SDW->toLocal(LF_CarBody_SDW_REV_location - LF_SDW_position);
		kinematicConstraint* LF_CarBody_SDW_F = createKinematicConstraint(
			"LF_CarBody_SDW_Revolution", kinematicConstraint::REVOLUTE,
			CarBody, spi, VEC3D(0, 1, 0), VEC3D(0, 0, 1),
			LF_SDW, spj, VEC3D(0, 1, 0), VEC3D(0, 0, -1));

		// (name, ibody, global loaction of i, jbody, global location of j, k, c)


		//Right-front suspension
		// (name, mass, diag_inertia, sym_inertia, position)
		//---------------------------------------------------------------------------------------------
		pointMass* RF_wheel = createPointMass(
			"RF_wheel", 15,
			VEC3D(0.5, 0.5, 0.5),
			VEC3D(0.0, 0.0, 0.0),
			RF_wheel_position);

		//pointMass* RF_wheel = createPointMass(
		//	"RF_wheel", 28.05721,
		//	VEC3D(0.51826380, 0.79352386, 0.51826382),
		//	VEC3D(0.0, 0.0, 0.0),
		//	RF_wheel_position);
		//---------------------------------------------------------------------------------------------
		pointMass* RF_SDW = createPointMass(
			"RF_Suspension_DownWishbon", 7.75290911388722,
			VEC3D(0.09485987, 0.09042979, 0.18342764),
			VEC3D(0.0, 0.0, 0.0),
			RF_SDW_position);//

		pointMass* RF_SUW = createPointMass(
			"RF_Suspension_Upwishbon", 5.16427580529021,
			VEC3D(0.03814834, 0.03106566, 0.06790762),
			VEC3D(0.0, 0.0, 0.0),
			RF_SUW_position);//

		pointMass* RF_SWN = createPointMass(
			"RF_Suspension_Wheel_Nuckle", 4.1589475304493,
			VEC3D(0.02145214, 0.02247661, 0.00464018),
			VEC3D(0.0, 0.0, 0.0),
			RF_SWN_position);//

		spi = RF_wheel->toLocal(RF_N_W_REV_location - RF_wheel_position);
		spj = RF_SWN->toLocal(RF_N_W_REV_location - RF_SWN_position);
		// (name, type, ibody, local from i, P point of i, Q point of i, jbody, local from j,  P point of j, Q point of j)
		kinematicConstraint* RF_WHEEL_NUCKLE_REV = createKinematicConstraint(
			"RF_Wheel_Nuckle_Revolution", kinematicConstraint::REVOLUTE,
			RF_wheel, spi, VEC3D(1, 0, 0), VEC3D(0, 0, 1),
			RF_SWN, spj, VEC3D(-1, 0, 0), VEC3D(0, 0, 1));

		spi = RF_SWN->toLocal(RF_N_SUW_SPH_location - RF_SWN_position);
		spj = RF_SUW->toLocal(RF_N_SUW_SPH_location - RF_SUW_position);
		kinematicConstraint* RF_N_SUW_SPH_R = createKinematicConstraint(
			"RF_Nuckle_SUW_Spherical", kinematicConstraint::SPHERICAL,
			RF_SWN, spi, VEC3D(1, 0, 0), VEC3D(0, 1, 0),
			RF_SUW, spj, VEC3D(-1, 0, 0), VEC3D(0, 1, 0));

		spi = RF_SWN->toLocal(RF_N_SDW_SPH_location - RF_SWN_position);
		spj = RF_SDW->toLocal(RF_N_SDW_SPH_location - RF_SDW_position);
		kinematicConstraint* RF_N_SDW_SPH_R = createKinematicConstraint(
			"RF_Nuckle_SDW_Spherical", kinematicConstraint::SPHERICAL,
			RF_SWN, spi, VEC3D(1, 0, 0), VEC3D(0, 1, 0),
			RF_SDW, spj, VEC3D(-1, 0, 0), VEC3D(0, 1, 0));

		spi = CarBody->toLocal(RF_CarBody_SUW_REV_location - CarBody_position);
		spj = RF_SUW->toLocal(RF_CarBody_SUW_REV_location - RF_SUW_position);
		kinematicConstraint* RF_CarBody_SUW_REV = createKinematicConstraint(
			"RF_CarBody_SUW_Revolution", kinematicConstraint::REVOLUTE,
			CarBody, spi, VEC3D(0, 1, 0), VEC3D(0, 0, 1),
			RF_SUW, spj, VEC3D(0, 1, 0), VEC3D(0, 0, -1));

		spi = CarBody->toLocal(RF_CarBody_SDW_REV_location - CarBody_position);
		spj = RF_SDW->toLocal(RF_CarBody_SDW_REV_location - RF_SDW_position);
		kinematicConstraint* RF_CarBody_SDW_F = createKinematicConstraint(
			"RF_CarBody_SDW_Revolution", kinematicConstraint::REVOLUTE,
			CarBody, spi, VEC3D(0, 1, 0), VEC3D(0, 0, 1),
			RF_SDW, spj, VEC3D(0, 1, 0), VEC3D(0, 0, -1));

		// (name, ibody, global loaction of i, jbody, global location of j, k, c)


		//Left-Rear suspension
		// (name, mass, diag_inertia, sym_inertia, position)

		//------------------------------------------------------------------------
		pointMass* LR_wheel = createPointMass(
			"LR_wheel", 15,
			VEC3D(0.5, 0.5, 0.5),
			VEC3D(0.0, 0.0, 0.0),
			LR_wheel_position);

		//
		//pointMass* LR_wheel = createPointMass(
		//	"LR_wheel", 28.05721,
		//	VEC3D(0.51826350, 0.79352363, 0.51826387),
		//	VEC3D(0.0, 0.0, 0.0),
		//	LR_wheel_position);

		//----------------------------------------------------------------------
		pointMass* LR_SDW = createPointMass(
			"LR_Suspension_DownWishbon", 5.74921731,
			VEC3D(0.07542368, 0.06583622, 0.13970432),
			VEC3D(0.0, 0.0, 0.0),
			LR_SDW_position);//

		pointMass* LR_SUW = createPointMass(
			"LR_Suspension_Upwishbon", 5.16427817822225,
			VEC3D(0.03814835, 0.03106567, 0.06790764),
			VEC3D(0.0, 0.0, 0.0),
			LR_SUW_position);//

		pointMass* LR_SWN = createPointMass(
			"LR_Suspension_Wheel_Nuckle", 4.15873758871336,
			VEC3D(0.02145226, 0.02247640, 0.00463949),
			VEC3D(0.0, 0.0, 0.0),
			LR_SWN_position);//

		spi = LR_wheel->toLocal(LR_N_W_REV_location - LR_wheel_position);
		spj = LR_SWN->toLocal(LR_N_W_REV_location - LR_SWN_position);

		// (name, type, ibody, local from i, P point of i, Q point of i, jbody, local from j,  P point of j, Q point of j)

		kinematicConstraint* LR_WHEEL_NUCKLE_REV = createKinematicConstraint(
			"LR_Wheel_Nuckle_Revolution", kinematicConstraint::REVOLUTE,
			LR_wheel, spi, VEC3D(1, 0, 0), VEC3D(0, 0, 1),
			LR_SWN, spj, VEC3D(-1, 0, 0), VEC3D(0, 0, 1));

		spi = LR_SWN->toLocal(LR_N_SUW_SPH_location - LR_SWN_position);
		spj = LR_SUW->toLocal(LR_N_SUW_SPH_location - LR_SUW_position);
		kinematicConstraint* LR_N_SUW_SPH_R = createKinematicConstraint(
			"LR_Nuckle_SUW_Spherical", kinematicConstraint::SPHERICAL,
			LR_SWN, spi, VEC3D(1, 0, 0), VEC3D(0, 1, 0),
			LR_SUW, spj, VEC3D(-1, 0, 0), VEC3D(0, 1, 0));

		spi = LR_SWN->toLocal(LR_N_SDW_SPH_location - LR_SWN_position);
		spj = LR_SDW->toLocal(LR_N_SDW_SPH_location - LR_SDW_position);
		kinematicConstraint* LR_N_SDW_SPH_R = createKinematicConstraint(
			"LR_Nuckle_SDW_Spherical", kinematicConstraint::SPHERICAL,
			LR_SWN, spi, VEC3D(1, 0, 0), VEC3D(0, 1, 0),
			LR_SDW, spj, VEC3D(-1, 0, 0), VEC3D(0, 1, 0));

		spi = CarBody->toLocal(LR_CarBody_SUW_REV_location - CarBody_position);
		spj = LR_SUW->toLocal(LR_CarBody_SUW_REV_location - LR_SUW_position);
		kinematicConstraint* LR_CarBody_SUW_REV = createKinematicConstraint(
			"LR_CarBody_SUW_Revolution", kinematicConstraint::REVOLUTE,
			CarBody, spi, VEC3D(0, 1, 0), VEC3D(0, 0, 1),
			LR_SUW, spj, VEC3D(0, 1, 0), VEC3D(0, 0, -1));

		spi = CarBody->toLocal(LR_CarBody_SDW_REV_location - CarBody_position);
		spj = LR_SDW->toLocal(LR_CarBody_SDW_REV_location - LR_SDW_position);
		kinematicConstraint* LR_CarBody_SDW_F = createKinematicConstraint(
			"LR_CarBody_SDW_Revolution", kinematicConstraint::REVOLUTE,
			CarBody, spi, VEC3D(0, 1, 0), VEC3D(0, 0, 1),
			LR_SDW, spj, VEC3D(0, 1, 0), VEC3D(0, 0, -1));

		// (name, ibody, global loaction of i, jbody, global location of j, k, c)


		// Right-Rear suspension
		// (name, mass, diag_inertia, sym_inertia, position)
		//------------------------------------------------------------------------------------
		pointMass* RR_wheel = createPointMass(
			"RR_wheel", 15,
			VEC3D(0.5, 0.5, 0.5),
			VEC3D(0.0, 0.0, 0.0),
			RR_wheel_position);

		////pointMass* RR_wheel = createPointMass(
		////	"RR_wheel", 28.05721,
		////	VEC3D(0.51826350, 0.79352363, 0.51826387),
		////	VEC3D(0.0, 0.0, 0.0),
		////	RR_wheel_position);
		////----------------------------------------------------------------------------
		pointMass* RR_SDW = createPointMass(
			"RR_Suspension_DownWishbon", 7.75290833,
			VEC3D(0.09485986, 0.09042979, 0.18342764
			),
			VEC3D(0.0, 0.0, 0.0),
			RR_SDW_position);//

		pointMass* RR_SUW = createPointMass(
			"RR_Suspension_Upwishbon", 5.164275805,
			VEC3D(0.03814834, 0.03106566, 0.06790762),
			VEC3D(0.0, 0.0, 0.0),
			RR_SUW_position);//

		pointMass* RR_SWN = createPointMass(
			"RR_Suspension_Wheel_Nuckle", 4.15894865961011,
			VEC3D(0.02145214, 0.02247662, 0.00464018
			),
			VEC3D(0.0, 0.0, 0.0),
			RR_SWN_position);//

		spi = RR_wheel->toLocal(RR_N_W_REV_location - RR_wheel_position);
		spj = RR_SWN->toLocal(RR_N_W_REV_location - RR_SWN_position);

		// (name, type, ibody, local from i, P point of i, Q point of i, jbody, local from j,  P point of j, Q point of j)
		kinematicConstraint* RR_WHEEL_NUCKLE_REV = createKinematicConstraint(
			"RR_Wheel_Nuckle_Revolution", kinematicConstraint::REVOLUTE,
			RR_wheel, spi, VEC3D(1, 0, 0), VEC3D(0, 0, 1),
			RR_SWN, spj, VEC3D(-1, 0, 0), VEC3D(0, 0, 1));

		spi = RR_SWN->toLocal(RR_N_SUW_SPH_location - RR_SWN_position);
		spj = RR_SUW->toLocal(RR_N_SUW_SPH_location - RR_SUW_position);
		kinematicConstraint* RR_N_SUW_SPH_R = createKinematicConstraint(
			"RR_Nuckle_SUW_Spherical", kinematicConstraint::SPHERICAL,
			RR_SWN, spi, VEC3D(1, 0, 0), VEC3D(0, 1, 0),
			RR_SUW, spj, VEC3D(-1, 0, 0), VEC3D(0, 1, 0));

		spi = RR_SWN->toLocal(RR_N_SDW_SPH_location - RR_SWN_position);
		spj = RR_SDW->toLocal(RR_N_SDW_SPH_location - RR_SDW_position);
		kinematicConstraint* RR_N_SDW_SPH_R = createKinematicConstraint(
			"RR_Nuckle_SDW_Spherical", kinematicConstraint::SPHERICAL,
			RR_SWN, spi, VEC3D(1, 0, 0), VEC3D(0, 1, 0),
			RR_SDW, spj, VEC3D(-1, 0, 0), VEC3D(0, 1, 0));

		spi = CarBody->toLocal(RR_CarBody_SUW_REV_location - CarBody->Position());
		spj = RR_SUW->toLocal(RR_CarBody_SUW_REV_location - RR_SUW_position);
		kinematicConstraint* RR_CarBody_SUW_REV = createKinematicConstraint(
			"RR_CarBody_SUW_Revolution", kinematicConstraint::REVOLUTE,
			CarBody, spi, VEC3D(0, 1, 0), VEC3D(0, 0, 1),
			RR_SUW, spj, VEC3D(0, 1, 0), VEC3D(0, 0, -1));

		spi = CarBody->toLocal(RR_CarBody_SDW_REV_location - CarBody_position);
		spj = RR_SDW->toLocal(RR_CarBody_SDW_REV_location - RR_SDW_position);
		kinematicConstraint* RR_CarBody_SDW_F = createKinematicConstraint(
			"RR_CarBody_SDW_Revolution", kinematicConstraint::REVOLUTE,
			CarBody, spi, VEC3D(0, 1, 0), VEC3D(0, 0, 1),
			RR_SDW, spj, VEC3D(0, 1, 0), VEC3D(0, 0, -1));

		/*(name, ibody, global loaction of i, jbody, global location of j, k, c)*/


		//Steer Bar constraint
		spi = CarBody->toLocal(LF_CarBody_Steer_Bar_SPH_location - CarBody_position);
		spj = LF_Steer_Bar->toLocal(LF_CarBody_Steer_Bar_SPH_location - LF_Steer_Bar_position);
		kinematicConstraint* LF_CarBody_Steer_Bar_SPH = createKinematicConstraint(
			"LF_CarBody_Steer_Bar_SPH", kinematicConstraint::SPHERICAL,
			CarBody, spi, VEC3D(1, 0, 0), VEC3D(0, 1, 0),
			LF_Steer_Bar, spj, VEC3D(-1, 0, 0), VEC3D(0, 1, 0));

		spi = LF_SWN->toLocal(LF_Nuckle_Steer_Bar_SPH_location - LF_SWN_position);
		spj = LF_Steer_Bar->toLocal(LF_Nuckle_Steer_Bar_SPH_location - LF_Steer_Bar_position);
		kinematicConstraint* LF_Nuckle_Steer_Bar_SPH = createKinematicConstraint(
			"LF_Nuckle_Steer_Bar_SPH", kinematicConstraint::SPHERICAL,
			LF_SWN, spi, VEC3D(1, 0, 0), VEC3D(0, 1, 0),
			LF_Steer_Bar, spj, VEC3D(-1, 0, 0), VEC3D(0, 1, 0));

		spi = CarBody->toLocal(RF_CarBody_Steer_Bar_SPH_location - CarBody_position);
		spj = RF_Steer_Bar->toLocal(RF_CarBody_Steer_Bar_SPH_location - RF_Steer_Bar_position);
		kinematicConstraint* RF_CarBody_Steer_Bar_SPH = createKinematicConstraint(
			"RF_CarBody_Steer_Bar_SPH", kinematicConstraint::SPHERICAL,
			CarBody, spi, VEC3D(1, 0, 0), VEC3D(0, 1, 0),
			RF_Steer_Bar, spj, VEC3D(-1, 0, 0), VEC3D(0, 1, 0));

		spi = RF_SWN->toLocal(RF_Nuckle_Steer_Bar_SPH_location - RF_SWN_position);
		spj = RF_Steer_Bar->toLocal(RF_Nuckle_Steer_Bar_SPH_location - RF_Steer_Bar_position);
		kinematicConstraint* RF_Nuckle_Steer_Bar_SPH = createKinematicConstraint(
			"RF_Nuckle_Steer_Bar_SPH", kinematicConstraint::SPHERICAL,
			RF_SWN, spi, VEC3D(1, 0, 0), VEC3D(0, 1, 0),
			RF_Steer_Bar, spj, VEC3D(-1, 0, 0), VEC3D(0, 1, 0));

		spi = CarBody->toLocal(LR_CarBody_Steer_Bar_SPH_location - CarBody_position);
		spj = LR_Steer_Bar->toLocal(LR_CarBody_Steer_Bar_SPH_location - LR_Steer_Bar_position);
		kinematicConstraint* LR_CarBody_Steer_Bar_SPH = createKinematicConstraint(
			"LR_CarBody_Steer_Bar_SPH", kinematicConstraint::SPHERICAL,
			CarBody, spi, VEC3D(1, 0, 0), VEC3D(0, 1, 0),
			LR_Steer_Bar, spj, VEC3D(-1, 0, 0), VEC3D(0, 1, 0));

		spi = LR_SWN->toLocal(LR_Nuckle_Steer_Bar_SPH_location - LR_SWN_position);
		spj = LR_Steer_Bar->toLocal(LR_Nuckle_Steer_Bar_SPH_location - LR_Steer_Bar_position);
		kinematicConstraint* LR_Nuckle_Steer_Bar_SPH = createKinematicConstraint(
			"LR_Nuckle_Steer_Bar_SPH", kinematicConstraint::SPHERICAL,
			LR_SWN, spi, VEC3D(1, 0, 0), VEC3D(0, 1, 0),
			LR_Steer_Bar, spj, VEC3D(-1, 0, 0), VEC3D(0, 1, 0));


		spi = CarBody->toLocal(RR_CarBody_Steer_Bar_SPH_location - CarBody_position);
		spj = RR_Steer_Bar->toLocal(RR_CarBody_Steer_Bar_SPH_location - RR_Steer_Bar_position);
		kinematicConstraint* RR_CarBody_Steer_Bar_SPH = createKinematicConstraint(
			"RR_CarBody_Steer_Bar_SPH", kinematicConstraint::SPHERICAL,
			CarBody, spi, VEC3D(1, 0, 0), VEC3D(0, 1, 0),
			RR_Steer_Bar, spj, VEC3D(-1, 0, 0), VEC3D(0, 1, 0));

		spi = RR_SWN->toLocal(RR_Nuckle_Steer_Bar_SPH_location - RR_SWN_position);
		spj = RR_Steer_Bar->toLocal(RR_Nuckle_Steer_Bar_SPH_location - RR_Steer_Bar_position);
		kinematicConstraint* RR_Nuckle_Steer_Bar_SPH = createKinematicConstraint(
			"RR_Nuckle_Steer_Bar_SPH", kinematicConstraint::SPHERICAL,
			RR_SWN, spi, VEC3D(1, 0, 0), VEC3D(0, 1, 0),
			RR_Steer_Bar, spj, VEC3D(-1, 0, 0), VEC3D(0, 1, 0));

		//ground Carbody fix joint
		springDamperModel* LF_SUS = createSpringDamperElement(
			"LF_Suspension", LF_SWN, LF_SUS_LOWER_location,
			CarBody, LF_SUS_UPPER_location, 20000, 2000);

		springDamperModel* RF_SUS = createSpringDamperElement(
			"RF_Suspension", RF_SWN, RF_SUS_LOWER_location,
			CarBody, RF_SUS_UPPER_location, 20000, 2000);

		springDamperModel* LR_SUS = createSpringDamperElement(
			"LR_Suspension", LR_SWN, LR_SUS_LOWER_location,
			CarBody, LR_SUS_UPPER_location, 20000, 2000);

		springDamperModel* RR_SUS = createSpringDamperElement(
			"RR_Suspension", RR_SWN, RR_SUS_LOWER_location,
			CarBody, RR_SUS_UPPER_location, 20000, 2000);

		// set velocity
// 		VEC3D InitVel;
// 		InitVel = { 10, 0, 0 };
// 		CarBody->setVelocity(InitVel);
// 		LF_Steer_Bar->setVelocity(InitVel);
// 		RF_Steer_Bar->setVelocity(InitVel);
// 		LR_Steer_Bar->setVelocity(InitVel);
// 		RR_Steer_Bar->setVelocity(InitVel);
// 		LF_wheel->setVelocity(InitVel);
// 		LF_SDW->setVelocity(InitVel);
// 		LF_SUW->setVelocity(InitVel);
// 		LF_SWN->setVelocity(InitVel);
// 		RF_wheel->setVelocity(InitVel);
// 		RF_SDW->setVelocity(InitVel);
// 		RF_SUW->setVelocity(InitVel);
// 		RF_SWN->setVelocity(InitVel);
// 		LR_wheel->setVelocity(InitVel);
// 		LR_SDW->setVelocity(InitVel);
// 		LR_SUW->setVelocity(InitVel);
// 		LR_SWN->setVelocity(InitVel);
// 		RR_wheel->setVelocity(InitVel);
// 		RR_SDW->setVelocity(InitVel);
// 		RR_SUW->setVelocity(InitVel);
// 		RR_SWN->setVelocity(InitVel);
		//spi = ground->toLocal(G_CarBody_Fix_location - ground->Position());
		//spj = CarBody->toLocal(G_CarBody_Fix_location - CarBody_position);

		kinematicConstraint* G_Car_FIX = createKinematicConstraint(
		"Car_Ground_Fix", kinematicConstraint::FIXED,
		ground, spi, VEC3D(1, 0, 0), VEC3D(0, 1, 0),
		CarBody, spj, VEC3D(0, 1, 0), VEC3D(-1, 0, 0));
		//-----------------------------------------------------------------------------
		/*spi = ground->toLocal(LF_N_W_REV_location - ground->Position());
		spj = LF_wheel->toLocal(LF_N_W_REV_location - LF_wheel_position);

		kinematicConstraint* LF_wheel_FIX = createKinematicConstraint(
		"LF_wheel_FIX", kinematicConstraint::FIXED,
		ground, spi, VEC3D(1, 0, 0), VEC3D(0, 1, 0),
		LF_wheel, spj, VEC3D(0, 1, 0), VEC3D(-1, 0, 0));

		spi = ground->toLocal(RF_N_W_REV_location - ground->Position());
		spj = RF_wheel->toLocal(RF_N_W_REV_location - RF_wheel_position);

		kinematicConstraint* RF_wheel_FIX = createKinematicConstraint(
		"RF_wheel_FIX", kinematicConstraint::FIXED,
		ground, spi, VEC3D(1, 0, 0), VEC3D(0, 1, 0),
		RF_wheel, spj, VEC3D(0, 1, 0), VEC3D(-1, 0, 0));

		spi = ground->toLocal(LR_N_W_REV_location - ground->Position());
		spj = LR_wheel->toLocal(LR_N_W_REV_location - LR_wheel_position);

		kinematicConstraint* LR_wheel_FIX = createKinematicConstraint(
		"LR_wheel_FIX", kinematicConstraint::FIXED,
		ground, spi, VEC3D(1, 0, 0), VEC3D(0, 1, 0),
		LR_wheel, spj, VEC3D(0, 1, 0), VEC3D(-1, 0, 0));

		spi = ground->toLocal(RR_N_W_REV_location - ground->Position());
		spj = RR_wheel->toLocal(RR_N_W_REV_location - RR_wheel_position);

		kinematicConstraint* RR_wheel_FIX = createKinematicConstraint(
		"RR_wheel_FIX", kinematicConstraint::FIXED,
		ground, spi, VEC3D(1, 0, 0), VEC3D(0, 1, 0),
		RR_wheel, spj, VEC3D(0, 1, 0), VEC3D(-1, 0, 0));*/

		//-----------------------------------------------------------------------------
// 		contactPair* LF_cp = createContactPair("LF_cp", LF_wheel, ground);
// 		LF_cp->setContactParameters(176400, 30000, 73600, 8094, 1, 0.8);
// 		contactPair* RF_cp = createContactPair("RF_cp", RF_wheel, ground);
// 		RF_cp->setContactParameters(176400, 30000, 73600, 8094, 1, 0.8);
// 		contactPair* LR_cp = createContactPair("LR_cp", LR_wheel, ground);
// 		LR_cp->setContactParameters(176400, 30000, 73600, 8094, 1, 0.8);
// 		contactPair* RR_cp = createContactPair("RR_cp", RR_wheel, ground);
// 		RR_cp->setContactParameters(176400, 30000, 73600, 8094, 1, 0.8);
		return true;

	}

private:

};

#endif
