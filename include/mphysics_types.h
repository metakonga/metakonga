#ifndef MPHYSICS_TYPES_H
#define MPHYSICS_TYPES_H

#include <QStringList>

inline QStringList getMaterialList(){
	QStringList stList;
	stList.push_back("steel");
	stList.push_back("medium clay");
	stList.push_back("polyethylene");
	stList.push_back("glass");
	stList.push_back("acrylic");
	stList.push_back("aluminum");
	stList.push_back("sand");
	return stList;
}

#define STEEL_YOUNGS_MODULUS 2e+011
#define STEEL_DENSITY 7865
#define STEEL_POISSON_RATIO 0.3
#define STEEL_SHEAR_MODULUS 7.9E+10

#define MEDIUM_CLAY_YOUNGS_MODULUS 35E+06
#define MEDIUM_CLAY_DENSITY 1900
#define MEDIUM_CLAY_POISSON_RATIO 0.45

#define SAND_YOUNGS_MODULUS 40E+06
#define SAND_DENSITY 2600
#define SAND_POISSON_RATIO 0.3
#define SAND_SHEAR_MODULUS 4.3E+10

#define POLYETHYLENE_YOUNGS_MODULUS 1.1E+9
#define POLYETHYLENE_DENSITY		950
#define POLYETHYLENE_POISSON_RATIO	0.42	

#define GLASS_YOUNGS_MODULUS	6.8e+10
#define GLASS_DENSITY		2180
#define GLASS_POISSON_RATIO	0.19

#define ACRYLIC_YOUNGS_MODULUS 3.2E+009
#define ACRYLIC_DENSITY			1185
#define ACRYLIC_POISSON_RATIO	0.37

#define ALUMINUM_YOUNGS_MODULUS  7.1E+10
#define ALUMINUM_DENSITY		2770
#define ALUMINUM_POISSON_RATIO	0.33

enum tKinematicConstraint { CONSTRAINT = 0, REVOLUTE };
enum tMaterial { STEEL = 0, MEDIUM_CLAY, POLYETHYLENE, GLASS, ACRYLIC, ALUMINUM, SAND };
enum tRoll{ NO_DEFINE_ROLL = -1, ROLL_BOUNDARY = 0, ROLL_PARTICLE, ROLL_MOVING };
enum tObject{ CUBE = 0, PLANE, POLYGON, CYLINDER, PARTICLES };
enum tSimulation{ DEM, SPH, MBD };
enum tCollisionPair{ NO_COLLISION_PAIR = 0, PARTICLES_CUBE, PARTICLES_PARTICLES, PARTICLES_PLANE, PARTICLES_CYLINDER, PARTICLES_POLYGONOBJECT };
enum tFileType{ BIN = 0, MDE, PAR, TXT };
enum tChangeType{ CHANGE_PARTICLE_POSITION = 0};
enum tContactModel { HMCM = 0 };
enum tSolveDevice { CPU = 0, GPU };
enum tUnit { MKS = 0, MMKS };
enum tGravity { PLUS_X = 0, PLUS_Y, PLUS_Z, MINUS_X, MINUS_Y, MINUS_Z };
enum tImport {NO_FORMAT = 0, MILKSHAPE_3D_ASCII };

typedef struct
{
	float kn, vn, ks, vs, mu;
}constant;



inline tCollisionPair getCollisionPair(tObject o1, tObject o2)
{
	if (o1 == PARTICLES){
		if (o2 == PLANE)
			return PARTICLES_PLANE;
	}
	else if (o2 == PARTICLES){
		if (o1 == PLANE)
			return PARTICLES_PLANE;
	}
	
	if (o1 == PARTICLES){
		if (o2 == CYLINDER)
			return PARTICLES_CYLINDER;
	}
	else if (o2 == PARTICLES){
		if (o1 == CYLINDER)
			return PARTICLES_CYLINDER;
	}

	if (o1 == PARTICLES){
		if (o2 == POLYGON)
			return PARTICLES_POLYGONOBJECT;
	}
	else if (o2 == PARTICLES){
		if (o1 == POLYGON)
			return PARTICLES_POLYGONOBJECT;
	}

	return NO_COLLISION_PAIR;
}

namespace material
{
	// 	template<typename T>
	// 	cMaterial<T> getMaterialCoefficient(Material t)
	// 	{
	// 		cMaterial<T> cm;
	// 		cm.density = getDensity(t);
	// 		cm.youngs = getYoungs(t);
	// 		cm.poisson = getPoisson(t);
	// 		return cm;
	// 	}

	inline float getDensity(tMaterial t)
	{
		float v;
		switch (t)
		{
		case STEEL:			v = (float)STEEL_DENSITY;			break;
		case MEDIUM_CLAY:	v = (float)MEDIUM_CLAY_DENSITY;		break;
		case POLYETHYLENE:	v = (float)POLYETHYLENE_DENSITY;	break;
		case GLASS:			v = (float)GLASS_DENSITY;			break;
		case ACRYLIC:		v = (float)ACRYLIC_DENSITY;			break;
		case ALUMINUM:		v = (float)ALUMINUM_DENSITY;		break;
		case SAND:			v = (float)SAND_DENSITY;			break;
		}
		return v;
	}

	inline float getYoungs(tMaterial t)
	{
		float v;
		switch (t)
		{
		case STEEL:			v = (float)STEEL_YOUNGS_MODULUS;		break;
		case MEDIUM_CLAY:	v = (float)MEDIUM_CLAY_YOUNGS_MODULUS;	break;
		case POLYETHYLENE:	v = (float)POLYETHYLENE_YOUNGS_MODULUS;	break;
		case GLASS:			v = (float)GLASS_YOUNGS_MODULUS;		break;
		case ACRYLIC:		v = (float)ACRYLIC_YOUNGS_MODULUS;		break;
		case ALUMINUM:		v = (float)ALUMINUM_YOUNGS_MODULUS;		break;
		case SAND:			v = (float)SAND_YOUNGS_MODULUS;			break;
		}
		return v;
	}

	inline float getPoisson(tMaterial t)
	{
		float v;
		switch (t)
		{
		case STEEL:			v = (float)STEEL_POISSON_RATIO;			break;
		case MEDIUM_CLAY:	v = (float)MEDIUM_CLAY_POISSON_RATIO;	break;
		case POLYETHYLENE:	v = (float)POLYETHYLENE_POISSON_RATIO;	break;
		case GLASS:			v = (float)GLASS_POISSON_RATIO;			break;
		case ACRYLIC:		v = (float)ACRYLIC_POISSON_RATIO;		break;
		case ALUMINUM:		v = (float)ALUMINUM_POISSON_RATIO;		break;
		case SAND:			v = (float)SAND_POISSON_RATIO;			break;
		}
		return v;
	}

	inline float getShearModulus(tMaterial t)
	{
		float v;
		switch (t)
		{
		case STEEL:			v = (float)STEEL_SHEAR_MODULUS;			break;
		case SAND:			v = (float)SAND_SHEAR_MODULUS;			break;
		//case MEDIUM_CLAY:	v = (float)MEDIUM_CLAY_POISSON_RATIO;	break;
		//case POLYETHYLENE:	v = (float)POLYETHYLENE_POISSON_RATIO;	break;
		//case GLASS:			v = (float)GLASS_POISSON_RATIO;			break;
		//case ACRYLIC:		v = (float)ACRYLIC_POISSON_RATIO;		break;
		//case ALUMINUM:		v = (float)ALUMINUM_POISSON_RATIO;		break;
		}
		return v;
	}
}



#endif