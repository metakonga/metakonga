#ifndef TYPES_H
#define TYPES_H

#include <string>
#include "materialLibrary.h"

#define NUM_INTEGRATOR 3

#ifndef MIN
#define MIN(a,b) ((a < b) ? a : b)
#endif
#ifndef MAX
#define MAX(a,b) ((a > b) ? a : b)
#endif
#define MAX_FRAME	2000
// #define POINTER(a) &a(0)
// #define POINTER3(a) &(a.x)
// #define POINTER4(a) &a.x

#define CLOSE_SAVE -1

enum geometry_type
{
	NO_GEOMETRY_TYPE = -1,
	CUBE = 0,
	PLANE = 1,
	LINE = 2,
	SPHERE = 3,
	SHAPE = 5,
	PARTICLES = 6,
	RECTANGLE = 7,
	OBJECT
};

enum geometry_use
{
	PARTICLE = 4,
	MASS = 6,
	BOUNDARY = 8
};

enum mass_type
{
	NO_MASS_TYPE = -1,
	RIGID_BODY = 0,
	DEFORM_BODY = 1
};

enum integrator_type
{
	EULER_METHOD = 0,
	VELOCITY_VERLET = 1,
	IMPLICIT_HHT = 2
};

enum device_type
{
	CPU,
	GPU
};

enum dimension_type
{
	DIM_2,
	DIM_3
};

enum precision_type
{
	SINGLE_PRECISION=1,
	DOUBLE_PRECISION=2
};

enum solver_type
{
	DEM = 0,
	SPH = 1,
	MBD = 2
};

template< typename T >
struct dependency_type
{
	bool dependency;
	T value;
};

// enum material_type
// {
// 	NO_MATERIAL = 0,
// 	STEEL=1,
// 	ACRYLIC=2,
// 	POLYETHYLENE=3,
// 	POLYSTYRENE=4,
// 	ROCK=5,
// 	MEDIUM_CLAY=6
// };
// 
// #define STEEL_YOUNGS_MODULUS 2e+011
// #define STEEL_DENSITY 7850
// #define STEEL_POISSON_RATIO 0.3
// 
// #define IRON_YOUNGS_MODULUS 1.1E+011
// #define IRON_DENSITY		7200
// #define IRON_POISSON_RATIO	0.28
// 
// #define MEDIUM_CLAY_YOUNGS_MODULUS 35E+06
// #define MEDIUM_CLAY_DENSITY 1900
// #define MEDIUM_CLAY_POISSON_RATIO 0.45
// 
// #define POLYETHYLENE_YOUNGS_MODULUS 0.2E+009
// #define POLYETHYLENE_DENSITY		1768
// #define POLYETHYLENE_POISSON_RATIO	0.46	
// 
// #define POLYSTYRENE_YOUNGS_MODULUS 3.25E+009
// #define POLYSTYRENE_DENSITY		   1050
// #define POLYSTYRENE_POISSON_RATIO  0.34	
// 
// #define GLASS_YOUNG_MODULUS	6.8e+10
// #define GLASS_DENSITY		2180
// #define GLASS_POISSON_RATIO	0.19
// 
// #define ACRYLIC_YOUNG_MODULUS 3.2E+009
// #define ACRYLIC_DENSITY			1185
// #define ACRYLIC_POISSON_RATIO	0.37

struct cmaterialType
{
	double density;
	double youngs;
	double poisson;
};

struct ccontactConstant
{
	double restitution;
	double friction;
	double ratio;
};

struct save_cube_info
{
	double px, py, pz;
	double sx, sy, sz;
};

struct save_cube_info_f
{
	float px, py, pz;
	float sx, sy, sz;
};

struct save_plane_info
{
	double p0x, p0y, p0z;
	double p1x, p1y, p1z;
	double p2x, p2y, p2z;
};

struct rock_properties
{
	std::string name;
	double max_diameter;
	double rmin;
	double rmax;
	double diameter_ratio;
	double porosity;
	double density;
	double youngs;
	double stiffness_ratio;
	double friction;
};

struct cement_properties
{
	std::string name;
	int bond_radius_multiplier;
	double youngs_modulus;
	double stiffness_ratio;
	double MTS_mean;
	double MTS_std_dev;
	double MSS_mean;
	double MSS_std_dev;
};

enum fileFormat
{
	ASCII = 0,
	BINARY = 1
};

enum color_type
{
	RED = 0,
	GREEN,
	BLUE
};

typedef struct  
{
	solver_type _svt;
	fileFormat _fmt;
}output_info;

/*template <typename T>*/
struct contact_coefficient_t
{
	float kn, vn, ks, vs, mu;
};
struct contact_coefficient
{
	double kn, vn, ks, vs, mu;
};

//template< typename T >
//int sign(T a)
//{
//	return a < 0 ? -1 : 1;
//}

inline material_type material_str2enum(std::string str)
{
	material_type mt;
	if(str == "acrylic"){
		mt = ACRYLIC;
	}
	else if(str == "steel"){
		mt = STEEL;
	}
	else if (str == "polyethylene"){
		mt = POLYETHYLENE;
	}
	else if(str == "no_material"){
		mt = NO_MATERIAL;
	}
	return mt;
}

inline std::string material_enum2str(int mt)
{
	std::string str;
	switch(mt){
	case ACRYLIC:	str = "acrylic";	break;
	case STEEL:		str = "steel";		break;
	}
	return str;
}

inline cmaterialType getMaterialConstant(int mt)
{
	cmaterialType cmt;
	switch (mt){
	case STEEL: cmt.density = STEEL_DENSITY; cmt.youngs = STEEL_YOUNGS_MODULUS; cmt.poisson = STEEL_POISSON_RATIO; break;
	case ACRYLIC: cmt.density = ACRYLIC_DENSITY; cmt.youngs = ACRYLIC_YOUNG_MODULUS; cmt.poisson = ACRYLIC_POISSON_RATIO; break;
	case POLYETHYLENE: cmt.density = POLYETHYLENE_DENSITY; cmt.youngs = POLYETHYLENE_YOUNGS_MODULUS; cmt.poisson = ACRYLIC_POISSON_RATIO; break;
	}

	return cmt;
}

inline contact_coefficient calculate_contact_coefficient(cmaterialType* pm
													   , cmaterialType* wm
													   , double p_rad
													   , double p_mass
													   , double cor, double w_cor, double mul)
{
	contact_coefficient cc;
	double effective_pm_youngs = (pm->youngs*wm->youngs)/(pm->youngs*(1 - wm->poisson*wm->poisson) + wm->youngs*(1 - pm->poisson*pm->poisson));
	double effective_youngs_modulus = pm->youngs / (2*(1 - pm->poisson*pm->poisson));
	double effective_radius = (p_rad * p_rad) / (p_rad + p_rad);
	double effective_mass = (p_mass * p_mass) / (p_mass + p_mass);
	if(wm)
		
	return cc;
}

#endif