#ifndef TYPES_H
#define TYPES_H

#include <QColor>
#include <QStringList>
#include "vectorTypes.h"

static QColor colors[10] = { QColor("cyan"), QColor("magenta"), QColor("red"),
QColor("darkRed"), QColor("darkCyan"), QColor("darkMagenta"),
QColor("green"), QColor("darkGreen"), QColor("yellow"),
QColor("blue") };

enum color_type
{ 
	CYAN = 0, MAGENTA, RED, DARKRED, DARKCYAN, DARKMAGENTA, GREEN, DARKGREEN, YELLOW, BLUE
};

#define RAD2DEG(r) r * 180.0 / M_PI 

#define DEFAULT_GRAVITY 9.80665

#define STEEL_YOUNGS_MODULUS 2e+011
#define STEEL_DENSITY 7850
#define STEEL_POISSON_RATIO 0.3
#define STEEL_SHEAR_MODULUS 0.0

#define MEDIUM_CLAY_YOUNGS_MODULUS 35E+06
#define MEDIUM_CLAY_DENSITY 1900
#define MEDIUM_CLAY_POISSON_RATIO 0.45
#define MEDIUM_SHEAR_MODULUS 0.0

#define POLYETHYLENE_YOUNGS_MODULUS 1.1E+9
#define POLYETHYLENE_DENSITY		950
#define POLYETHYLENE_POISSON_RATIO	0.42	
#define POLYETHYLENE_SHEAR_MODULUS 0.0

#define GLASS_YOUNG_MODULUS	6.8e+10
#define GLASS_DENSITY		2180
#define GLASS_POISSON_RATIO	0.19
#define GLASS_SHEAR_MODULUS 0.0

#define ACRYLIC_YOUNG_MODULUS 3.2E+009
#define ACRYLIC_DENSITY			1185
#define ACRYLIC_POISSON_RATIO	0.37
#define ACRYLIC_SHEAR_MODULUS 0.0

#define ALUMINUM_YOUNG_MODULUS  70.0E+9
#define ALUMINUM_DENSITY		2700
#define ALUMINUM_POISSON_RATIO	0.34
#define ALUMINUM_SHEAR_MODULUS 0.0

#define SAND_YOUNG_MODULUS 4.0E+7
#define SAND_DENSITY 2600
#define SAND_POISSON_RATIO 0.3
#define SAND_SHEAR_MODULUS 0.0

#define kor(str) QString::fromLocal8Bit(str)

inline QStringList getMaterialList(){
	QStringList stList;
	stList.push_back("steel");
	stList.push_back("medium clay");
	stList.push_back("polyethylene");
	stList.push_back("glass");
	stList.push_back("acrylic");
	stList.push_back("aluminum");
	stList.push_back("sand");
	stList.push_back("user input");
	return stList;
}

inline QStringList getPreDefinedMBDList()
{
	QStringList stList;
	//stList.push_back("SliderCrank3D");
	stList.push_back("FullCarModel");
	return stList;
}

inline QString getFileExtend(QString f)
{
	int begin = f.lastIndexOf(".");
	return f.mid(begin+1);
}

enum geometry_type{	
	NO_GEOMETRY_TYPE=-1, CUBE = 2, PLANE = 3, LINE = 5,	SPHERE = 7,
	POLYGON_SHAPE = 19,	PARTICLES = 13,	RECTANGLE = 17, CIRCLE = 23, CYLINDER = 27, OBJECT = 100
};
enum material_type{	
	NO_MATERIAL = -1, STEEL = 0,	MEDIUM_CLAY,
	POLYETHYLENE, GLASS, ACRYLIC, ALUMINUM, SAND, USER_INPUT
};
enum geometry_use{	NO_USE_TYPE=-1, PARTICLE=0,	MASS,	BOUNDAR_WALL };
enum mass_type{	NO_MASS_TYPE,	RIGID_BODY,	DEFORM_BODY };
enum integrator_type{ EULER_METHOD,	VELOCITY_VERLET,	IMPLICIT_HHT };
enum device_type{ CPU,	GPU };
enum dimension_type{ DIM2 = 3,	DIM3 = 7 };
enum precision_type{ SINGLE_PRECISION, DOUBLE_PRECISION };
enum solver_type{ DEM, SPH, MBD };
enum fileFormat{ ASCII, BINARY };
enum contactForce_type { DHS = 0 };
enum unit_type{ MKS=0, MMKS };
enum gravity_direction{ PLUS_X = 0, PLUS_Y, PLUS_Z, MINUS_X, MINUS_Y, MINUS_Z };
enum particle_type { GEOMETRY = 0, FLUID, FLOATING, BOUNDARY, DUMMY, GHOST, FREE_DUMMY, FLOATING_DUMMY, PARTICLE_TYPE_COUNT };
enum correction_type { CORRECTION = 0, GRADIENT_CORRECTION, KERNEL_CORRECTION, MIXED_CORRECTION };
enum kernel_type{ CUBIC_SPLINE = 0, QUADRATIC, QUINTIC, WENDLAND, GAUSS, MODIFIED_GAUSS };
enum boundary_type{ DUMMY_PARTICLE_METHOD };
enum import_shape_type { NO_SUPPORT_FORMAT = 0, MILKSHAPE_3D_ASCII, STL_ASCII };
enum context_object_type{ VIEW_OBJECT = 0, GEOMETRY_OBJECT, CONSTRAINT_OBJECT, CONTACT_OBJECT };
enum context_menu{ CONTEXT_PROPERTY = 0, CONTEXT_REFINEMENT };

typedef struct 
{
	double rest;
	double fric;
	double rfric;
	double coh;
	double sratio;
}contact_parameter;

typedef struct 
{
	double Ei, Ej;
	double pri, prj;
	double Gi, Gj;
}material_property_pair;

typedef struct
{
	double density;
	double youngs;
	double poisson;
	double shear;
}cmaterialType;

typedef struct 
{
	double restitution;
	double friction;
	double ratio;
}ccontactConstant;

typedef struct 
{
	double px, py, pz;
	double sx, sy, sz;
}save_cube_info;

typedef struct 
{
	float px, py, pz;
	float sx, sy, sz;
}save_cube_info_f;

typedef struct 
{
	double p0x, p0y, p0z;
	double p1x, p1y, p1z;
	double p2x, p2y, p2z;
}save_plane_info;

typedef struct 
{
	QString name;
	double max_diameter;
	double rmin;
	double rmax;
	double diameter_ratio;
	double porosity;
	double density;
	double youngs;
	double stiffness_ratio;
	double friction;
}rock_properties;

typedef struct 
{
	QString name;
	int bond_radius_multiplier;
	double youngs_modulus;
	double stiffness_ratio;
	double MTS_mean;
	double MTS_std_dev;
	double MSS_mean;
	double MSS_std_dev;
}cement_properties;

typedef struct
{
	double t;
	double p;
	double v;
}expression_info;

typedef struct
{
	int id;
	VEC3D P;
	VEC3D Q;
	VEC3D R;
	VEC3D V;
	VEC3D W;
	VEC3D N;
}host_polygon_info;

typedef struct
{
	int id;
	VEC3F P;
	VEC3F Q;
	VEC3F R;
	VEC3F V;
	VEC3F W;
	VEC3F N;
}host_polygon_info_f;

typedef struct
{
	VEC3D origin;
	VEC3D vel;
	VEC3D omega;
	EPD ep;
}host_polygon_mass_info;

typedef struct
{
	VEC3D v3;
	EPD ep;
}v3epd_type;

typedef struct  
{
	solver_type _svt;
	fileFormat _fmt;
}output_info;

typedef struct
{
	double time;
	double value;
}time_double;

typedef struct
{
	double rad;
	VEC3D p, q, r;
	VEC3D n;
}triangle_info;

/*template <typename T>*/
typedef struct 
{
	float kn, vn, ks, vs, mu;
}contact_coefficient_t;

typedef struct 
{
	double kn, vn, ks, vs, mu;
}contact_coefficient;

inline contact_coefficient calculate_contact_coefficient(cmaterialType* pm
													   , cmaterialType* wm
													   , double p_rad
													   , double p_mass
													   , double cor, double w_cor, double mul)
{
	contact_coefficient cc;
	//double effective_pm_youngs = (pm->youngs*wm->youngs)/(pm->youngs*(1 - wm->poisson*wm->poisson) + wm->youngs*(1 - pm->poisson*pm->poisson));
	//double effective_youngs_modulus = pm->youngs / (2*(1 - pm->poisson*pm->poisson));
	//double effective_radius = (p_rad * p_rad) / (p_rad + p_rad);
	//double effective_mass = (p_mass * p_mass) / (p_mass + p_mass);
	//if(wm)
	//	
	return cc;
}

inline material_type material_str2enum(QString str)
{
	material_type mt;
	if (str == "acrylic"){
		mt = ACRYLIC;
	}
	else if (str == "steel"){
		mt = STEEL;
	}
	else if (str == "polyethylene"){
		mt = POLYETHYLENE;
	}
	else if (str == "sand")
		mt = SAND;
	else if (str == "user input"){
		mt = USER_INPUT;
	}
	return mt;
}

inline QString material_enum2str(int mt)
{
	QString str;
	switch (mt){
	case ACRYLIC:	str = "acrylic";	break;
	case STEEL:		str = "steel";		break;
	case SAND:		str = "sand";		break;
	}
	return str;
}

inline cmaterialType getMaterialConstant(int mt)
{
	cmaterialType cmt;
	switch (mt){
	case STEEL: cmt.density = STEEL_DENSITY; cmt.youngs = STEEL_YOUNGS_MODULUS; cmt.poisson = STEEL_POISSON_RATIO; cmt.shear = STEEL_SHEAR_MODULUS; break;
	case ACRYLIC: cmt.density = ACRYLIC_DENSITY; cmt.youngs = ACRYLIC_YOUNG_MODULUS; cmt.poisson = ACRYLIC_POISSON_RATIO; cmt.shear = ACRYLIC_SHEAR_MODULUS; break;
	case POLYETHYLENE: cmt.density = POLYETHYLENE_DENSITY; cmt.youngs = POLYETHYLENE_YOUNGS_MODULUS; cmt.poisson = POLYETHYLENE_POISSON_RATIO; cmt.shear = POLYETHYLENE_SHEAR_MODULUS; break;
	case MEDIUM_CLAY: cmt.density = MEDIUM_CLAY_DENSITY; cmt.youngs = MEDIUM_CLAY_YOUNGS_MODULUS; cmt.poisson = MEDIUM_CLAY_POISSON_RATIO; cmt.shear = MEDIUM_SHEAR_MODULUS; break;
	case GLASS: cmt.density = GLASS_DENSITY; cmt.youngs = GLASS_YOUNG_MODULUS; cmt.poisson = GLASS_POISSON_RATIO; cmt.shear = GLASS_SHEAR_MODULUS; break;
	case ALUMINUM: cmt.density = ALUMINUM_DENSITY; cmt.youngs = ALUMINUM_YOUNG_MODULUS; cmt.poisson = ALUMINUM_POISSON_RATIO; cmt.shear = ALUMINUM_SHEAR_MODULUS; break;
	case SAND: cmt.density = SAND_DENSITY; cmt.youngs = SAND_YOUNG_MODULUS; cmt.poisson = SAND_POISSON_RATIO; cmt.shear = SAND_SHEAR_MODULUS; break;
	//case SAND: cmt.density = SAND_DENSITY; cmt.youngs = SAND_YOUNGS_MODULUS; cmt.poisson = SAND_POISSON_RATIO; break;
	}

	return cmt;
}

inline QStringList getPMResultString()
{
	QStringList stList;
	stList.push_back("PX"); stList.push_back("PY"); stList.push_back("PZ");
	stList.push_back("VX"); stList.push_back("VY"); stList.push_back("VZ");
	stList.push_back("RVX"); stList.push_back("RVY"); stList.push_back("RVZ");
	stList.push_back("AX"); stList.push_back("AY"); stList.push_back("AZ");
	stList.push_back("RAX"); stList.push_back("RAY"); stList.push_back("RAZ");
	return stList;
}

inline QStringList getRFResultString()
{
	QStringList stList;
	stList.push_back("FX"); stList.push_back("FY"); stList.push_back("FZ");
	stList.push_back("TX"); stList.push_back("TY"); stList.push_back("TZ");
	// 	stList.push_back("RVX"); stList.push_back("RVY"); stList.push_back("RVZ");
	// 	stList.push_back("AX"); stList.push_back("AY"); stList.push_back("AZ");
	// 	stList.push_back("RAX"); stList.push_back("RAY"); stList.push_back("RAZ");
	return stList;
}

template<typename T>
inline int nFit(int n, T d)
{
	QString ch = QString("%1").arg(d);
	int len = ch.length();
	int r = n - len;
	return r;
}


struct hardPoint{ QString name; VEC3D loc; };
struct gravity{ double x, y, z; };
struct transformationMatrix{ double a0, a1, a2, a3; };
struct neighborInfo{ unsigned int j; double W; VEC3D dp; VEC3D gradW; };
struct resultDataType
{
	double time;
	VEC3D pos, vel, acc;
	double ang, angv, anga;
};

struct jointResultDataType
{
	double time, fm, fx, fy, fz, rm, rx, ry, rz;
	VEC3D loc;
};

enum jointType{ NO_TYPE = 0, REVOLUTE, TRANSLATION, DRIVING, FIXEDJOINT };
enum coordinateType{ AXIS_X = 0, AXIS_Y, AXIS_Z };

typedef struct
{
	particle_type type;
	bool isFreeSurface;
	double pressure;
	VEC3D pos;
	VEC3D vel;
}opData;

typedef struct
{
	VEC3D position;
	VEC3D normal;
	VEC3D tangent;
}corner;

typedef struct
{
	double t;
	double p;
	double v;
}tExpression;

typedef struct
{
	unsigned int sid;
	unsigned int cnt;
	VEC3D iniVel;
	corner c1;
	corner c2;
	corner c3;
	bool inner;
}overlappingCorner;

typedef struct {
	unsigned int sid;
	unsigned int cnt;
	VEC3D p1;
	VEC3D p2;
	VEC3D t1;
	VEC3D t2;
}overlappingLine;

typedef struct
{
	int xMin, xMax, y, z, upDownDir;
	bool goLeft, goRight;
}queuedParticle;

typedef struct
{
	kernel_type kernel;
	bool correction;
	double h;
	double h_sq;
	double h_inv;
	double h_inv_sq;
	double h_inv_2;
	double h_inv_3;
	double h_inv_4;
	double h_inv_5;
}smoothing_kernel;

typedef struct
{
	bool enable;
	unsigned int frequency;
	double factor;
}particle_shift;

typedef struct
{
	bool enable;
	VEC3D direction;
	VEC3D limits;
	VEC3D velocity;
}periodicCondition;

typedef struct
{
	QString nm;
	unsigned int sid;
	unsigned int np;
}floatingBodyInfo;

typedef struct
{
	double t;
	VEC3D pos;
	VEC3D vel;
}structFloatingResult;

typedef struct
{
	unsigned int id, bid;
	double mass, rho, press;
	VEC3D pos, vel, wvel, avel;
}ghostParticle;

typedef struct
{
	bool _isOpenModel;
	QString m_path;
}checkOpenModel;

typedef struct
{
	bool enable;
	double alpha;
	double start_point;
	double length;
}wave_damping_condition;

#endif