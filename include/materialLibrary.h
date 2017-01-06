#ifndef MATERIALLIBRARY_H
#define MATERIALLIBRARY_H

#include <QStringList>

// inline QStringList getMaterialList(){
// 	QStringList stList;
// 	stList.push_back("steel");
// 	stList.push_back("medium clay");
// 	stList.push_back("polyethylene");
// 	stList.push_back("glass");
// 	stList.push_back("acrylic");
// 	stList.push_back("aluminum");
// 	return stList;
// }

enum material_type{
	NO_MATERIAL = 0,
	STEEL = 1,
	MEDIUM_CLAY = 2,
	POLYETHYLENE = 3,
	GLASS = 4,
	ACRYLIC = 5,
	ALUMINUM = 6
};

#define STEEL_YOUNGS_MODULUS 2e+011
#define STEEL_DENSITY 7850
#define STEEL_POISSON_RATIO 0.3

#define MEDIUM_CLAY_YOUNGS_MODULUS 35E+06
#define MEDIUM_CLAY_DENSITY 1900
#define MEDIUM_CLAY_POISSON_RATIO 0.45

#define POLYETHYLENE_YOUNGS_MODULUS 1.1E+9
#define POLYETHYLENE_DENSITY		950
#define POLYETHYLENE_POISSON_RATIO	0.42	

#define GLASS_YOUNG_MODULUS	6.8e+10
#define GLASS_DENSITY		2180
#define GLASS_POISSON_RATIO	0.19

#define ACRYLIC_YOUNG_MODULUS 3.2E+009
#define ACRYLIC_DENSITY			1185
#define ACRYLIC_POISSON_RATIO	0.37

#define ALUMINUM_YOUNG_MODULUS  7.1E+10
#define ALUMINUM_DENSITY		2770
#define ALUMINUM_POISSON_RATIO	0.33

#endif