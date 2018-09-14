#ifndef VMARKER_H
#define VMARKER_H

#include "vobject.h"

class QTextStream;

class vmarker : public vobject
{
public:
	vmarker();
	vmarker(QString& _name, bool mcf = true);
	//vmarker(QTextStream& in);
	virtual ~vmarker();

	virtual void draw(GLenum eMode);

	//void setMarkerScale
	bool define(VEC3D p);
	void setMarkerScaleFlag(bool b) { markerScaleFlag = b; }
	static void setMarkerScale(float sc) { scale = sc; }
	//bool makeCubeGeometry(QTextStream& in);
	//bool makeCubeGeometry(QString& _name, geometry_use _tr, material_type _tm, VEC3F& _mp, VEC3F& _sz);

private:
//	void setIndexList();
//	void setNormalList();
	bool markerScaleFlag;
	static float scale;
	unsigned int glList;
	//static float icon_scale;
	//unsigned int glHiList;
//	float loc[3];
// 	int indice[24];
// 	float vertice[24];
// 	float normal[18];
};

#endif