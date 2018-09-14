#ifndef VOBJECT_H
#define VOBJECT_H

// #ifndef QT_OPENGL_ES_2
 #include <gl/glew.h>
// #include <gl/glu.h>
// #endif

#include <QGLWidget>
#include <QString>
#include <QObject>
#include <QMessageBox>
#include <QFile>
#include <map>

#include <QColor>
#include "VController.h"
#include "types.h"
#include "algebraMath.h"

static QColor colors[10] = { QColor("cyan"), QColor("magenta"), QColor("red"),
QColor("darkRed"), QColor("darkCyan"), QColor("darkMagenta"),
QColor("green"), QColor("darkGreen"), QColor("yellow"),
QColor("blue") };

class vobject
{
public:
	enum viewGeometryObjectType{ VIEW_OBJECT = 0, GEOMETRY_OBJECT, CONSTRAINT_OBJECT };

	vobject();
	vobject(QString& _name);
	virtual ~vobject();

	void setInitialPosition(VEC3D ip) { pos0 = ip; }
	void setInitialAngle(VEC3D ia) { ang0 = ia; }
	void setCurrentPosition(VEC3D cp) { cpos = cp; }
	void setCurrentAngle(VEC3D ca) { cang = ca; }
	void animationFrame();
	void setResultData(unsigned int n);
	void insertResultData(unsigned int i, VEC3D& p, EPD& r);
	VEC3D InitialPosition() { return pos0; }
	int ID() { return id; }
	QString& name() { return nm; }
	void setName(QString n) { nm = n; }
	void setDisplay(bool _dis) { display = _dis; }
	QColor color() { return clr; }
	static void msgBox(QString ch, QMessageBox::Icon ic);
	void copyCoordinate(GLuint _coord);
	void setDrawingMode(GLenum dm) { drawingMode = dm; }
	void setSelected(bool b) { isSelected = b; }
	viewGeometryObjectType ViewGeometryObjectType() { return vot; }
	virtual void draw(GLenum eMode) = 0;

protected:
	int id;
	bool isSelected;
	viewGeometryObjectType vot;
	QString nm;			// object name
	GLuint coord;
	GLenum drawingMode;
	bool display;
	QColor clr;
	static int count;
	VEC3D pos0;
	VEC3D ang0;
	VEC3D cpos;
	VEC3D cang;
	VEC3D* outPos;
	EPD* outRot;
};

#endif