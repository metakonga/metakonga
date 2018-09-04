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
	vobject();
	vobject(QString& _name);
	virtual ~vobject();

	void setInitialPosition(VEC3F ip) { pos0 = ip; }
	void setInitialAngle(VEC3F ia) { ang0 = ia; }
	void setCurrentPosition(VEC3F cp) { cpos = cp; }
	void setCurrentAngle(VEC3F ca) { cang = ca; }
	void animationFrame();
	void setResultData(unsigned int n);
	void insertResultData(unsigned int i, VEC3D& p, EPD& r);
	int ID() { return id; }
	QString& name() { return nm; }
	void setName(QString n) { nm = n; }
	void setDisplay(bool _dis) { display = _dis; }
	QColor color() { return clr; }
	static void msgBox(QString ch, QMessageBox::Icon ic);
	void copyCoordinate(GLuint _coord);
	void setDrawingMode(GLenum dm) { drawingMode = dm; }
	void setSelected(bool b) { isSelected = b; }
	virtual void draw(GLenum eMode) = 0;

protected:
	int id;
	bool isSelected;
	QString nm;			// object name
	GLuint coord;
	GLenum drawingMode;
	bool display;
	QColor clr;
	static int count;
	VEC3F pos0;
	VEC3F ang0;
	VEC3F cpos;
	VEC3F cang;
	VEC3D* outPos;
	EPD* outRot;
};

#endif