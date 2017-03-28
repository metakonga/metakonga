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
#include "mphysics_types.h"
#include "mphysics_numeric.h"

static QColor colors[10] = { QColor("cyan"), QColor("magenta"), QColor("red"),
QColor("darkRed"), QColor("darkCyan"), QColor("darkMagenta"),
QColor("green"), QColor("darkGreen"), QColor("yellow"),
QColor("blue") };

class vobject
{
public:
	vobject();
	vobject(QString& _name);
	~vobject();

	void setResultData(unsigned int n);
	void insertResultData(unsigned int i, VEC3D& p, EPD& r);
	virtual void draw(GLenum eMode) = 0;
	int ID() { return id; }
	QString& name() { return nm; }
	void setDisplay(bool _dis) { display = _dis; }
	QColor color() { return clr; }
	static void msgBox(QString ch, QMessageBox::Icon ic);
	void copyCoordinate(GLuint _coord);

protected:
	int id;
	QString nm;			// object name
	GLuint coord;
	bool display;
	QColor clr;
	static int count;
	VEC3D* outPos;
	EPD* outRot;
};

#endif