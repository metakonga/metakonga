#ifndef VOBJECT_H
#define VOBJECT_H

#ifndef QT_OPENGL_ES_2
#include <gl/glew.h>
#include <gl/glu.h>
#endif

#include <QString>
#include <QObject>
#include <QMessageBox>
#include <QFile>
#include <map>

#include <QColor>
#include "VController.h"
#include "mphysics_types.h"
#include "mphysics_numeric.h"

// QT_BEGIN_NAMESPACE
// class QLineEdit;
// class QComboBox;
// class QLabel;
// QT_END_NAMESPACE

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
// 	int roll_id() { return idr; }
// 	int material_id() { return idm; }
	QColor color() { return clr; }
	static void msgBox(QString ch, QMessageBox::Icon ic);
	void copyCoordinate(GLuint _coord);

protected:
	int id;
	QString nm;			// object name
	GLuint coord;
	bool display;
	QColor clr;
	//object* m_object;
	static int count;
	VEC3D* outPos;
	EPD* outRot;
};

#endif