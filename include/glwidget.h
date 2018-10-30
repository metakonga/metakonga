#ifndef GLWIDGET_H
#define GLWIDGET_H

#include "vobject.h"
#include "vparticles.h"
#include "vpolygon.h"
#include "vmarker.h"
#include "contactConstant.h"
/*#include <QGLWidget>*/
#include <QMenu>
#include <list>
#include <QFile>
#include <QKeyEvent>

#include <QGraphicsSimpleTextItem>
#include <QGraphicsRectItem>

class cube;
class plane;
class polygonObject;
class cylinder;
class particle_system;
class vparticles;
class modeler;

#define SELECT_BUF_SIZE 512

GLuint makeCubeObject(int* index, float* vertex);

enum viewObjectType{
	ALL_DISPLAY,
	ONLY_FRAME,
	ONLY_PARTICLE
};

enum projectionType{
	ORTHO_PROJECTION = 0,
	PERSPECTIVE_PROJECTION
};

class GLWidget : public QGLWidget
{
	Q_OBJECT

public:
	GLWidget(int argc, char** argv, QWidget *parent = 0);
	~GLWidget();

	static GLWidget* GLObject();

//	void setModeler(modeler* _md) { md = _md; }
	void makeCube(cube* c);
	void makePlane(plane* p);
	void makeLine();
	void makeCylinder(cylinder* cy);
	void makeParticle(double* pos, unsigned int n);
	void makeParticle_f(float* pos, unsigned int n);
	vmarker* makeMarker(QString n, VEC3D p, bool mcf = true);
	vpolygon* makePolygonObject(QString _nm, import_shape_type t, QString file);
//	bool change(QString& fp, tChangeType ct, tFileType ft);
	void makeMassCoordinate(QString& _name);
	void drawReferenceCoordinate();
	void drawGroundCoordinate(GLenum eMode);
	void setStartingData(QMap<QString, v3epd_type> d);

	int xRotation() const { return xRot; }
	int yRotation() const { return yRot; }
	int zRotation() const { return zRot; }
	void fitView();
	void renderText(double x, double y, double z, const QString& str, QColor& c);
	float& getZoom() { return trans_z; }
	void setKeyState(bool s, int i) { keyID[i] = s; };
	void actionDelete(const QString& tg);
	std::list<parview::contactConstant>* ContactConstants(){ return NULL; }// &cconsts;	}
	void onAnimation() { isAnimation = true; }
	GLuint makePolygonObject(double* points, double* normals, int* indice, int size);
	void setViewObject(viewObjectType viewType) { votype = viewType; };
	int getWindowHeight() { return wHeight; }
	bool is_set_particle() { return isSetParticle; }
	void openMbd(QString& fl);
	void openResults(QStringList& fl);
	void ChangeDisplayOption(int oid);
	vobject* Object(QString n);
	QMap<QString, vobject*>& Objects() { return v_objs; }
	vparticles* vParticles() { return vp; }
	vobject* getVObjectFromName(QString name);
	vpolygon* getVPolyObjectFromName(QString name);
	QString selectedObjectName();// { return selectedObject->name(); }
	vobject* selectedObjectWithCast();
	projectionType changeProjectionViewMode() { protype = protype == ORTHO_PROJECTION ? PERSPECTIVE_PROJECTION : ORTHO_PROJECTION; return protype; }
//	bool changePaletteMode() { sketch.isSketching = sketch.isSketching ? false : true; return sketch.isSketching; }
	void glObjectClear();
	void sketchingMode();

	public slots:
	void setXRotation(int angle);
	void setYRotation(int angle);
	void setZRotation(int angle);
	void ShowContextMenu(const QPoint& pos);
	void setSketchSpace();

signals:
	void xRotationChanged(int angle);
	void yRotationChanged(int angle);
	void zRotationChanged(int angle);

protected:
	void processHits(unsigned int uHits, unsigned int *pBuffer);
	void drawObject(GLenum eMode);
	vobject* setSelectMarking(QString sn);
	void initializeGL();
	void paintGL();
	void resizeGL(int width, int height);
	void mousePressEvent(QMouseEvent *event);
	void mouseReleaseEvent(QMouseEvent *event);
	void mouseMoveEvent(QMouseEvent *event);
	void keyPressEvent(QKeyEvent *);
	void keyReleaseEvent(QKeyEvent *);
	void wheelEvent(QWheelEvent *);

private:
	void picking(int x, int y);
	float& verticalMovement() { return trans_y; }
	float& horizontalMovement() { return trans_x; }
	void setMaxViewPosition(float x, float y, float z);
	void setMinViewPosition(float x, float y, float z);
	//void DrawCartesianCoordinates(vector3<double>& pos, vector3<double>& angle);
	//GLuint makeCoordinate();
	
	//GLuint makePolygonObject(float* points, float* normals, int* indice, int size);
	void normalizeAngle(int *angle);
	int viewOption;
	bool isRtOpenFile;
	bool isSetParticle;
	GLuint coordinate;
	GLuint polygons[256];
	int numPolygons;
	int wWidth;
	int wHeight;
	int xRot;
	int yRot;
	int zRot;
	int unit;
	float gridSize;
	float moveScale;
	float ratio;
	//float zoom;
	float trans_x;
	float trans_y;
	float trans_z;

	float IconScale;

	//bool isSketching;
	bool zRotationFlag;
	bool onZoom;
	bool onRotation;
	bool keyID[256];
	bool selected[256];
	unsigned choose;
	bool LBOTTON;
	QPoint lastPos;
	int aFrame;
	QMap<QString, vobject*> selectedObjects;
	bool isAnimation;

	float times[1000];
	
	vmarker ref_marker;
	vmarker *ground_marker;
	viewObjectType votype;
	projectionType protype;
	QMap<QString, vobject*> v_objs;
	//QMap<QString, vpolygon*> v_pobjs;
	QMap<int, void*> v_wobjs;
	vparticles *vp;
	QStringList outputNameList;

// 	float maxViewPoint[3];
// 	float minViewPoint[3];

	float minView[3];
	float maxView[3];
	//float up[3];
	vobject* selectedObject;

signals:
	void mySignal();
	//void propertySignal(QString, context_object_type);
	void contextSignal(QString, context_menu);
};


#endif // GLWIDGET_H

