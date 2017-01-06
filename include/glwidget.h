#ifndef GLWIDGET_H
#define GLWIDGET_H

#include "vobject.h"
#include "vparticles.h"
#include "vpolygon.h"
#include "contactConstant.h"
#include <QGLWidget>
#include <QMenu>
#include <list>
#include <QFile>

class cube;
class plane;
class polygonObject;
class cylinder;
class particle_system;
class vparticles;


#define SELECT_BUF_SIZE 512

GLuint makeCubeObject(int* index, float* vertex);

enum viewObjectType{
	ALL_DISPLAY,
	ONLY_FRAME,
	ONLY_PARTICLE
};

class GLWidget : public QGLWidget
{
	Q_OBJECT

public:
	GLWidget(int argc, char** argv, QWidget *parent = 0);
	~GLWidget();

	void makeCube(cube* c);
	void makePlane(plane* p);
	void makeLine();
	//void makePolygon(polygon* po);
	void makeCylinder(cylinder* cy);
	void makeParticle(particle_system* ps);
	void makePolygonObject(QMap<QString, polygonObject>& pObjs);
	void defineCollidConst();
	bool change(QString& fp, tChangeType ct, tFileType ft);
	void makeMassCoordinate(QString& _name);

	int xRotation() const { return xRot; }
	int yRotation() const { return yRot; }
	int zRotation() const { return zRot; }
	float& getZoom() { return zoom; }
	void setKeyState(bool s, int i) { keyID[i] = s; };

	std::list<parview::contactConstant>* ContactConstants(){ return NULL; }// &cconsts;	}
	void onAnimation() { isAnimation = true; }
	GLuint makePolygonObject(double* points, double* normals, int* indice, int size);
	void ExportForceData();
	//GLuint makeCubeObject(int* index, float* vertex);
	void onParticleSystem();
	//void bindingObjects(elements* obj) { objects = obj; }
	int selection(int x, int y);
	//void NextAnimationPart(int times = 1) { objects->sphData->NextAnimationPart(times); }
	//void PriviousAnimationPart(int times = 1) { objects->sphData->PriviousAnimationPart(times); }
	void setViewObject(viewObjectType viewType) { votype = viewType; };
	void getFileData(QFile& pf);
	void getSphFileData(QStringList& fnames);
	void getDemFileData(QStringList& fnames, bool rt);
	void OpenFiles(QStringList& fnames);
	void OpenTXT_file(QString& file);
	void saveCurrentData(QFile& pf);
	bool SaveModel(QFile& pf);
	void UpdateRtDEMData();
	int getWindowHeight() { return wHeight; }
	bool is_set_particle() { return isSetParticle; }
	// 		parview::Object *getParticle_ptr() { return pview_ptr;  }
	// 		parview::Object *getViewParticle()
	// 		{ 
	// 			return pview_ptr;
	// 		}
	void openSph(QString& fl);
	void openMbd(QString& fl);
	void openResults(QStringList& fl);
	void ChangeShapeData(QString& sname);
	void AddParticlesFromPolygonObject(VEC4D* _sphere);
	void AddBondData(QString& fname);
	void ChangeDisplayOption(int oid);
	QMap<QString, vobject*>& Objects() { return v_objs; }
	vparticles* vParticles() { return vp; }

	void glObjectClear();

	public slots:
	void setXRotation(int angle);
	void setYRotation(int angle);
	void setZRotation(int angle);
	void ShowContextMenu(const QPoint& pos);

signals:
	void xRotationChanged(int angle);
	void yRotationChanged(int angle);
	void zRotationChanged(int angle);

protected:
	void initializeGL();
	void paintGL();
	void resizeGL(int width, int height);
	void mousePressEvent(QMouseEvent *event);
	void mouseReleaseEvent(QMouseEvent *event);
	void mouseMoveEvent(QMouseEvent *event);
	void keyPressEvent(QKeyEvent *);
	void wheelEvent(QWheelEvent *);

private:
	void picking(int x, int y);
	float& verticalMovement() { return trans_y; }
	float& horizontalMovement() { return trans_x; }
	void DrawCartesianCoordinates(vector3<double>& pos, vector3<double>& angle);
	GLuint makeCoordinate();
	//GLuint makePolygonObject(float* points, float* normals, int* indice, int size);
	void normalizeAngle(int *angle);
	//void checkForCgError(const char *situation);
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
	float moveScale;
	float ratio;
	float zoom;
	float trans_x;
	float trans_y;

	float IconScale;

	bool onZoom;
	bool onRotation;
	bool keyID[256];
	bool selected[256];
	unsigned choose;
	bool LBOTTON;
	QPoint lastPos;
	int aFrame;
	bool isAnimation;

	float times[1000];

	viewObjectType votype;
	QMap<QString, vobject*> v_objs;
	QMap<QString, vpolygon*> v_pobjs;
	vparticles *vp;
	QStringList outputNameList;

	float maxViewPoint[3];
	float minViewPoint[3];

signals:
	void mySignal();
};


#endif // GLWIDGET_H

