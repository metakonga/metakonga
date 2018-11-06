#include "glwidget.h"
#include <QDebug>
//#include <QKeyEvent>
#include <QMouseEvent>
#include <QLineEdit>
#include <QTimer>

#include <math.h>

#include "cube.h"
#include "vcube.h"
#include "plane.h"
#include "vplane.h"
#include "vpolygon.h"
//#include "vcylinder.h"
#include "particleManager.h"
#include "vparticles.h"
#include "polygonObject.h"
#include "geometryObjects.h"
#include "vcontroller.h"
#include "modelManager.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define METER 1000

GLWidget* ogl;

GLuint makeCubeObject(int* index, float* vertex)
{
	GLuint list = glGenLists(1);
	glNewList(list, GL_COMPILE);
	glShadeModel(GL_SMOOTH);
	//glColor3f(0.0f, 0.0f, 1.0f);
	//glPolygonMode(GL_FRONT_AND_BACK, GL_LINES);
	glBegin(GL_QUADS);
	for (int i(0); i < 6; i++){
		int *id = &index[i * 4];
		glVertex3f(vertex[id[3] * 3 + 0], vertex[id[3] * 3 + 1], vertex[id[3] * 3 + 2]);
		glVertex3f(vertex[id[2] * 3 + 0], vertex[id[2] * 3 + 1], vertex[id[2] * 3 + 2]);
		glVertex3f(vertex[id[1] * 3 + 0], vertex[id[1] * 3 + 1], vertex[id[1] * 3 + 2]);
		glVertex3f(vertex[id[0] * 3 + 0], vertex[id[0] * 3 + 1], vertex[id[0] * 3 + 2]);
	}
	glEnd();
	glEndList();
	return list;
}

// GLfloat LightAmbient[] = { 0.1f, 0.1f, 0.1f, 1.0f };
// GLfloat LightDiffuse[] = { 1.0f, 1.0f, 1.0f, 1.0f };
// GLfloat LightPosition[] = { 10.0f, 10.0f, -10.0f, 1.0f };
// GLfloat LightPosition2[] = { 10.0f, 10.0f, 10.0f, 1.0f };

GLWidget::GLWidget(int argc, char** argv, QWidget *parent)
	: QGLWidget(parent)
	, vp(NULL)
	, ground_marker(NULL)
	, selectedObject(NULL)
	, zRotationFlag(false)
	, isPressRightMouseButton(false)
	//, Doc(_Doc)
{
	//eye[0] = 0; eye[1] = 0; eye[2] = 2;
	
	ogl = this;
	gridSize = 0.1f;
	viewOption = 0;
	xRot = 0;
	yRot = 0;
	zRot = 0;
	unit = 1;
	trans_z = -1.0;
	//zoom = -6.16199875;
	trans_x = 0;
	moveScale = 0.01f;
	trans_y = 0;
	IconScale = 0.1;
	isSetParticle = false;
	// 	sketch = { 0, };
	// 	sketch.space = 0.02;
	//particle_ptr = NULL;
	QTimer *timer = new QTimer(this);
	connect(timer, SIGNAL(timeout()), this, SLOT(updateGL()));
	setContextMenuPolicy(Qt::CustomContextMenu);
	connect(this, SIGNAL(customContextMenuRequested(const QPoint&)), this, SLOT(ShowContextMenu(const QPoint&)));
	timer->start(1);
	for (int i = 0; i < 256; i++)
	{
		keyID[i] = false;
		selected[i] = false;
	}
	LBOTTON = false;
	onZoom = false;
	onRotation = false;
	isAnimation = false;
	aFrame = 0;
	for (int i(0); i < 256; i++){
		polygons[i] = 0;
	}
	numPolygons = 0;
	//selectedIndex = -1;
	votype = ALL_DISPLAY;
	//glewInit();
	//drawingMode = GL_LINE;
	vglew::vglew(argc, argv);
	setFocusPolicy(Qt::StrongFocus);
	memset(minView, 0, sizeof(float) * 3);
	memset(maxView, 0, sizeof(float) * 3);
}

GLWidget::~GLWidget()
{
	makeCurrent();
	glDeleteLists(coordinate, 1);
	glObjectClear();
}

GLWidget* GLWidget::GLObject()
{
	return ogl;
}

void GLWidget::glObjectClear()
{
	//	qDeleteAll(v_pobjs);
	qDeleteAll(v_objs);
	if (ground_marker) delete ground_marker; ground_marker = NULL;
	if (vp) delete vp; vp = NULL;
}

void GLWidget::setXRotation(int angle)
{
	normalizeAngle(&angle);
	if (angle != xRot) {
		xRot = angle;
		emit xRotationChanged(angle);
		updateGL();
	}
}

void GLWidget::setYRotation(int angle)
{
	normalizeAngle(&angle);
	if (angle != yRot) {
		yRot = angle;
		emit yRotationChanged(angle);
		updateGL();
	}
}

void GLWidget::setZRotation(int angle)
{
	normalizeAngle(&angle);
	if (angle != zRot) {
		zRot = angle;
		emit zRotationChanged(angle);
		updateGL();
	}
}

void GLWidget::ShowContextMenu(const QPoint& pos)
{
	QPoint globalPos = this->mapToGlobal(pos);
	QMenu myMenu;
	QList<QMenu*> menus;
	//selectedObject = NULL;
	//vobject* vobj = NULL;
	if (selectedObjects.size())
	{
		QString name;
		foreach(vobject* vobj, selectedObjects)
		{
			//unsigned int id = selectedIndice.at(i);
			//if (id < 1000){
			//vobject* vobj = static_cast<vobject*>(v_wobjs[id]);
			name = vobj->name();
			QMenu *subMenu = new QMenu(name);
			subMenu->addAction("Select");
			subMenu->addAction("Delete");
			subMenu->addAction("Property");
			if (vobj->ViewObjectType() == vobject::V_POLYGON)
			{
				subMenu->addAction("Refinement");
			}
			myMenu.addMenu(subMenu);
			menus.push_back(subMenu);
		}
// 		for (unsigned int i = 0; i < selectedIndice.size(); i++)
// 		{
// 
// 			unsigned int id = selectedIndice.at(i);
// 			//if (id < 1000){
// 			vobject* vobj = static_cast<vobject*>(v_wobjs[id]);
// 			name = vobj->name();
// 			QMenu *subMenu = new QMenu(name);
// 			subMenu->addAction("Select");
// 			subMenu->addAction("Delete");
// 			subMenu->addAction("Property");
// 			if (vobj->ViewObjectType() == vobject::V_POLYGON)
// 			{
// 				subMenu->addAction("Refinement");
// 			}
// 			myMenu.addMenu(subMenu);
// 			menus.push_back(subMenu);
// 		}
		myMenu.addSeparator();
		myMenu.addAction("Wireframe");
		myMenu.addAction("Solid");
		myMenu.addAction("Shade");
	}

	QAction *selectedItem = myMenu.exec(globalPos);

	if (selectedItem){
		glDisable(GL_BLEND);
		glEnable(GL_DEPTH_TEST);
		glDepthFunc(GL_LEQUAL);
		QString txt = selectedItem->text();
		if (txt == "Wireframe"){
			selectedObject->setDrawingMode(GL_LINE);
		}
		else if (txt == "Solid"){
			selectedObject->setDrawingMode(GL_FILL);
		}
		else if (txt == "Shade"){
			glEnable(GL_BLEND);
			glDisable(GL_DEPTH_TEST);
			glDepthFunc(GL_LEQUAL);
			selectedObject->setDrawingMode(GL_FILL);
		}
		else{
			QString pmenuTitle = ((QMenu*)selectedItem->parentWidget())->title();
			if (txt == "Delete"){
				actionDelete(pmenuTitle);
				modelManager::MM()->ActionDelete(pmenuTitle);
			}
			else if (txt == "Select")
			{
				setSelectMarking(pmenuTitle);
			}
			else if (txt == "Property"){
				emit contextSignal(pmenuTitle, CONTEXT_PROPERTY);
			}
			else if (txt == "Refinement")
			{
				setSelectMarking(pmenuTitle);
				emit contextSignal(pmenuTitle, CONTEXT_REFINEMENT);
			}
		}
	}
	qDeleteAll(menus);
}

void GLWidget::fitView()
{
	VEC3D ang(DEG2RAD * xRot / 16, DEG2RAD * yRot / 16, DEG2RAD * zRot / 16);
	VEC3D maxp = local2global_bryant(ang, VEC3D(maxView[0], maxView[1], maxView[2]));
	VEC3D minp = local2global_bryant(ang, VEC3D(minView[0], minView[1], minView[2]));
	VEC3D dp = maxp - minp;

	trans_x = dp.x;
	trans_y = dp.y;
	trans_z = dp.z - 1.0;
}

void GLWidget::renderText(double x, double y, double z, const QString& str, QColor& c)
{
	qglColor(c);
	QGLWidget::renderText(x, y, z, str);
}

void GLWidget::actionDelete(const QString& tg)
{
	vobject* vobj = v_objs.take(tg);
	if (vobj)
		delete vobj;
}

void GLWidget::initializeGL()
{
	glEnable(GL_DEPTH_TEST);
	glShadeModel(GL_SMOOTH);                              // 매끄러운 세이딩 사용
	//	glEnable(GL_CULL_FACE);                               // 후면 제거

	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	//glEnable(GL_RESCALE_NORMAL); 

	GLfloat LightAmbient[] = { 0.3f, 0.3f, 0.3f, 1.0f };
	GLfloat LightDiffuse[] = { 0.5f, 0.5f, 0.5f, 1.0f };
	GLfloat LightSpecular[] = { 1.0f, 1.0f, 1.0f, 1.0f };
	GLfloat Lightemissive[] = { 0.f, 0.f, 0.f, 1.0f };
	GLfloat LightPosition[] = { 100.0f, 100.0f, 100.0f, 1.0f };

	glLightfv(GL_LIGHT0, GL_AMBIENT, LightAmbient);		// Setup The Ambient Light
	glLightfv(GL_LIGHT0, GL_DIFFUSE, LightDiffuse);		// Setup The Diffuse Light
	glLightfv(GL_LIGHT0, GL_SPECULAR, LightSpecular);
	glLightfv(GL_LIGHT0, GL_POSITION, LightPosition);	// Position The Light

	GLfloat material_Ka[] = { 0.5f, 0.0f, 0.0f, 1.0f };
	GLfloat material_Kd[] = { 0.4f, 0.4f, 0.5f, 1.0f };
	GLfloat material_Ks[] = { 0.8f, 0.8f, 0.0f, 1.0f };
	GLfloat material_Ke[] = { 0.1f, 0.0f, 0.0f, 0.0f };
	GLfloat material_Se = 20.0f;

	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, material_Ka);
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, material_Kd);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, material_Ks);
	glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, material_Ke);
	glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, material_Se);

	ref_marker.setName("ref_marker");
	ref_marker.setAttchedMass(false);
	ref_marker.setMarkerScaleFlag(false);
	ref_marker.define(VEC3D(-0.85, -0.85, 0.0));

	ground_marker = new vmarker(QString("ground_marker"), false);
	ground_marker->define(VEC3D(0.0, 0.0, 0.0));
	ground_marker->setAttchedMass(false);
	v_wobjs[ground_marker->ID()] = (void*)ground_marker;

	protype = PERSPECTIVE_PROJECTION;
}

void GLWidget::makeMassCoordinate(QString& _name)
{
	QMap<QString, vobject*>::iterator it = v_objs.find(_name);
	vobject* vobj = it.value();
}

void GLWidget::drawReferenceCoordinate()
{
	VEC3D ang(DEG2RAD * xRot / 16, DEG2RAD * yRot / 16, DEG2RAD * zRot / 16);
	VEC3D xp = VEC3D(-0.85, -0.85, 0.0) + local2global_bryant(ang, VEC3D(0.13, 0.0, 0.0));
	VEC3D yp = VEC3D(-0.85, -0.85, 0.0) + local2global_bryant(ang, VEC3D(0.0, 0.13, 0.0));
	VEC3D zp = VEC3D(-0.85, -0.85, 0.0) + local2global_bryant(ang, VEC3D(0.0, 0.0, 0.13));
	renderText(xp.x, xp.y, xp.z, QString("X"), QColor(255, 0, 0));
	renderText(yp.x, yp.y, yp.z, QString("Y"), QColor(0, 255, 0));
	renderText(zp.x, zp.y, zp.z, QString("Z"), QColor(0, 0, 255));
	ref_marker.setCurrentAngle(VEC3D(xRot, yRot, zRot));
	ref_marker.draw(GL_RENDER);
}

void GLWidget::drawGroundCoordinate(GLenum eMode)
{
	VEC3D ang(DEG2RAD * xRot / 16, DEG2RAD * yRot / 16, DEG2RAD * zRot / 16);
	renderText(0.13, 0.0, 0.0, QString("X"), QColor(255, 0, 0));
	renderText(0.0, 0.13, 0.0, QString("Y"), QColor(0, 255, 0));
	renderText(0.0, 0.0, 0.13, QString("Z"), QColor(0, 0, 255));
	ground_marker->draw(eMode);
}

void GLWidget::setStartingData(QMap<QString, v3epd_type> d)
{
	QMapIterator<QString, v3epd_type> it(d);
	while (it.hasNext())
	{
		it.next();
		QString s = it.key();
		v3epd_type p = it.value();
		vobject* vo = Object(s);
		vobject* vm = Object(s + "_marker");
		VEC3D ang = ep2e(p.ep);
		double xi = (ang.x * 180) / M_PI;
		double th = (ang.y * 180) / M_PI;
		double ap = (ang.z * 180) / M_PI;
		if (vo)
		{
			vo->setInitialPosition(p.v3);
			vo->setInitialAngle(VEC3D(xi, th, ap));
		}
		if (vm)
		{
			vm->setInitialPosition(p.v3);
			vm->setInitialAngle(VEC3D(xi, th, ap));
		}
	
	}
}

vobject* GLWidget::getVObjectFromName(QString name)
{
	return v_objs.find(name).value();
}

vpolygon* GLWidget::getVPolyObjectFromName(QString name)
{
	return NULL;
}

QString GLWidget::selectedObjectName()
{
	return selectedObject ? selectedObject->name() : "";
}

vobject* GLWidget::selectedObjectWithCast()
{
	if (!selectedObject)
		return NULL;
	
	return selectedObject;
}

GLuint GLWidget::makePolygonObject(double* points, double* normals, int* indice, int size)
{
	GLuint list = glGenLists(1);
	glNewList(list, GL_COMPILE);
	glShadeModel(GL_SMOOTH);
	glColor3f(0.0f, 0.0f, 1.0f);
	for (int i(0); i < size; i++){
		glBegin(GL_TRIANGLES);
		{
			glNormal3dv(&normals[i * 3]);
			glVertex3dv(&points[indice[i * 3 + 0] * 3]);
			glVertex3dv(&points[indice[i * 3 + 1] * 3]);
			glVertex3dv(&points[indice[i * 3 + 2] * 3]);
		}
		glEnd();
	}
	glEndList();
	polygons[numPolygons] = list;
	numPolygons++;
	return list;
}


void GLWidget::drawObject(GLenum eMode)
{
	glTranslatef(trans_x, trans_y, trans_z);
	glRotated(xRot / 16.0, 1.0, 0.0, 0.0);
	glRotated(yRot / 16.0, 0.0, 1.0, 0.0);
	glRotated(zRot / 16.0, 0.0, 0.0, 1.0);
	//qDebug() << xRot << " " << yRot << " " << zRot;
	drawGroundCoordinate(eMode);
	QMapIterator<QString, vobject*> obj(v_objs);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	while (obj.hasNext()){
		obj.next();
		obj.value()->draw(eMode);
		//qDebug() << obj.value()->name();
	}
	glDisable(GL_BLEND);
}

vobject* GLWidget::setSelectMarking(QString sn)
{
	//unsigned int id = selectedIndice.at(0);
	vobject* obj = selectedObjects[sn];// static_cast<vobject*>(v_wobjs[id]);
	obj->setSelected(true);
	selectedObject = obj;

#ifdef _DEBUG
	qDebug() << obj->name() << " is selected.";
#endif
	return obj;
}

void GLWidget::processHits(unsigned int uHits, unsigned int *pBuffer)
{
	unsigned int i, j;
	unsigned int uiName, *ptr;
	ptr = pBuffer;
	if (!isPressRightMouseButton)
	{
		selectedObject = NULL;
		foreach(vobject* vobj, selectedObjects)
			vobj->setSelected(false);
	}
		
// 	foreach(int v, selectedIndice)
// 		static_cast<vobject*>(v_wobjs[v])->setSelected(false);
	if (selectedObjects.size())
		selectedObjects.clear();
	for (i = 0; i < uHits; i++){
		uiName = *ptr;
		ptr += 3;
		int idx = *ptr;
		vobject* _vobj = static_cast<vobject*>(v_wobjs[idx]);
		selectedObjects[_vobj->name()] = _vobj;
		//selectedIndice.push_back(*ptr);// selectedIndice[i] = *ptr;

		//static_cast<vobject*>(v_wobjs[id])->setSelected(true);
		ptr++;
	}
	if (selectedObjects.size() == 1)
	{
		setSelectMarking(selectedObjects.firstKey());
	}
}

void GLWidget::setSketchSpace()
{
	QLineEdit* le = dynamic_cast<QLineEdit*>(sender());
	//sketch.space = le->text().toDouble();
}

void GLWidget::sketchingMode()
{
	// 	glClearColor(0.5f, 0.5f, 0.5f, 0.5f);
	// 	gluOrtho2D(-1.02f, 1.02f, -1.02f, 1.02f);
	// 	unsigned int numGrid = static_cast<unsigned int>(1.0f / gridSize) * 2 + 1;
	// 	glMatrixMode(GL_MODELVIEW);
	// 	glLoadIdentity();
	//glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	// 	/*glPushMatrix();*/
	//glDisable(GL_LIGHTING);
	// 	glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
	// 	glBegin(GL_LINES);
	// 	{
	// 		double sx = floor((sketch.sx + (sketch.ex - sketch.sx) * 0.1) * 10) * 0.1;
	// 		double ex = -sx;
	// 		double sy = floor((sketch.sy + (sketch.ey - sketch.sy) * 0.1) * 10) * 0.1;
	// 		double ey = -sy;
	// 		double lx = (floor(ex / sketch.space)) * sketch.space;
	// 		double ly = (floor(ey / sketch.space)) * sketch.space;
	// 		glPushMatrix();
	// 		glVertex3d(sx, sy, 0); glVertex3d(ex, sy, 0);
	// 		glVertex3d(ex, sy, 0); glVertex3d(ex, ey, 0);
	// 		glVertex3d(ex, ey, 0); glVertex3d(sx, ey, 0);
	// 		glVertex3d(sx, ey, 0); glVertex3d(sx, sy, 0);
	// 		glPopMatrix();
	// 		int nx = static_cast<int>((ex - sx) / sketch.space + 1e-9);
	// 		int ny = static_cast<int>((ey - sy) / sketch.space + 1e-9);
	// 		float fTmp1[16] = { 0.f, };
	// 		glGetFloatv(GL_PROJECTION_MATRIX, fTmp1);
	// 		for (int ix = 1; ix < nx; ix++)
	// 		{
	// 			double x = sx + sketch.space * ix;
	// 			glPushMatrix();
	// 			glVertex3d(x, sy, 0);
	// 			glVertex3d(x, ey, 0);
	// 			glPopMatrix();
	//  			
	// 		}
	// 		for (int iy = 1; iy < ny; iy++)
	// 		{
	// 			double y = sy + sketch.space * iy;
	// 			glPushMatrix();
	// 			glVertex3d(sx, y, 0);
	// 			glVertex3d(ex, y, 0);
	// 			glPopMatrix();
	// 		}
	// // 		for (double x = sx; x < ex; x += sketch.space){
	// // 			double rx = floor(x + 10e-9);
	// // 			for (double y = sy; y < ey; y += sketch.space){
	// // 				double ry = floor(y + 10e-9);
	// // 				glPushMatrix();
	// // 				glVertex3d(x, y, 0);
	// // 				glVertex3d(lx, y, 0);
	// // 
	// // // 				glVertex3f(x, y, 0.f);
	// // // 				glVertex3f(x, ly, 0.f);
	// // 				glPopMatrix();
	// // 			}
	// // 			glPushMatrix();
	// // 			glVertex3d(x, sy, 0);
	// // 			glVertex3d(x, ly, 0);
	// // 			glPopMatrix();
	// // 		}
	// // // 		glVertex2f(-0.98f, -0.98f);
	// // // 		glVertex2f(-0.98f, 0.98f);
	// // // 
	// // // 		glVertex2f(-0.98f, 0.98f);
	// // // 		glVertex2f(0.98f, 0.98f);
	// // // 
	// // // 		glVertex2f(0.98f, 0.98f);
	// // // 		glVertex2f(0.98f, -0.98f);
	// // // 
	// // // 		glVertex2f(0.98f, -0.98f);
	// // // 		glVertex2f(-0.98f, -0.98f);
	//  	}
	//  	glEnd();

}

void GLWidget::paintGL()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glMatrixMode(GL_PROJECTION);

	glLoadIdentity();
	glClearColor(1.0, 1.0, 1.0, 1.0);
	gluOrtho2D(-1.0f, 1.0f, -1.0f, 1.0f);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glPushMatrix();
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	drawReferenceCoordinate();
	resizeGL(wWidth, wHeight);

	drawObject(GL_RENDER);

	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	if (vp)
	{
		model::isSinglePrecision ? 
			vp->draw_f(GL_RENDER, wHeight, protype, abs(trans_z)) :
			vp->draw(GL_RENDER, wHeight, protype, abs(trans_z));
		
	}
		

	if (vcontroller::Play()){
		vcontroller::update_frame();
		emit mySignal();
	}

	glEnable(GL_COLOR_MATERIAL);
}

void GLWidget::resizeGL(int width, int height)
{
	wWidth = width; wHeight = height;

	glViewport(0, 0, width, height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	ratio = (GLfloat)(width) / (GLfloat)(height);
	float z = abs(trans_z);
	float c = z * tanf(30.0f * M_PI / 180.0f);
	switch (protype)
	{
	case PERSPECTIVE_PROJECTION:
		gluPerspective(60.0f, ratio, 0.01f, 1000.0f);
		break;
	case ORTHO_PROJECTION:
		if (width <= height){
			glOrtho(-1.0f * c, 1.0f * c, (-1.0f / ratio) * c, (1.0f / ratio) * c, 0.01f, 1000.f);
		}
		else{
			glOrtho(-1.0f * c * ratio, 1.0f * c * ratio, -1.0f * c, 1.0f * c, 0.01f, 1000.f);
		}
		break;
	}

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}

void GLWidget::wheelEvent(QWheelEvent *e)
{
	QPoint  p = e->angleDelta();
	float pzoom = trans_z;
	p.y() > 0 ? trans_z -= 2.0f*moveScale : trans_z += 2.0f*moveScale;

	setFocusPolicy(Qt::StrongFocus);
}

void GLWidget::mousePressEvent(QMouseEvent *event)
{
	lastPos = event->pos();
	if (event->button() == Qt::RightButton){
		picking(lastPos.x(), lastPos.y());
		isPressRightMouseButton = true;
	}
	if (event->button() == Qt::MiddleButton){
		onZoom = true;
	}
	if (event->button() == Qt::LeftButton){
		isPressRightMouseButton = false;
		if (keyID[82])
			onRotation = true;
		else if (!keyID[84])
			picking(lastPos.x(), lastPos.y());
	}
}

void GLWidget::mouseReleaseEvent(QMouseEvent *event)
{
	if (onRotation)
		onRotation = false;

	if (onZoom)
		onZoom = false;

	if (event->button() == Qt::LeftButton){
		if (keyID[90])
			keyID[90] = false;
		if (keyID[84])
			keyID[84] = false;
		if (keyID[82])
			keyID[82] = false;
	}
}

void GLWidget::mouseMoveEvent(QMouseEvent *event)
{
	int dx = event->x() - lastPos.x();
	int dy = event->y() - lastPos.y();
	if (keyID[84]){
		dy > 0 ? trans_y -= 0.1f*moveScale*dy : trans_y -= 0.1f*moveScale*dy;
		dx > 0 ? trans_x += 0.1f*moveScale*dx : trans_x += 0.1f*moveScale*dx;
	}
	if (keyID[82] && onRotation) {
		if (zRotationFlag)
			setZRotation(zRot - 8 * dx);
		else
		{
			setXRotation(xRot + 8 * dy);
			setYRotation(yRot + 8 * dx);
		}
	}
	if (onZoom)
	{
		dy > 0 ? trans_z -= 0.01f*moveScale : trans_z += 0.01f*moveScale;
	}
	lastPos = event->pos();
}

void GLWidget::picking(int x, int y)
{
	
	unsigned int aSelectBuffer[SELECT_BUF_SIZE];
	unsigned int uiHits;
	int aViewport[4];

	glGetIntegerv(GL_VIEWPORT, aViewport);

	glSelectBuffer(SELECT_BUF_SIZE, aSelectBuffer);
	glRenderMode(GL_SELECT);

	glInitNames();
	glPushName(0);

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();

	gluPickMatrix((double)x, (double)(aViewport[3] - y), 5.0, 5.0, aViewport);

	switch (protype)
	{
	case PERSPECTIVE_PROJECTION: gluPerspective(60.0, ratio, 0.01f, 1000.0f); break;
	case ORTHO_PROJECTION:
		if (wWidth <= wHeight)
			glOrtho(-1.f * abs(trans_z), 1.f * abs(trans_z), -1.f / ratio * abs(trans_z), 1.f / ratio * abs(trans_z), 0.01f, 1000.f);
		else
			glOrtho(-1.f * abs(trans_z) * ratio, 1.f * ratio * abs(trans_z), -1.f * abs(trans_z), 1.f * abs(trans_z), 0.01f, 1000.f);
		break;
	}

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	//gluLookAt(eye[0], eye[1], eye[2], 0, 0, 0, 0, 1, 0);
	drawObject(GL_SELECT);
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	uiHits = glRenderMode(GL_RENDER);
	processHits(uiHits, aSelectBuffer);
	glMatrixMode(GL_MODELVIEW);

}

void GLWidget::keyReleaseEvent(QKeyEvent *e)
{
	switch (e->key())
	{
	case Qt::Key_Control:
		zRotationFlag = false;
		break;
	}
}

void GLWidget::keyPressEvent(QKeyEvent *e)
{
	switch (e->key()){
	case Qt::Key_Up:
		verticalMovement() += 0.1f*moveScale;
		break;
	case Qt::Key_Down:
		verticalMovement() -= 0.1f*moveScale;
		break;
	case Qt::Key_Left:
		horizontalMovement() -= 0.1f*moveScale;
		break;
	case Qt::Key_Right:
		horizontalMovement() += 0.1f*moveScale;
		break;
	case Qt::Key_Plus:
		moveScale += 0.001f;
		break;
	case Qt::Key_Minus:
		moveScale -= 0.001f;
		break;
	case Qt::Key_PageUp:
		if (vp)
			vp->upParticleScale(1);
		break;
	case Qt::Key_PageDown:
		if (vp)
			vp->downParticleScale(1);
		break;
	case Qt::Key_Control:
		if (keyID[82])
			zRotationFlag = true;
		break;
	case 90:
		setKeyState(true, 90);
		break;
	case 84:
		setKeyState(true, 84);
		break;
	case 82:
		setKeyState(true, 82);
		break;
	}
	if (moveScale <= 0){
		moveScale = 0.0f;
	}
}

void GLWidget::setMaxViewPosition(float x, float y, float z)
{
	if (x > maxView[0]) maxView[0] = x;
	if (y > maxView[1]) maxView[1] = y;
	if (z > maxView[2]) maxView[2] = z;
}

void GLWidget::setMinViewPosition(float x, float y, float z)
{
	if (x < minView[0]) minView[0] = x;
	if (y < minView[1]) minView[1] = y;
	if (z < minView[2]) minView[2] = z;
}

void GLWidget::normalizeAngle(int *angle)
{
	while (*angle < 0)
		*angle += 360 * 16;
	while (*angle > 360 * 16)
		*angle -= 360 * 16;
}

void GLWidget::openMbd(QString& file)
{
	QFile qf(file);
	qf.open(QIODevice::ReadOnly);

	unsigned int nout = 0;
	unsigned int nm = 0;
	unsigned int id = 0;
	unsigned int cnt = 0;
	unsigned int name_size = 0;
	char ch;

	//str.
	double ct = 0.f;
	VEC3D p, v, a;
	EPD ep, ev, ea;
	QMap<unsigned int, QString>::iterator it;
	qf.read((char*)&nm, sizeof(unsigned int));
	qf.read((char*)&nout, sizeof(unsigned int));
	while (!qf.atEnd()){

		qf.read((char*)&cnt, sizeof(unsigned int));
		for (unsigned int i = 0; i < nm; i++){
			QString str;
			qf.read((char*)&name_size, sizeof(unsigned int));
			for (unsigned int j = 0; j < name_size; j++){
				qf.read(&ch, sizeof(char));
				str.push_back(ch);
			}
			qf.read((char*)&id, sizeof(unsigned int));
			qf.read((char*)&ct, sizeof(double));
			qf.read((char*)&p, sizeof(VEC3D));
			qf.read((char*)&ep, sizeof(EPD));
			qf.read((char*)&v, sizeof(VEC3D));
			qf.read((char*)&ev, sizeof(EPD));
			qf.read((char*)&a, sizeof(VEC3D));
			qf.read((char*)&ea, sizeof(EPD));

			vobject* vobj = v_objs.find(str).value();
			vobj->setResultData(nout);
			vobj->insertResultData(cnt, p, ep);
		}
	}

	vcontroller::setTotalFrame(nout);
}

void GLWidget::ChangeDisplayOption(int oid)
{
	viewOption = oid;
}

vobject* GLWidget::Object(QString nm)
{
	QStringList l = v_objs.keys();
	QStringList::const_iterator it = qFind(l, nm);
	if (it == l.end())
		return NULL;
	return v_objs[nm];
}

void GLWidget::makeCube(cube* c)
{
	if (!c)
		return;
	vcube *vc = new vcube(c->Name());
	//qDebug() << c->Name();
	vc->makeCubeGeometry(c->Name(), c->RollType(), c->MaterialType(), c->min_point().To<float>(), c->cube_size().To<float>());
	v_objs[c->Name()] = vc;
	//qDebug() << vc;
	v_wobjs[vc->ID()] = (void*)vc;
}

void GLWidget::makePlane(plane* p)
{
	if (!p)
		return;
	vplane *vpp = new vplane(p->Name());
	vpp->makePlaneGeometry(p->XW(), p->W2(), p->W3(), p->W4());
	v_objs[p->Name()] = vpp;
	v_wobjs[vpp->ID()] = (void*)vpp;
	//qDebug() << p->Name() << " is made";
}

void GLWidget::makeCylinder(cylinder* cy)
{
	if (!cy)
		return;
}

void GLWidget::makeLine()
{

}

vpolygon* GLWidget::makePolygonObject(QString _nm, import_shape_type t, QString file, double x, double y, double z)
{
	vpolygon* vpoly = new vpolygon(_nm);
	vpoly->define(t, file, x, y, z);
	v_objs[_nm] = vpoly;
	v_wobjs[vpoly->ID()] = (void*)vpoly;
	return vpoly;
}

void GLWidget::makeParticle(double* pos, unsigned int n)
{
	if (!vp)
	{
		vp = new vparticles;
	}
	if (vp->define(pos, n))
		isSetParticle = true;
	else
	{
		vp->resizeMemory(pos, n);
	}
}

void GLWidget::makeParticle_f(float* pos, unsigned int n)
{
	if (!vp)
	{
		vp = new vparticles;
	}
	if (vp->define_f(pos, n))
		isSetParticle = true;
	else
	{
		vp->resizeMemory_f(pos, n);
	}
}

vmarker* GLWidget::makeMarker(QString n, VEC3D p, bool mcf)
{
	QString _name = n + "_marker";
	vmarker* vm = new vmarker(_name, mcf);
	vm->setAttachObject(n);
	vm->define(p);
	v_objs[_name] = vm;
	v_wobjs[vm->ID()] = (void*)vm;

	setMaxViewPosition(p.x, p.y, p.z);
	setMinViewPosition(p.x, p.y, p.z);
	return vm;
}

void GLWidget::openResults(QStringList& fl)
{

}