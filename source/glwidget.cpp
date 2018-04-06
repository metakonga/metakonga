#include "glwidget.h"
#include <QDebug>
//#include <QKeyEvent>
#include <QMouseEvent>
#include <QLineEdit>
#include <QTimer>


#include <math.h>
#include "mphysics_types.h"
//#include "gl/freeglut.h"

#include "cube.h"
#include "vcube.h"
#include "plane.h"
#include "vplane.h"
#include "vpolygon.h"
#include "cylinder.h"
#include "vcylinder.h"
#include "particle_system.h"
#include "vparticles.h"
#include "polygonObject.h"
#include "modeler.h"
#include "vcontroller.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define METER 1000

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

GLfloat LightAmbient[] = { 0.392157f, 0.007843f, 0.039216f, 1.0f };
GLfloat LightDiffuse[] = { 0.643137f, 0.988235f, 0.611765f, 1.0f };
GLfloat LightSpecular[] = { 0.f, 0.f, 0.f, 1.0f };
GLfloat Lightemissive[] = { 0.f, 0.f, 0.f, 1.0f };
GLfloat LightPosition[] = { 1.0f, 1.0f, -1.0f, 1.0f };
GLfloat LightPosition2[] = { 1.0f, 1.0f, 1.0f, 1.0f };
// GLfloat LightAmbient[] = { 0.1f, 0.1f, 0.1f, 1.0f };
// GLfloat LightDiffuse[] = { 1.0f, 1.0f, 1.0f, 1.0f };
// GLfloat LightPosition[] = { 10.0f, 10.0f, -10.0f, 1.0f };
// GLfloat LightPosition2[] = { 10.0f, 10.0f, 10.0f, 1.0f };

GLWidget::GLWidget(int argc, char** argv, QWidget *parent)
	: QGLWidget(parent)
	, vp(NULL)
	//, Doc(_Doc)
{
	gridSize = 0.1f;
	viewOption = 0;
	xRot =  0;
	yRot =  0;
	zRot = 0;
	unit = 1;
	zoom = -1.0;
	//zoom = -6.16199875;
	trans_x =  0;
	moveScale = 0.01f;
	trans_y =  0;
	IconScale = 0.1;
	isSetParticle = false;
	sketch = { 0, };
	sketch.space = 0.02;
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
	drawingMode = GL_LINE;
	vglew::vglew(argc, argv);
	setFocusPolicy(Qt::StrongFocus);
	//glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	
// 	QPen penBorder;
// 	penBorder.setColor(QColor(0, 0, 0));
// 	penBorder.setWidth(1);
// 	m_rectHovered = new QGraphicsRectItem();
// 	m_rectHovered->setBrush(QBrush(Qt::yellow));
// 	m_coordHoverX = new QGraphicsSimpleTextItem(m_rectHovered);
// 	m_coordHoverY = new QGraphicsSimpleTextItem(m_rectHovered);
// 	penBorder.setColor(QColor(0, 0, 0));
// 	penBorder.setWidth(1);
// 	m_coordHoverX->setPen(penBorder);
// 	m_coordHoverY->setPen(penBorder);
}

GLWidget::~GLWidget()
{
	makeCurrent();
	glDeleteLists(coordinate, 1);
	glObjectClear();
}

void GLWidget::glObjectClear()
{
	qDeleteAll(v_pobjs);
	qDeleteAll(v_objs);
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
	if (selectedIndice.size())
	{
		QString name;
		for (unsigned int i = 0; i < selectedIndice.size(); i++)
		{
			
			unsigned int id = selectedIndice.at(i);
			if (id < 1000){
				vobject* vobj = (vobject*)(v_wobjs[id]);
				name = vobj->name();
			}
			else{
				vpolygon* vpobj = (vpolygon*)(v_wobjs[id]);
				name = vpobj->name();
			}
			QMenu *subMenu = new QMenu(name);
			subMenu->addAction("Delete");
			subMenu->addAction("Property");
			myMenu.addMenu(subMenu);
			menus.push_back(subMenu);
		}
	}
	myMenu.addSeparator();
	myMenu.addAction("Wireframe");
	myMenu.addAction("Solid");
	myMenu.addAction("Shade");
	
	QAction *selectedItem = myMenu.exec(globalPos);
	
	if (selectedItem){
		QString txt = selectedItem->text();
		if (txt == "Wireframe"){
			drawingMode = GL_LINE;
			glDisable(GL_BLEND);
		}
		else if (txt == "Solid"){
			drawingMode = GL_FILL;
			glDisable(GL_BLEND);
		}
		else if (txt == "Shade"){
			drawingMode = GL_FILL;
			glEnable(GL_BLEND);
		}
		else{	
			QString pmenuTitle = ((QMenu*)selectedItem->parentWidget())->title();
			if (txt == "Delete"){
				md->actionDelete(pmenuTitle);
				this->actionDelete(pmenuTitle);
			}
			else if (txt == "Property"){

			}
		}		
	}
	qDeleteAll(menus);
}

void GLWidget::actionDelete(const QString& tg)
{
	vobject* vobj = v_objs.take(tg);
	if (vobj)
		delete vobj;
	vpolygon* pobj = v_pobjs.take(tg);
	if (pobj)
		delete pobj;	
}

void GLWidget::initializeGL()
{
	glEnable(GL_LIGHTING);
	//glEnable(GL_LIGHT0);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	//glDisable(GL_DEPTH_TEST);
	//glHint(GL_POLYGON_SMOOTH, GL_NICEST)
	glLightfv(GL_LIGHT1, GL_AMBIENT, LightAmbient);		// Setup The Ambient Light
	glLightfv(GL_LIGHT1, GL_DIFFUSE, LightDiffuse);		// Setup The Diffuse Light
	glLightfv(GL_LIGHT1, GL_SPECULAR, LightSpecular);
	glLightfv(GL_LIGHT1, GL_EMISSION, Lightemissive);
	glLightfv(GL_LIGHT1, GL_POSITION, LightPosition);	// Position The Light
	glEnable(GL_LIGHT1);								// Enable Light One

	glLightfv(GL_LIGHT2, GL_AMBIENT, LightAmbient);		// Setup The Ambient Light
	glLightfv(GL_LIGHT2, GL_DIFFUSE, LightDiffuse);		// Setup The Diffuse Light
	glLightfv(GL_LIGHT2, GL_SPECULAR, LightSpecular);
	glLightfv(GL_LIGHT2, GL_EMISSION, Lightemissive);
	glLightfv(GL_LIGHT2, GL_POSITION, LightPosition2);	// Position The Light
	glEnable(GL_LIGHT2);								// Enable Light One

	coordinate = makeCoordinate();
	//protype = PERSPECTIVE_PROJECTION;
	protype = ORTHO_PROJECTION;

	glEnable(GL_NORMALIZE);

	glClearColor(103.0f / 255.0f, 153.0f / 255.0f, 255.0f / 255.0f, 1.0);
}

void GLWidget::makeMassCoordinate(QString& _name)
{
	QMap<QString, vobject*>::iterator it = v_objs.find(_name);
	vobject* vobj = it.value();
}

vobject* GLWidget::getVObjectFromName(QString name)
{
	return v_objs.find(name).value();
}

vpolygon* GLWidget::getVPolyObjectFromName(QString name)
{
	return v_pobjs.find(name).value();
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

GLuint GLWidget::makeCoordinate()
{
	GLuint list = glGenLists(1);
	glNewList(list, GL_COMPILE);
	glShadeModel(GL_FLAT);

	qglColor(QColor(255, 0, 0));
	glBegin(GL_LINES);
	{
		glVertex3f(0.0f, 0.0f, 0.0f);
		glVertex3f(IconScale*1.0f, 0.0f, 0.0f);
	}
	glEnd();
	glBegin(GL_TRIANGLE_FAN);
	{
		glVertex3f(IconScale*1.5f, 0.0f, 0.0f);
		float angInc = (float)(45 * (M_PI / 180));
		for (int i = 0; i < 9; i++)
		{
			float rad = angInc * i;
			glVertex3f(IconScale*1.0f, cos(rad)*IconScale*0.15f, sin(rad)*IconScale*0.15f);
		}
	}
	glEnd();
	glBegin(GL_LINES);
	{
		glVertex3f(IconScale*1.5f, IconScale*(-0.1f), 0.0f);
		glVertex3f(IconScale*1.3f, IconScale*(-0.3f), 0.0f);
		glVertex3f(IconScale*1.3f, IconScale*(-0.1f), 0.0f);
		glVertex3f(IconScale*1.5f, IconScale*(-0.3f), 0.0f);
	}
	glEnd();

	qglColor(QColor(0, 255, 0));
	glBegin(GL_LINES);
	{
		glVertex3f(0.0f, 0.0f, 0.0f);
		glVertex3f(0.0f, IconScale*1.0f, 0.0f);
	}
	glEnd();
	glBegin(GL_TRIANGLE_FAN);
	{
		glVertex3f(0.0f, IconScale*1.5f, 0.0f);
		float angInc = (float)(45 * (M_PI / 180));
		for (int i = 0; i < 9; i++)
		{
			float rad = angInc * i;
			glVertex3f(cos(rad)*IconScale*0.15f, IconScale*1.0f, sin(rad)*IconScale*0.15f);
		}
	}
	glEnd();

	glBegin(GL_LINES);
	{
		glVertex3f(IconScale*(-0.1f), IconScale*1.5f, 0.0f);
		glVertex3f(IconScale*(-0.2f), IconScale*1.4f, 0.0f);
		glVertex3f(IconScale*(-0.3f), IconScale*1.5f, 0.0f);
		glVertex3f(IconScale*(-0.2f), IconScale*1.4f, 0.0f);
		glVertex3f(IconScale*(-0.2f), IconScale*1.4f, 0.0f);
		glVertex3f(IconScale*(-0.2f), IconScale*1.25f, 0.0f);
	}
	glEnd();

	qglColor(QColor(0, 0, 255));
	glBegin(GL_LINES);
	{
		glVertex3f(0.0f, 0.0f, 0.0f);
		glVertex3f(0.0f, 0.0f, IconScale*1.0f);
	}
	glEnd();
	glBegin(GL_TRIANGLE_FAN);
	{
		glVertex3f(0.0f, 0.0f, IconScale*1.5f);
		float angInc = (float)(45 * (M_PI / 180));
		for (int i = 0; i < 9; i++)
		{
			float rad = angInc * i;
			glVertex3f(cos(rad)*IconScale*0.15f, sin(rad)*IconScale*0.15f, IconScale*1.0f);
		}
	}
	glEnd();

	glBegin(GL_LINES);
	{
		glVertex3f(IconScale*(-0.3f), 0.0f, IconScale*1.5f);
		glVertex3f(IconScale*(-0.1f), 0.0f, IconScale*1.5f);
		glVertex3f(IconScale*(-0.1f), 0.0f, IconScale*1.5f);
		glVertex3f(IconScale*(-0.3f), IconScale*(-0.2f), IconScale*1.5f);
		glVertex3f(IconScale*(-0.3f), IconScale*(-0.2f), IconScale*1.5f);
		glVertex3f(IconScale*(-0.1f), IconScale*(-0.2f), IconScale*1.5f);
	}
	glEnd();

	glEndList();
	return list;
}

void GLWidget::DrawCartesianCoordinates(vector3<double>& pos, vector3<double>& angle)
{
	glDisable(GL_LIGHTING);
	glTranslated(unit*pos.x, unit*pos.y, unit*pos.z);
	glRotated(angle.x / 16, 1.0, 0.0, 0.0);
	glRotated(angle.y / 16, 0.0, 1.0, 0.0);
	glRotated(angle.z / 16, 0.0, 0.0, 1.0);
	glCallList(coordinate);
	glEnable(GL_LIGHTING);
}

void GLWidget::drawObject(GLenum eMode)
{
	glTranslatef(0.0f, 0.0f, zoom);
	glTranslatef(trans_x, trans_y, 0.0f);
	glRotated(xRot / 16.0, 1.0, 0.0, 0.0);
	glRotated(yRot / 16.0, 0.0, 1.0, 0.0);
	glRotated(zRot / 16.0, 0.0, 0.0, 1.0);
	if (sketch.isSketching)
	{
		sketchingMode();
	}
	glPolygonMode(GL_FRONT_AND_BACK, drawingMode);
	QMapIterator<QString, vobject*> obj(v_objs);
	
	

	while (obj.hasNext()){
		obj.next();
		obj.value()->color().setAlpha(0.2);
		qglColor(obj.value()->color());
		obj.value()->draw(eMode);
	}

	QMapIterator<QString, vpolygon*> pobj(v_pobjs);
	while (pobj.hasNext()){
		pobj.next();
		//qglColor(QColor("red"));
		pobj.value()->draw(eMode);
	}
	
}

void GLWidget::processHits(unsigned int uHits, unsigned int *pBuffer)
{
	unsigned int i, j;
	unsigned int uiName, *ptr;
	ptr = pBuffer;
	if (selectedIndice.size())
		selectedIndice.clear();
	for (i = 0; i < uHits; i++){
		uiName = *ptr;
		ptr += 3;
		selectedIndice.push_back(*ptr);// selectedIndice[i] = *ptr;
		ptr++;
	}
}

void GLWidget::setSketchSpace()
{
	QLineEdit* le = dynamic_cast<QLineEdit*>(sender());
	sketch.space = le->text().toDouble();
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
	glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
	glBegin(GL_LINES);
	{
		double sx = floor((sketch.sx + (sketch.ex - sketch.sx) * 0.1) * 10) * 0.1;
		double ex = -sx;
		double sy = floor((sketch.sy + (sketch.ey - sketch.sy) * 0.1) * 10) * 0.1;
		double ey = -sy;
		double lx = (floor(ex / sketch.space)) * sketch.space;
		double ly = (floor(ey / sketch.space)) * sketch.space;
		glPushMatrix();
		glVertex3d(sx, sy, 0); glVertex3d(ex, sy, 0);
		glVertex3d(ex, sy, 0); glVertex3d(ex, ey, 0);
		glVertex3d(ex, ey, 0); glVertex3d(sx, ey, 0);
		glVertex3d(sx, ey, 0); glVertex3d(sx, sy, 0);
		glPopMatrix();
		int nx = static_cast<int>((ex - sx) / sketch.space + 1e-9);
		int ny = static_cast<int>((ey - sy) / sketch.space + 1e-9);
		float fTmp1[16] = { 0.f, };
		glGetFloatv(GL_PROJECTION_MATRIX, fTmp1);
		for (int ix = 1; ix < nx; ix++)
		{
			double x = sx + sketch.space * ix;
			glPushMatrix();
			glVertex3d(x, sy, 0);
			glVertex3d(x, ey, 0);
			glPopMatrix();
 			
		}
		for (int iy = 1; iy < ny; iy++)
		{
			double y = sy + sketch.space * iy;
			glPushMatrix();
			glVertex3d(sx, y, 0);
			glVertex3d(ex, y, 0);
			glPopMatrix();
		}
// 		for (double x = sx; x < ex; x += sketch.space){
// 			double rx = floor(x + 10e-9);
// 			for (double y = sy; y < ey; y += sketch.space){
// 				double ry = floor(y + 10e-9);
// 				glPushMatrix();
// 				glVertex3d(x, y, 0);
// 				glVertex3d(lx, y, 0);
// 
// // 				glVertex3f(x, y, 0.f);
// // 				glVertex3f(x, ly, 0.f);
// 				glPopMatrix();
// 			}
// 			glPushMatrix();
// 			glVertex3d(x, sy, 0);
// 			glVertex3d(x, ly, 0);
// 			glPopMatrix();
// 		}
// // 		glVertex2f(-0.98f, -0.98f);
// // 		glVertex2f(-0.98f, 0.98f);
// // 
// // 		glVertex2f(-0.98f, 0.98f);
// // 		glVertex2f(0.98f, 0.98f);
// // 
// // 		glVertex2f(0.98f, 0.98f);
// // 		glVertex2f(0.98f, -0.98f);
// // 
// // 		glVertex2f(0.98f, -0.98f);
// // 		glVertex2f(-0.98f, -0.98f);
 	}
 	glEnd();

}

void GLWidget::paintGL()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
// 	if (isSketching){
// 		sketchingMode();
// 		return;
// 	}
// 	else{
	glClearColor(103.0f / 255.0f, 153.0f / 255.0f, 255.0f / 255.0f, 1.0);
	//}
	gluOrtho2D(-1.0f, 1.0f, -1.0f, 1.0f);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glPushMatrix();
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	DrawCartesianCoordinates(vector3<double>(-0.9, -0.9f, 0.0f), vector3<double>(xRot, yRot, zRot));
	glPopMatrix();

	resizeGL(wWidth, wHeight/*, -trans_x, -trans_y*/);
	glDisable(GL_DEPTH_TEST);
	
	drawObject(GL_RENDER);

	if (vp)
		vp->draw(GL_RENDER, wHeight, protype);

	if (vcontroller::Play()){
		vcontroller::move2forward1x();
		emit mySignal();
	}

	glEnable(GL_COLOR_MATERIAL);
}

void GLWidget::resizeGL(int width, int height)
{
	wWidth = width; wHeight = height;
// 	if (isSketching)
// 		return;
	glViewport(0, 0, width, height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
// 	if (!isSketching)
// 	{
	ratio = (GLfloat)(width) / (GLfloat)(height);
	float z = abs(zoom);
	float ocef = tanf(60 * M_PI / 360.0f);
	switch (protype)
	{
	case PERSPECTIVE_PROJECTION:
		gluPerspective(60.0f, ratio, 0.01f, 1000.0f);
		break;
	case ORTHO_PROJECTION:
		if (width <= height){
			glOrtho(-1 * z, 1 * z, -1 / ratio * z, 1 / ratio * z, 0.01, 1000.);
			sketch.sx = -1 * z; sketch.ex = 1.f * z;
			sketch.sy = -1 / ratio * z; sketch.ey = 1.f / ratio * z;
		}
		else{
			glOrtho(-1.f * z * ratio, 1.f * ratio * z, -1.f * z, 1.f * z, 0.01f, 1000.f);
			sketch.sx = -1.f * z * ratio; sketch.ex = 1.f * ratio * z;
			sketch.sy = -1.f * z; sketch.ey = 1.f * z;
		}
		break;
	}

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}

void GLWidget::wheelEvent(QWheelEvent *e)
{
	QPoint  p = e->angleDelta();
	float pzoom = zoom;
// 	if (isSketching){
// 		gridSize *= p.y() > 0 ? 0.1f : 10.0f;
// 		if (gridSize > 0.99f)
// 			gridSize = 0.1f;
// 		else if (gridSize < 0.00099)
// 			gridSize = 0.001f;
// 	}
// 	else
		p.y() > 0 ? zoom -= 2.0f*moveScale : zoom += 2.0f*moveScale;
	
	setFocusPolicy(Qt::StrongFocus);
}

void GLWidget::mousePressEvent(QMouseEvent *event)
{
	// 	if(event->type() == QEvent::None) 
	// 		return;
	lastPos = event->pos();
	if (event->button() == Qt::RightButton){
		picking(lastPos.x(), lastPos.y());
	}
	if (event->button() == Qt::MiddleButton){
		onZoom = true;
	}
	//if (!keyID[90] && !keyID[84] && !keyID[82]){
	if (event->button() == Qt::LeftButton){
		//int choose = selection(lastPos.x(), lastPos.y());
		if (keyID[82])
			onRotation = true;
		else
			picking(lastPos.x(), lastPos.y());
	//}
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
// 	m_rectHovered->setRect(event->x(), event->y() - 31, 40, 30);
// 	qreal rectX = m_rectHovered->rect().x();
// 	qreal rectY = m_rectHovered->rect().y();
// 	qreal rectW = m_rectHovered->rect().width();
// 	qreal rectH = m_rectHovered->rect().height();
// 	m_coordHoverX->setPos(rectX + rectW / 4 - 3, rectY + 1);
// 	m_coordHoverY->setPos(rectX + rectW / 4 - 3, rectY + rectH / 2 + 1);
// 	m_coordHoverX->setText(QString("%1").arg(event->x(), 4, 'f', 2, '0'));
// 	m_coordHoverY->setText(QString("%1").arg(event->y(), 4, 'f', 2, '0'));
// 	//QGLWidget::mapToParent()
// 	if (sketch.isSketching)
// 	{
// 		m_coordHoverX->setVisible(true);
// 		m_coordHoverY->setVisible(true);
// 		m_rectHovered->setVisible(true);
// 	}
// 	else
// 	{
// 		m_coordHoverX->setVisible(false);
// 		m_coordHoverY->setVisible(false);
// 		m_rectHovered->setVisible(false);
// 	}
	//glMap
	if (keyID[84]){
		dy > 0 ? trans_y -= 0.1f*moveScale*dy : trans_y -= 0.1f*moveScale*dy;
		dx > 0 ? trans_x += 0.1f*moveScale*dx : trans_x += 0.1f*moveScale*dx;
	}
	if (keyID[82] && onRotation) {
		setXRotation(xRot + 8 * dy);
		setYRotation(yRot + 8 * dx);
	}
	if (onZoom)
	{
		dy > 0 ? zoom -= 0.01f*moveScale : zoom += 0.01f*moveScale;
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
			glOrtho(-1.f * abs(zoom), 1.f * abs(zoom), -1.f / ratio * abs(zoom), 1.f / ratio * abs(zoom), 0.01f, 1000.f);
		else
			glOrtho(-1.f * abs(zoom) * ratio, 1.f * ratio * abs(zoom), -1.f * abs(zoom), 1.f * abs(zoom), 0.01f, 1000.f);
		
			
		break;
	}
	
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	drawObject(GL_SELECT);
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	uiHits = glRenderMode(GL_RENDER);
	processHits(uiHits, aSelectBuffer);
	glMatrixMode(GL_MODELVIEW);
	
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

void GLWidget::normalizeAngle(int *angle)
{
	while (*angle < 0)
		*angle += 360 * 16;
	while (*angle > 360 * 16)
		*angle -= 360 * 16;
}

bool GLWidget::change(QString& fp, tChangeType ct, tFileType ft)
{
	QFile pf(fp);
	pf.open(QIODevice::ReadOnly);
	switch (ct)
	{
	case CHANGE_PARTICLE_POSITION:{
			VEC4D_PTR _pos = new VEC4D[vp->Np()];
			VEC3D_PTR _vel = new VEC3D[vp->Np()];
			float time = 0.f;
			unsigned int _np = 0;
			pf.read((char*)&_np, sizeof(unsigned int));
			pf.read((char*)&time, sizeof(double));
			pf.read((char*)_pos, sizeof(VEC4D) * vp->Np());
			pf.read((char*)_vel, sizeof(VEC3D) * vp->Np());
			vp->changeParticles(_pos);
			vp->getParticleSystem()->setVelocity(&(_vel[0].x));
			delete[] _pos;
			delete[] _vel;
		}
		break;
	}
	pf.close();

	return true;
}

void GLWidget::openSph(QString& file)
{
	QFile qf(file);
	qf.open(QIODevice::ReadOnly);
	QTextStream ts(&qf);
	QString path;
	ts >> path >> path;
	QString ch;
	ts >> ch >> ch;
	unsigned int nfluid, nbound, ndummy;
	ts >> ch >> nfluid;
	ts >> ch >> nbound;
	ts >> ch >> ndummy;
	vp = new vparticles;
	vp->settingSphParticles(nfluid+nbound+ndummy, path + "/part0000.bin");
	isSetParticle = true;
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
 			//it = names.find(id);
			if (v_pobjs.find(str) != v_pobjs.end()){
				vpolygon* vpoly = v_pobjs.find(str).value();
				vpoly->setResultData(nout);
				vpoly->insertResultData(cnt, p, ep);
			}
			else{
				vobject* vobj = v_objs.find(str).value();
				vobj->setResultData(nout);
				vobj->insertResultData(cnt, p, ep);
			}
			
		}
	}

	vcontroller::setTotalFrame(nout);
}

void GLWidget::ChangeDisplayOption(int oid)
{
	viewOption = oid;
}

void GLWidget::makeCube(cube* c)
{
	if (!c)
		return;
	vcube *vc = new vcube(c->objectName());
	vc->makeCubeGeometry(c->objectName(), c->rolltype(), c->materialType(), c->min_point().To<float>(), c->cube_size().To<float>());
	v_objs[c->objectName()] = vc;
	v_wobjs[vc->ID()] = (void*)vc;
}

void GLWidget::makePlane(plane* p)
{
	if (!p)
		return;
	vplane *vpp = new vplane(p->objectName());
	vpp->makePlaneGeometry((float)p->L1(), p->XW().To<float>(), p->PA().To<float>(), p->PB().To<float>(), p->U1().To<float>());
	v_objs[p->objectName()] = vpp;
	v_wobjs[vpp->ID()] = (void*)vpp;
}

void GLWidget::makeCylinder(cylinder* cy)
{
	if (!cy)
		return;
	vcylinder *vcy = new vcylinder(cy->objectName());
	vcy->makeCylinderGeometry((float)cy->baseRadisu(), (float)cy->topRadius(), (float)cy->length(), cy->origin().To<float>(), cy->basePos().To<float>(), cy->topPos().To<float>());
	vcy->copyCoordinate(coordinate);
	//cy->setOrientation(M_PI * vcy->eulerAngle_1()/180, M_PI * vcy->eulerAngle_2() / 180, M_PI * vcy->eulerAngle_3()/180);
	v_objs[cy->objectName()] = vcy;
	v_wobjs[vcy->ID()] = (void*)vcy;
}

void GLWidget::makeLine()
{

}

void GLWidget::makePolygonObject(polygonObject* po)
{
	vpolygon* vpoly = new vpolygon(po->objectName());
	vpoly->define(po->getOrigin(), po->hostPolygonInfo(), po->hostSphereSet(), po->vertexSet(), po->indexSet(), po->numIndex(), po->numVertex());
	v_pobjs[po->objectName()] = vpoly;
	v_wobjs[vpoly->ID()] = (void*)vpoly;
 }

void GLWidget::makeParticle(particle_system* ps)
{
	if (!ps){
		return;
	}
	if (!vp){
		vp = new vparticles(ps);
		if (vp->define())
			isSetParticle = true;
		v_objs[ps->baseObject()]->setDisplay(true);
	}
	else{
		vp->resizeMemory();
		vp->define();
	}
}

void GLWidget::openResults(QStringList& fl)
{
	vp->setResultFileList(fl);
	vcontroller::setTotalFrame(fl.size());
}