#ifndef OPENGL_H
#define OPENGL_H

#include <gl/glew.h>
#include <gl/freeglut.h>

GLfloat LightAmbient[] =		{ 0.1f, 0.1f, 0.1f, 1.0f };
GLfloat LightDiffuse[] =		{ 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat LightPosition[] =	{ 1.0f, 1.0f, 0.0f, 1.0f };

// void display();
// void reshape(int w, int h);
// void mouse(int button, int state, int x, int y);
// void motion(int x, int y);
// void specialkeys(int skey, int x, int y);
// void keyboard(unsigned char key, int x, int y);
// void wheel(int wheel, int direction, int x, int y);
// void idle(void);

typedef struct 
{
	bool sphere_view;
	bool polygon_view;
}viewType;

template< typename system >
class opengl
{
private:
	void initGL(int *argc, char **argv, char *win_name="No Name", int win_width=640, int win_height=480)
	{
		glutInit(argc,argv);
		glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
		glutInitWindowSize(win_width,win_height);
		glutCreateWindow(win_name);

		glewInit();

		glEnable(GL_DEPTH_TEST);
		glEnable(GL_LIGHTING);
		//glMaterialfv(GL_FRONT, GL_AMBIENT, m)

		glLightfv(GL_LIGHT1, GL_AMBIENT, LightAmbient);		// Setup The Ambient Light
		glLightfv(GL_LIGHT1, GL_DIFFUSE, LightDiffuse);		// Setup The Diffuse Light
		glLightfv(GL_LIGHT1, GL_POSITION,LightPosition);	// Position The Light
		glEnable(GL_LIGHT1);								// Enable Light One
		glClearColor(103.0f / 255.0f, 153.0f / 255.0f, 255.0f / 255.0f, 1.0);
	}

	void reshape(int w, int h)
	{
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		gluPerspective(60.0, (float) w / (float) h, 0.1, 1000.0);

		glMatrixMode(GL_MODELVIEW);
		glViewport(0, 0, w, h);
	}

	void mouse(int button, int state, int x, int y)
	{
		if (state == GLUT_DOWN) {
			mouse_buttons |= 1<<button;
		} else if (state == GLUT_UP) {
			mouse_buttons = 0;
		}

		mouse_old_x = x;
		mouse_old_y = y;
		glutPostRedisplay();
	}

	void motion(int x, int y)
	{
		float dx, dy;
		dx = (float)(x - mouse_old_x);
		dy = (float)(y - mouse_old_y);

		if (mouse_buttons & 1) {
			rotate_x += dy * 0.2f;
			rotate_y += dx * 0.2f;
		} else if (mouse_buttons & 4) {
			translate_z += dy * 0.01f;
		}

		mouse_old_x = x;
		mouse_old_y = y;
		glutPostRedisplay();
	}

	void specialkeys(int skey, int x, int y)
	{
		switch(skey)
		{
		case GLUT_KEY_UP:
			translate_y += 0.2f;
			break;
		case GLUT_KEY_DOWN:
			translate_y -= 0.2f;
			break;
		case GLUT_KEY_LEFT:
			translate_x -= 0.2f;
			break;
		case GLUT_KEY_RIGHT:
			translate_x += 0.2f;
			break;
		}
	}

	void keyboard(unsigned char key, int x, int y)
	{
		switch(key)
		{
		case 'q':
			if(sf)
				delete sf;
			exit(0);
			break;
		case 'p':
			glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
			break;
		case 'o':
			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
			break;
		case 's':
			viewT.sphere_view ? viewT.sphere_view = false : viewT.sphere_view = true;
			break;
		case 't':
			viewT.polygon_view ? viewT.polygon_view = false : viewT.polygon_view = true;
			break;
		case 'z':
			translate_z += 0.5f;
			break;
		case 'x':
			translate_z -= 0.5f;
			break;
		}
	}

	////////////////////////////////////////////////////////////////////////////////
	//! Display callback
	////////////////////////////////////////////////////////////////////////////////
	void display()
	{
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		glMatrixMode(GL_MODELVIEW);
		glPushMatrix();
		glLoadIdentity();
		glTranslatef(translate_x, translate_y, translate_z);
		glRotatef(rotate_x, 1.0, 0.0, 0.0);
		glRotatef(rotate_y, 0.0, 1.0, 0.0);
		glPushMatrix();	

		glEnable(GL_COLOR_MATERIAL);
		//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		sys->view(viewT);

		glPopMatrix();
		glutSwapBuffers();
	}

	void wheel(int wheel, int direction, int x, int y)
	{
		if(direction==1) translate_z+=0.5f;
		else translate_z-=0.5f;
	}

	void idle(void)
	{
		glutPostRedisplay();
	}

public:
	opengl(int *argc, char **argv, system* _sys, char *win_name="No Name", int win_width=640, int win_height=480)
		: mouse_buttons(0)
		, isclicking(false)
		, rotate_x(0)
		, rotate_y(0)
		, translate_x(0)
		, translate_y(0)
		, translate_z(-25)
	{
		sys = _sys;
		viewT.polygon_view = true;
		viewT.sphere_view = false;

		// 		LightAmbient[4] = { 0.1f, 0.1f, 0.1f, 1.0f };
		// 		LightDiffuse[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
		// 		LightPosition[4] = { 1.0f, 1.0f, 0.0f, 1.0f };

		initGL(argc, argv, win_name, win_width, win_height);
		//pdisplay = &opengl<system>::display;
		glutDisplayFunc(display);
		glutReshapeFunc(reshape);
		glutMouseFunc(mouse);
		glutKeyboardFunc(keyboard);
		glutSpecialFunc(specialkeys);
		glutMouseWheelFunc(wheel);
		glutMotionFunc(motion);
		glutIdleFunc(idle);
	}
	~opengl()
	{

	}

	void call_GlutMainLoop()
	{
		glutMainLoop();
	}

	system* sys;
	viewType viewT;

private:
	int mouse_old_x, mouse_old_y;
	int mouse_buttons;
	bool isclicking;
	float rotate_x, rotate_y;
	float translate_x, translate_y, translate_z;

	//void opengl<system>::(*pdisplay)();
};

#endif