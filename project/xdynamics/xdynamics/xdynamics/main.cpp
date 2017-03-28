#include "xdynamics.h"
#include <QtWidgets/QApplication>
#include <crtdbg.h>

int main(int argc, char *argv[])
{
// 	double o1 = sin((90 * M_PI / 180) / 2);
// 	double o3 = 0.707106781186547524401 * 0.707106781186547524401;
// 	double o2 = o1 * o1;
// 	long double o = 0.70710678118654749 * 0.70710678118654749;
	if (AllocConsole())
	{
		freopen("CONIN$", "rb", stdin);
		freopen("CONOUT$", "wb", stdout);
		freopen("CONOUT$", "wb", stderr);
	}
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
	QApplication a(argc, argv);
	xdynamics w(argc, argv);
	w.show();
	return a.exec();
}
