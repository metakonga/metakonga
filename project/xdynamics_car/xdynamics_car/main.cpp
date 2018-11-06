#include "xdynamics_car.h"
#include <QtWidgets/QApplication>
#include <crtdbg.h>
#include <windows.h>


int main(int argc, char *argv[])
{
	if (AllocConsole())
	{
		freopen("CONIN$", "rb", stdin);
		freopen("CONOUT$", "wb", stdout);
		freopen("CONOUT$", "wb", stderr);
	}
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
	QApplication a(argc, argv);
	xdynamics_car w(argc, argv);
	w.show();
	return a.exec();
}
