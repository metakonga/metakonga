#include "xdynamics.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	xdynamics w(argc, argv);
	w.show();
	return a.exec();
}
