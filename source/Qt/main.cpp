#include "parview.h"
#include <crtdbg.h>
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
	//_CrtSetBreakAlloc(51143);
	QApplication a(argc, argv);
	parview::parVIEW w(argc, argv);
	w.show();
	return a.exec();
}
