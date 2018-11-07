#include "xdynamics.h"
#include "numeric_utility.h"
#include <QtWidgets/QApplication>
#include <crtdbg.h>

int main(int argc, char *argv[])
{
// 	double o1 = sin((90 * M_PI / 180) / 2);
// 		double o3 = 0.707106781186547524401 * 0.707106781186547524401;
// 		double o2 = o1 * o1;
// 		long double o = 0.70710678118654749 * 0.70710678118654749;
// 		MATD lhs(7,9);
// 		lhs(0, 0) = 1; lhs(0, 1) = 0; lhs(0, 2) = 0; lhs(0, 3) = 0;     lhs(0, 4) = 0;    lhs(0, 5) = 0;     lhs(0, 6) = 0;    lhs(0, 7) = 0;     lhs(0, 8) = 0;
// 		lhs(1, 0) = 0; lhs(1, 1) = 1; lhs(1, 2) = 0; lhs(1, 3) = 0;     lhs(1, 4) = 0;    lhs(1, 5) = 0;     lhs(1, 6) = 0;    lhs(1, 7) = 0;     lhs(1, 8) = 0;
// 		lhs(2, 0) = 0; lhs(2, 1) = 0; lhs(2, 2) = 1; lhs(2, 3) = 0;     lhs(2, 4) = 0;    lhs(2, 5) = 0;     lhs(2, 6) = 0;    lhs(2, 7) = 0;     lhs(2, 8) = 0;
// 		lhs(3, 0) = 1; lhs(3, 1) = 0; lhs(3, 2) = 0; lhs(3, 3) = -1;    lhs(3, 4) = 0;    lhs(3, 5) = 0.45;  lhs(3, 6) = 0;    lhs(3, 7) = 0;     lhs(3, 8) = 0;
// 		lhs(4, 0) = 0; lhs(4, 1) = 1; lhs(4, 2) = 0; lhs(4, 3) = 0;     lhs(4, 4) = -1;   lhs(4, 5) = -0.21;  lhs(4, 6) = 0;    lhs(4, 7) = 0;     lhs(4, 8) = 0;
// 		lhs(5, 0) = 0; lhs(5, 1) = 0; lhs(5, 2) = 0; lhs(5, 3) = -0.91; lhs(5, 4) = 0.42;  lhs(5, 5) = -0.5;  lhs(5, 6) = 0.91;  lhs(5, 7) = -0.42; lhs(5, 8) = 0;
// 		lhs(6, 0) = 0; lhs(6, 1) = 0; lhs(6, 2) = 0; lhs(6, 3) = 0;     lhs(6, 4) = 0;    lhs(6, 5) = 1;     lhs(6, 6) = 0;    lhs(6, 7) = 0;     lhs(6, 8) = -1;
// 		
// 		VECUI pv;
// 		pv.alloc(9);
// 		pv.initSequence();
	if (AllocConsole())
	{
		freopen("CONIN$", "rb", stdin);
		freopen("CONOUT$", "wb", stdout);
		freopen("CONOUT$", "wb", stderr);
	}
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
// 	numeric::utility::coordinatePartioning(lhs, pv);
// 	QFile qf("C:/metakonga/mat.txt");
// 	qf.open(QIODevice::WriteOnly);
// 	QTextStream qts(&qf);
// 	for (unsigned int i = 0; i < lhs.rows(); i++)
// 	{
// 		for (unsigned int j = 0; j < lhs.cols(); j++)
// 		{
// 			qts << lhs(i, j) << " ";
// 		}
// 		qts << endl;
// 	}
// 	qf.close();
	QApplication a(argc, argv);
	xdynamics w(argc, argv);
	w.show();
	return a.exec();
}

