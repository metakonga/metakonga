// #include "ExportDialog.h"
// #include "modeler.h"
// #include <QtWidgets>
// 
// ExportDialog::ExportDialog(modeler *_md)
// 	: dlg(NULL)
// 	, isDialogOk(false)
// 	, md(_md)
// {
// 
// }
// 
// ExportDialog::~ExportDialog()
// {
// 
// }
// 
// bool ExportDialog::callDialog()
// {
// 	if (!dlg)
// 	{
// 		dlg = new QDialog;
// 		QLabel *LFrom = new QLabel("From");
// 		QLabel *LTo = new QLabel("To");
// 		QLabel *LTarget = new QLabel("Target");
// 		LEFrom = new QLineEdit;
// 		LETo = new QLineEdit;
// 		LETarget = new QLineEdit;
// 		QPushButton *PBFrom = new QPushButton("F");
// 		QPushButton *PBTo = new QPushButton("T");
// 		QHBoxLayout *Hlayout = new QHBoxLayout;
// 		QDialogButtonBox *buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
// 		connect(buttonBox, SIGNAL(accepted()), this, SLOT(Click_ok()));
// 		connect(buttonBox, SIGNAL(rejected()), this, SLOT(Click_cancel()));
// 		Hlayout->addWidget(LFrom);
// 		Hlayout->addWidget(LEFrom);
// 		Hlayout->addWidget(PBFrom);
// 		Hlayout->addWidget(LTo);
// 		Hlayout->addWidget(LETo);
// 		Hlayout->addWidget(PBTo);
// 		Hlayout->addWidget(LTarget);
// 		Hlayout->addWidget(LETarget);
// 		QVBoxLayout *Vlayout = new QVBoxLayout;
// 		Vlayout->addLayout(Hlayout);
// 		Vlayout->addWidget(buttonBox);
// 		dlg->setLayout(Vlayout);
// 		connect(PBFrom, SIGNAL(clicked()), this, SLOT(FromAction()));
// 		connect(PBTo, SIGNAL(clicked()), this, SLOT(ToAction()));
// 	}
// 	dlg->exec();
// 	return isDialogOk ? true : false;
// }
// 
// void ExportDialog::FromAction()
// {
// 	QString dir = md->modelPath();
// 	QString fileName = QFileDialog::getOpenFileName(this, tr("open"), dir);
// 	if (fileName.isEmpty())
// 		return;
// // 	int begin = fileName.at(0).lastIndexOf("/");
// // 	QString fileN = fileName.at(0).right(begin + 1);
// 	LEFrom->setText(fileName);
// }
// 
// void ExportDialog::ToAction()
// {
// 	QString dir = md->modelPath();
// 	QString fileName = QFileDialog::getSaveFileName(this, tr("save"), dir);
// 	if (fileName.isEmpty())
// 		return;
// // 	int begin = fileName.lastIndexOf("/");
// // 	QString fileN = fileName.right(begin + 1);
// 	LETo->setText(fileName);
// }
// 
// void ExportDialog::Click_ok()
// {
// 	unsigned int t = LETarget->text().toUInt();
// 	QString Fext = LEFrom->text();
// 	QString Text = LETo->text();
// // 	if (Fext == ".bin" && Text == ".txt")
// // 	{
// 		unsigned int n = 0;
// 		float time = 0.f;
// 		QFile txt(Text);
// 		txt.open(QIODevice::WriteOnly | QIODevice::Text);
// 		QTextStream out(&txt);
// 		out << "Time"
// 			<< " " << "Position(x)" << " " << "Position(y)" << " " << "Position(z)"
// 			<< " " << "Velocity(x)" << " " << "Velocity(y)" << " " << "Velocity(z)"
// 			<< " " << "Force(x)" << " " << "Force(y)" << " " << "Force(z)"
// 			<< " " << "Moment(x)" << " " << "Moment(y)" << " " << "Moment(z)" << endl;
// 
// 		QFile io(Fext);
// 		io.open(QIODevice::ReadOnly);
// 		io.read((char*)&n, sizeof(unsigned int));
// 		io.read((char*)&time, sizeof(float));
// 		double* p = new double[n * 4];
// 		double* v = new double[n * 3];
// 		//float* f = new float[n * 3];
// 		//float* m = new float[n * 3];
// 		io.read((char*)p, sizeof(double)*n * 4);
// 		io.read((char*)v, sizeof(double)*n * 3);
// 		//io.read((char*)f, sizeof(float)*n * 3);
// 		//io.read((char*)m, sizeof(float)*n * 3);
// 		for (unsigned int i = 0; i < n; i++){
// 			out << time
// 				<< " " << p[i * 4 + 0] << " " << p[i * 4 + 1] << " " << p[i * 4 + 2]
// 				<< " " << v[i * 3 + 0] << " " << v[i * 3 + 1] << " " << v[i * 3 + 2] << endl;
// 		}
// 		
// 		//<< " " << f[t * 4 + 0] << " " << f[t * 4 + 1] << " " << f[t * 4 + 2]
// 		//<< " " << m[t * 4 + 0] << " " << m[t * 4 + 1] << " " << m[t * 4 + 2] << endl;
// 		io.close();
// 		delete[] p;
// 		delete[] v;
// 		//delete[] f;
// 		//delete[] m;
// 	
// 		txt.close();
// 	//}
// 	dlg->close();
// 
// 	delete dlg;
// 	dlg = NULL;
// 	isDialogOk = true;
// }
// 
// void ExportDialog::Click_cancel()
// {
// 	dlg->close();
// 
// 	delete dlg;
// 	dlg = NULL;
// 	isDialogOk = true;
// }
// 
