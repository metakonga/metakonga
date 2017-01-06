#include "newDialog.h"
#include "ui_newModel.h"
#include <QtWidgets>

newDialog::newDialog(QWidget* parent)
	: QDialog(parent)
	, ui(new Ui::newModelDialog)
	, isDialogOk(false)
{
	ui->setupUi(this);
	connect(ui->PBOK, SIGNAL(clicked()), this, SLOT(Click_ok()));
	connect(ui->PBBrowse, SIGNAL(clicked()), this, SLOT(Click_browse()));
	_path = "C:/C++/kangsia/case/";
	_name = "Model1";
	ui->CBGravity->setCurrentIndex(4);
}

newDialog::~newDialog()
{
	disconnect(ui->PBOK);
	disconnect(ui->PBBrowse);
	if (ui) delete ui; ui = NULL;
}

bool newDialog::callDialog()
{
// 	QLabel *LName = new QLabel("Name");
// 	QLabel *LPath = new QLabel("Path");
// 	QPushButton *PBBrowse = new QPushButton("Browse");
// 	LEName = new QLineEdit;
// 	LEPath = new QLineEdit;
// 	LEName->setText("isph_test");
// 	LEPath->setText("C:/C++/kangsia/case/parSPH_V2");
// 	QGridLayout *newLayout = new QGridLayout;
// 	QPushButton *PBOk = new QPushButton("OK");
// 	QPushButton *PBCancel = new QPushButton("Cancel");
//	connect(PBOk, SIGNAL(clicked()), this, SLOT(Click_ok()));
//	connect(PBCancel, SIGNAL(clicked()), this, SLOT(Click_cancel()));
// 	newLayout->addWidget(LName, 0, 0); newLayout->addWidget(LEName, 0, 1, 1, 2);
// 	newLayout->addWidget(LPath, 1, 0); newLayout->addWidget(LEPath, 1, 1, 1, 2); newLayout->addWidget(PBBrowse, 1, 3, 1, 1);
// 	newLayout->addWidget(PBOk, 2, 2); newLayout->addWidget(PBCancel, 2, 3);
//	this->setLayout(newLayout);
	ui->LEName->setText(_name);
	this->exec();

	return isDialogOk;
}

void newDialog::Click_ok()
{
	_name = ui->LEName->text();
	//_path = ui->LEPath->text();
	_unit = (tUnit)ui->CBUnit->currentIndex();
	_dir_g = (tGravity)ui->CBGravity->currentIndex();
	//_path += "/";
	this->close();
	isDialogOk = true;
}

void newDialog::Click_browse()
{
	QString fileName = QFileDialog::getOpenFileName(this, tr("open"), _path, tr("Model File(*.mde)"));
	if (!fileName.isEmpty()){
		_fullPath = fileName;
		int begin = fileName.lastIndexOf("/");
		int end = fileName.lastIndexOf(".");
		_name = fileName.mid(begin + 1, end - begin - 1);
		_path = fileName.mid(0, begin + 1);
		this->close();
	}
	//this->close();
	isDialogOk = false;
}