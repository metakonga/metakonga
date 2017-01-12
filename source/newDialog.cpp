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
	_path = getenv("USERPROFILE");
	_path += "/Documents/xdynamics/";
	_name = "Model1";
	if (!QDir(_path).exists())
		QDir().mkdir(_path);
	//QFileDialog::getExistingDirectory(this, "", _path);
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