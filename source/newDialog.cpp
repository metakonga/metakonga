#include "newDialog.h"
#include "model.h"
#include <QtWidgets>

newDialog::newDialog(QWidget* parent)
	: QDialog(parent)
	, isBrowser(false)
{
	setupUi(this);
	connect(PB_Ok, SIGNAL(clicked()), this, SLOT(Click_ok()));
	connect(PB_Browse, SIGNAL(clicked()), this, SLOT(Click_browse()));
	path = model::path + "/";// kor(getenv("USERPROFILE"));
	//_path += "/Documents/xdynamics/";
	name = "Model1";
	if (!QDir(path).exists())
		QDir().mkdir(path);
	LE_Name->setText(name);
	//QFileDialog::getExistingDirectory(this, "", _path);
	CB_GravityDirection->setCurrentIndex(4);
}

newDialog::~newDialog()
{
	//disconnect(PB_Ok);
	//disconnect(PBBrowse);
	//if (ui) delete ui; ui = NULL;
}

void newDialog::Click_ok()
{
	name = LE_Name->text();
	//_path = ui->LEPath->text();
	unit = (unit_type)CB_Unit->currentIndex();
	dir_g = (gravity_direction)CB_GravityDirection->currentIndex();
	isSinglePrecision = CBH_SingleFloating->isChecked();
	//_path += "/";
	this->close();
	this->setResult(QDialog::Accepted);
}

void newDialog::Click_browse()
{
	QString fileName = QFileDialog::getOpenFileName(this, tr("open"), path, tr("Model File(*.xdm)"));
	if (!fileName.isEmpty()){
		fullPath = fileName;
		int begin = fileName.lastIndexOf("/");
		int end = fileName.lastIndexOf(".");
		name = fileName.mid(begin + 1, end - begin - 1);
		path = fileName.mid(0, begin + 1);
	}
	else
		return;
	fullPath = fileName;
	isBrowser = true;
	this->close();
	this->setResult(QDialog::Accepted);
}