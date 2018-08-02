// #ifndef EXPORTDIALOG_H
// #define EXPORTDIALOG_H
// 
// #include <QDialog>
// 
// QT_BEGIN_NAMESPACE
// class QStringList;
// class QLineEdit;
// QT_END_NAMESPACE
// 
// class modeler;
// 
// class ExportDialog : public QDialog
// {
// 	Q_OBJECT
// 
// public:
// 	ExportDialog(modeler* _md);
// 	~ExportDialog();
// 
// 	bool callDialog();
// 
// 	QDialog *dlg;
// 	//QStringList fileNames;
// 	//QString fileName;
// 	QLineEdit *LEFrom;
// 	QLineEdit *LETo;
// 	QLineEdit *LETarget;
// 
// 	bool isDialogOk;
// 
// private:
// 	modeler *md;
// 
// 	private slots:
// 	void Click_ok();
// 	void Click_cancel();
// 	void FromAction();
// 	void ToAction();
// 
// 	
// };
// 
// #endif