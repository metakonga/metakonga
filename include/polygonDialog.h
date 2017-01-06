// #ifndef POLYGONDIALOG_H
// #define POLYGONDIALOG_H
// 
// #include <QDialog>
// 
// QT_BEGIN_NAMESPACE
// class QLabel;
// class QLineEdit;
// class QPushButton;
// class QGridLayout;
// class QRadioButton;
// class QComboBox;
// QT_END_NAMESPACE
// 
// class modeler;
// class polygon;
// 
// class polygonDialog : public QDialog
// {
// 	Q_OBJECT
// 
// public:
// 	polygonDialog();
// 	//cube(std::map<QString, QObject*> *_objs);
// 	~polygonDialog();
// 	polygon* callDialog(modeler *md);
// 
// 	bool isDialogOk;
// 	QComboBox *CBMaterial;
// 	QLineEdit *LEName;
// 	QLineEdit *LEP;
// 	QLineEdit *LEQ;
// 	QLineEdit *LER;
// 	QGridLayout *polyLayout;
// 	QPushButton *PBOk;
// 	QPushButton *PBCancel;
// 
// 	private slots:
// 	void Click_ok();
// 	void Click_cancel();
// };
// 
// 
// #endif