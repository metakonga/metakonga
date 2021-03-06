/********************************************************************************
** Form generated from reading UI file 'newModel.ui'
**
** Created by: Qt User Interface Compiler version 5.10.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_NEWMODEL_H
#define UI_NEWMODEL_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QDialog>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_DLG_NewModel
{
public:
    QGroupBox *GBNewModel;
    QPushButton *PB_Ok;
    QWidget *layoutWidget;
    QHBoxLayout *horizontalLayout;
    QVBoxLayout *layout_1;
    QLabel *L_Name;
    QLabel *L_Unit;
    QLabel *L_Gravity;
    QVBoxLayout *Layout_2;
    QLineEdit *LE_Name;
    QComboBox *CB_Unit;
    QComboBox *CB_GravityDirection;
    QCheckBox *CBH_SingleFloating;
    QLabel *LOpenModel;
    QPushButton *PB_Browse;

    void setupUi(QDialog *DLG_NewModel)
    {
        if (DLG_NewModel->objectName().isEmpty())
            DLG_NewModel->setObjectName(QStringLiteral("DLG_NewModel"));
        DLG_NewModel->setWindowModality(Qt::ApplicationModal);
        DLG_NewModel->resize(440, 200);
        QSizePolicy sizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(DLG_NewModel->sizePolicy().hasHeightForWidth());
        DLG_NewModel->setSizePolicy(sizePolicy);
        DLG_NewModel->setMinimumSize(QSize(440, 178));
        DLG_NewModel->setMaximumSize(QSize(440, 200));
        DLG_NewModel->setModal(true);
        GBNewModel = new QGroupBox(DLG_NewModel);
        GBNewModel->setObjectName(QStringLiteral("GBNewModel"));
        GBNewModel->setGeometry(QRect(10, 10, 421, 151));
        QFont font;
        font.setBold(false);
        font.setWeight(50);
        GBNewModel->setFont(font);
        GBNewModel->setAutoFillBackground(false);
        PB_Ok = new QPushButton(GBNewModel);
        PB_Ok->setObjectName(QStringLiteral("PB_Ok"));
        PB_Ok->setGeometry(QRect(330, 23, 75, 91));
        QFont font1;
        font1.setPointSize(34);
        PB_Ok->setFont(font1);
        layoutWidget = new QWidget(GBNewModel);
        layoutWidget->setObjectName(QStringLiteral("layoutWidget"));
        layoutWidget->setGeometry(QRect(11, 23, 311, 91));
        horizontalLayout = new QHBoxLayout(layoutWidget);
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        horizontalLayout->setContentsMargins(0, 0, 0, 0);
        layout_1 = new QVBoxLayout();
        layout_1->setObjectName(QStringLiteral("layout_1"));
        L_Name = new QLabel(layoutWidget);
        L_Name->setObjectName(QStringLiteral("L_Name"));

        layout_1->addWidget(L_Name);

        L_Unit = new QLabel(layoutWidget);
        L_Unit->setObjectName(QStringLiteral("L_Unit"));

        layout_1->addWidget(L_Unit);

        L_Gravity = new QLabel(layoutWidget);
        L_Gravity->setObjectName(QStringLiteral("L_Gravity"));

        layout_1->addWidget(L_Gravity);


        horizontalLayout->addLayout(layout_1);

        Layout_2 = new QVBoxLayout();
        Layout_2->setObjectName(QStringLiteral("Layout_2"));
        LE_Name = new QLineEdit(layoutWidget);
        LE_Name->setObjectName(QStringLiteral("LE_Name"));

        Layout_2->addWidget(LE_Name);

        CB_Unit = new QComboBox(layoutWidget);
        CB_Unit->addItem(QString());
        CB_Unit->setObjectName(QStringLiteral("CB_Unit"));

        Layout_2->addWidget(CB_Unit);

        CB_GravityDirection = new QComboBox(layoutWidget);
        CB_GravityDirection->addItem(QString());
        CB_GravityDirection->addItem(QString());
        CB_GravityDirection->addItem(QString());
        CB_GravityDirection->addItem(QString());
        CB_GravityDirection->addItem(QString());
        CB_GravityDirection->addItem(QString());
        CB_GravityDirection->setObjectName(QStringLiteral("CB_GravityDirection"));

        Layout_2->addWidget(CB_GravityDirection);


        horizontalLayout->addLayout(Layout_2);

        CBH_SingleFloating = new QCheckBox(GBNewModel);
        CBH_SingleFloating->setObjectName(QStringLiteral("CBH_SingleFloating"));
        CBH_SingleFloating->setGeometry(QRect(10, 123, 211, 19));
        LOpenModel = new QLabel(DLG_NewModel);
        LOpenModel->setObjectName(QStringLiteral("LOpenModel"));
        LOpenModel->setGeometry(QRect(20, 170, 81, 16));
        PB_Browse = new QPushButton(DLG_NewModel);
        PB_Browse->setObjectName(QStringLiteral("PB_Browse"));
        PB_Browse->setGeometry(QRect(340, 167, 75, 23));
        PB_Browse->setStyleSheet(QStringLiteral("border-bottom-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, stop:0 rgba(255, 0, 0, 255), stop:0.339795 rgba(255, 0, 0, 255), stop:0.339799 rgba(255, 255, 255, 255), stop:0.662444 rgba(255, 255, 255, 255), stop:0.662469 rgba(0, 0, 255, 255), stop:1 rgba(0, 0, 255, 255));"));
        PB_Browse->setFlat(false);
        QWidget::setTabOrder(LE_Name, CB_Unit);
        QWidget::setTabOrder(CB_Unit, CB_GravityDirection);
        QWidget::setTabOrder(CB_GravityDirection, PB_Ok);
        QWidget::setTabOrder(PB_Ok, PB_Browse);

        retranslateUi(DLG_NewModel);

        PB_Browse->setDefault(false);


        QMetaObject::connectSlotsByName(DLG_NewModel);
    } // setupUi

    void retranslateUi(QDialog *DLG_NewModel)
    {
        DLG_NewModel->setWindowTitle(QApplication::translate("DLG_NewModel", "New Model", nullptr));
        GBNewModel->setTitle(QApplication::translate("DLG_NewModel", "New Model", nullptr));
        PB_Ok->setText(QApplication::translate("DLG_NewModel", "OK", nullptr));
        L_Name->setText(QApplication::translate("DLG_NewModel", "Name", nullptr));
        L_Unit->setText(QApplication::translate("DLG_NewModel", "Unit", nullptr));
        L_Gravity->setText(QApplication::translate("DLG_NewModel", "Gravity", nullptr));
        CB_Unit->setItemText(0, QApplication::translate("DLG_NewModel", "MKS(Meter/Kilogram/Newton/Second)", nullptr));

        CB_GravityDirection->setItemText(0, QApplication::translate("DLG_NewModel", "+X", nullptr));
        CB_GravityDirection->setItemText(1, QApplication::translate("DLG_NewModel", "+Y", nullptr));
        CB_GravityDirection->setItemText(2, QApplication::translate("DLG_NewModel", "+Z", nullptr));
        CB_GravityDirection->setItemText(3, QApplication::translate("DLG_NewModel", "-X", nullptr));
        CB_GravityDirection->setItemText(4, QApplication::translate("DLG_NewModel", "-Y", nullptr));
        CB_GravityDirection->setItemText(5, QApplication::translate("DLG_NewModel", "-Z", nullptr));

        CBH_SingleFloating->setText(QApplication::translate("DLG_NewModel", "Single floating point type for DEM", nullptr));
        LOpenModel->setText(QApplication::translate("DLG_NewModel", "Open Model", nullptr));
        PB_Browse->setText(QApplication::translate("DLG_NewModel", "Browse", nullptr));
    } // retranslateUi

};

namespace Ui {
    class DLG_NewModel: public Ui_DLG_NewModel {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_NEWMODEL_H
