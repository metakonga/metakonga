/********************************************************************************
** Form generated from reading UI file 'makeParticle.ui'
**
** Created by: Qt User Interface Compiler version 5.7.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAKEPARTICLE_H
#define UI_MAKEPARTICLE_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QDialog>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QTabWidget>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_DLG_MakeParticle
{
public:
    QTabWidget *TB_Method;
    QWidget *T_Cube;
    QWidget *layoutWidget;
    QHBoxLayout *horizontalLayout;
    QLabel *L_CUBE_NX;
    QLineEdit *LE_CUBE_NX;
    QLabel *L_CUBE_NY;
    QLineEdit *LE_CUBE_NY;
    QLabel *L_CUBE_NZ;
    QLineEdit *LE_CUBE_NZ;
    QWidget *layoutWidget1;
    QHBoxLayout *horizontalLayout_6;
    QLabel *L_CUBE_LOC;
    QLineEdit *LE_CUBE_LOC;
    QWidget *T_Plane;
    QWidget *layoutWidget2;
    QHBoxLayout *horizontalLayout_7;
    QLabel *L_PLANE_LOC;
    QLineEdit *LE_PLANE_LOC;
    QWidget *layoutWidget3;
    QHBoxLayout *horizontalLayout_3;
    QLabel *L_PLANE_NX;
    QLineEdit *LE_PLANE_NX;
    QLabel *L_PLANE_NZ;
    QLineEdit *LE_PLANE_NZ;
    QWidget *layoutWidget4;
    QHBoxLayout *horizontalLayout_5;
    QLabel *L_PLANE_DIR;
    QLineEdit *LE_PLANE_DIR;
    QWidget *T_Sphere;
    QGroupBox *groupBox;
    QWidget *layoutWidget5;
    QHBoxLayout *horizontalLayout_2;
    QLabel *L_MIN_RADIUS;
    QLineEdit *LE_MIN_RADIUS;
    QLabel *L_MAX_RADIUS;
    QLineEdit *LE_MAX_RADIUS;
    QWidget *layoutWidget6;
    QGridLayout *gridLayout;
    QLabel *L_P_SPACING;
    QLineEdit *LE_P_SPACING;
    QLabel *L_NUM_PARTICLE;
    QLineEdit *LE_NUM_PARTICLE;
    QGroupBox *GB_MaterialProperty;
    QWidget *layoutWidget_2;
    QGridLayout *gridLayout_2;
    QLabel *L_Type;
    QComboBox *CB_Type;
    QLabel *L_YoungsModulus;
    QLineEdit *LE_Youngs;
    QLabel *L_Density;
    QLineEdit *LE_Density;
    QLabel *L_PoissonRatio;
    QLineEdit *LE_PoissonRatio;
    QLabel *L_ShearModulus;
    QLineEdit *LE_ShearModulus;
    QWidget *layoutWidget7;
    QHBoxLayout *horizontalLayout_4;
    QPushButton *PB_Ok;
    QPushButton *PB_Cancle;
    QGroupBox *GB_BASIC;
    QLabel *L_Name;
    QLineEdit *LE_Name;

    void setupUi(QDialog *DLG_MakeParticle)
    {
        if (DLG_MakeParticle->objectName().isEmpty())
            DLG_MakeParticle->setObjectName(QStringLiteral("DLG_MakeParticle"));
        DLG_MakeParticle->resize(389, 485);
        TB_Method = new QTabWidget(DLG_MakeParticle);
        TB_Method->setObjectName(QStringLiteral("TB_Method"));
        TB_Method->setGeometry(QRect(10, 70, 371, 91));
        T_Cube = new QWidget();
        T_Cube->setObjectName(QStringLiteral("T_Cube"));
        layoutWidget = new QWidget(T_Cube);
        layoutWidget->setObjectName(QStringLiteral("layoutWidget"));
        layoutWidget->setGeometry(QRect(10, 10, 351, 22));
        horizontalLayout = new QHBoxLayout(layoutWidget);
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        horizontalLayout->setContentsMargins(0, 0, 0, 0);
        L_CUBE_NX = new QLabel(layoutWidget);
        L_CUBE_NX->setObjectName(QStringLiteral("L_CUBE_NX"));

        horizontalLayout->addWidget(L_CUBE_NX);

        LE_CUBE_NX = new QLineEdit(layoutWidget);
        LE_CUBE_NX->setObjectName(QStringLiteral("LE_CUBE_NX"));

        horizontalLayout->addWidget(LE_CUBE_NX);

        L_CUBE_NY = new QLabel(layoutWidget);
        L_CUBE_NY->setObjectName(QStringLiteral("L_CUBE_NY"));

        horizontalLayout->addWidget(L_CUBE_NY);

        LE_CUBE_NY = new QLineEdit(layoutWidget);
        LE_CUBE_NY->setObjectName(QStringLiteral("LE_CUBE_NY"));

        horizontalLayout->addWidget(LE_CUBE_NY);

        L_CUBE_NZ = new QLabel(layoutWidget);
        L_CUBE_NZ->setObjectName(QStringLiteral("L_CUBE_NZ"));

        horizontalLayout->addWidget(L_CUBE_NZ);

        LE_CUBE_NZ = new QLineEdit(layoutWidget);
        LE_CUBE_NZ->setObjectName(QStringLiteral("LE_CUBE_NZ"));

        horizontalLayout->addWidget(LE_CUBE_NZ);

        layoutWidget1 = new QWidget(T_Cube);
        layoutWidget1->setObjectName(QStringLiteral("layoutWidget1"));
        layoutWidget1->setGeometry(QRect(10, 40, 351, 22));
        horizontalLayout_6 = new QHBoxLayout(layoutWidget1);
        horizontalLayout_6->setObjectName(QStringLiteral("horizontalLayout_6"));
        horizontalLayout_6->setContentsMargins(0, 0, 0, 0);
        L_CUBE_LOC = new QLabel(layoutWidget1);
        L_CUBE_LOC->setObjectName(QStringLiteral("L_CUBE_LOC"));

        horizontalLayout_6->addWidget(L_CUBE_LOC);

        LE_CUBE_LOC = new QLineEdit(layoutWidget1);
        LE_CUBE_LOC->setObjectName(QStringLiteral("LE_CUBE_LOC"));

        horizontalLayout_6->addWidget(LE_CUBE_LOC);

        TB_Method->addTab(T_Cube, QString());
        T_Plane = new QWidget();
        T_Plane->setObjectName(QStringLiteral("T_Plane"));
        layoutWidget2 = new QWidget(T_Plane);
        layoutWidget2->setObjectName(QStringLiteral("layoutWidget2"));
        layoutWidget2->setGeometry(QRect(10, 40, 351, 22));
        horizontalLayout_7 = new QHBoxLayout(layoutWidget2);
        horizontalLayout_7->setObjectName(QStringLiteral("horizontalLayout_7"));
        horizontalLayout_7->setContentsMargins(0, 0, 0, 0);
        L_PLANE_LOC = new QLabel(layoutWidget2);
        L_PLANE_LOC->setObjectName(QStringLiteral("L_PLANE_LOC"));

        horizontalLayout_7->addWidget(L_PLANE_LOC);

        LE_PLANE_LOC = new QLineEdit(layoutWidget2);
        LE_PLANE_LOC->setObjectName(QStringLiteral("LE_PLANE_LOC"));

        horizontalLayout_7->addWidget(LE_PLANE_LOC);

        layoutWidget3 = new QWidget(T_Plane);
        layoutWidget3->setObjectName(QStringLiteral("layoutWidget3"));
        layoutWidget3->setGeometry(QRect(10, 10, 191, 22));
        horizontalLayout_3 = new QHBoxLayout(layoutWidget3);
        horizontalLayout_3->setObjectName(QStringLiteral("horizontalLayout_3"));
        horizontalLayout_3->setContentsMargins(0, 0, 0, 0);
        L_PLANE_NX = new QLabel(layoutWidget3);
        L_PLANE_NX->setObjectName(QStringLiteral("L_PLANE_NX"));

        horizontalLayout_3->addWidget(L_PLANE_NX);

        LE_PLANE_NX = new QLineEdit(layoutWidget3);
        LE_PLANE_NX->setObjectName(QStringLiteral("LE_PLANE_NX"));

        horizontalLayout_3->addWidget(LE_PLANE_NX);

        L_PLANE_NZ = new QLabel(layoutWidget3);
        L_PLANE_NZ->setObjectName(QStringLiteral("L_PLANE_NZ"));

        horizontalLayout_3->addWidget(L_PLANE_NZ);

        LE_PLANE_NZ = new QLineEdit(layoutWidget3);
        LE_PLANE_NZ->setObjectName(QStringLiteral("LE_PLANE_NZ"));

        horizontalLayout_3->addWidget(LE_PLANE_NZ);

        layoutWidget4 = new QWidget(T_Plane);
        layoutWidget4->setObjectName(QStringLiteral("layoutWidget4"));
        layoutWidget4->setGeometry(QRect(210, 10, 151, 22));
        horizontalLayout_5 = new QHBoxLayout(layoutWidget4);
        horizontalLayout_5->setObjectName(QStringLiteral("horizontalLayout_5"));
        horizontalLayout_5->setContentsMargins(0, 0, 0, 0);
        L_PLANE_DIR = new QLabel(layoutWidget4);
        L_PLANE_DIR->setObjectName(QStringLiteral("L_PLANE_DIR"));

        horizontalLayout_5->addWidget(L_PLANE_DIR);

        LE_PLANE_DIR = new QLineEdit(layoutWidget4);
        LE_PLANE_DIR->setObjectName(QStringLiteral("LE_PLANE_DIR"));

        horizontalLayout_5->addWidget(LE_PLANE_DIR);

        TB_Method->addTab(T_Plane, QString());
        T_Sphere = new QWidget();
        T_Sphere->setObjectName(QStringLiteral("T_Sphere"));
        TB_Method->addTab(T_Sphere, QString());
        groupBox = new QGroupBox(DLG_MakeParticle);
        groupBox->setObjectName(QStringLiteral("groupBox"));
        groupBox->setGeometry(QRect(10, 170, 371, 106));
        layoutWidget5 = new QWidget(groupBox);
        layoutWidget5->setObjectName(QStringLiteral("layoutWidget5"));
        layoutWidget5->setGeometry(QRect(10, 20, 351, 22));
        horizontalLayout_2 = new QHBoxLayout(layoutWidget5);
        horizontalLayout_2->setObjectName(QStringLiteral("horizontalLayout_2"));
        horizontalLayout_2->setContentsMargins(0, 0, 0, 0);
        L_MIN_RADIUS = new QLabel(layoutWidget5);
        L_MIN_RADIUS->setObjectName(QStringLiteral("L_MIN_RADIUS"));

        horizontalLayout_2->addWidget(L_MIN_RADIUS);

        LE_MIN_RADIUS = new QLineEdit(layoutWidget5);
        LE_MIN_RADIUS->setObjectName(QStringLiteral("LE_MIN_RADIUS"));

        horizontalLayout_2->addWidget(LE_MIN_RADIUS);

        L_MAX_RADIUS = new QLabel(layoutWidget5);
        L_MAX_RADIUS->setObjectName(QStringLiteral("L_MAX_RADIUS"));

        horizontalLayout_2->addWidget(L_MAX_RADIUS);

        LE_MAX_RADIUS = new QLineEdit(layoutWidget5);
        LE_MAX_RADIUS->setObjectName(QStringLiteral("LE_MAX_RADIUS"));

        horizontalLayout_2->addWidget(LE_MAX_RADIUS);

        layoutWidget6 = new QWidget(groupBox);
        layoutWidget6->setObjectName(QStringLiteral("layoutWidget6"));
        layoutWidget6->setGeometry(QRect(10, 50, 351, 48));
        gridLayout = new QGridLayout(layoutWidget6);
        gridLayout->setObjectName(QStringLiteral("gridLayout"));
        gridLayout->setContentsMargins(0, 0, 0, 0);
        L_P_SPACING = new QLabel(layoutWidget6);
        L_P_SPACING->setObjectName(QStringLiteral("L_P_SPACING"));

        gridLayout->addWidget(L_P_SPACING, 0, 0, 1, 1);

        LE_P_SPACING = new QLineEdit(layoutWidget6);
        LE_P_SPACING->setObjectName(QStringLiteral("LE_P_SPACING"));

        gridLayout->addWidget(LE_P_SPACING, 0, 1, 1, 1);

        L_NUM_PARTICLE = new QLabel(layoutWidget6);
        L_NUM_PARTICLE->setObjectName(QStringLiteral("L_NUM_PARTICLE"));

        gridLayout->addWidget(L_NUM_PARTICLE, 1, 0, 1, 1);

        LE_NUM_PARTICLE = new QLineEdit(layoutWidget6);
        LE_NUM_PARTICLE->setObjectName(QStringLiteral("LE_NUM_PARTICLE"));
        LE_NUM_PARTICLE->setReadOnly(true);

        gridLayout->addWidget(LE_NUM_PARTICLE, 1, 1, 1, 1);

        GB_MaterialProperty = new QGroupBox(DLG_MakeParticle);
        GB_MaterialProperty->setObjectName(QStringLiteral("GB_MaterialProperty"));
        GB_MaterialProperty->setGeometry(QRect(10, 284, 371, 161));
        layoutWidget_2 = new QWidget(GB_MaterialProperty);
        layoutWidget_2->setObjectName(QStringLiteral("layoutWidget_2"));
        layoutWidget_2->setGeometry(QRect(10, 21, 351, 126));
        gridLayout_2 = new QGridLayout(layoutWidget_2);
        gridLayout_2->setObjectName(QStringLiteral("gridLayout_2"));
        gridLayout_2->setContentsMargins(0, 0, 0, 0);
        L_Type = new QLabel(layoutWidget_2);
        L_Type->setObjectName(QStringLiteral("L_Type"));

        gridLayout_2->addWidget(L_Type, 0, 0, 1, 1);

        CB_Type = new QComboBox(layoutWidget_2);
        CB_Type->setObjectName(QStringLiteral("CB_Type"));

        gridLayout_2->addWidget(CB_Type, 0, 1, 1, 1);

        L_YoungsModulus = new QLabel(layoutWidget_2);
        L_YoungsModulus->setObjectName(QStringLiteral("L_YoungsModulus"));

        gridLayout_2->addWidget(L_YoungsModulus, 1, 0, 1, 1);

        LE_Youngs = new QLineEdit(layoutWidget_2);
        LE_Youngs->setObjectName(QStringLiteral("LE_Youngs"));

        gridLayout_2->addWidget(LE_Youngs, 1, 1, 1, 1);

        L_Density = new QLabel(layoutWidget_2);
        L_Density->setObjectName(QStringLiteral("L_Density"));

        gridLayout_2->addWidget(L_Density, 2, 0, 1, 1);

        LE_Density = new QLineEdit(layoutWidget_2);
        LE_Density->setObjectName(QStringLiteral("LE_Density"));

        gridLayout_2->addWidget(LE_Density, 2, 1, 1, 1);

        L_PoissonRatio = new QLabel(layoutWidget_2);
        L_PoissonRatio->setObjectName(QStringLiteral("L_PoissonRatio"));

        gridLayout_2->addWidget(L_PoissonRatio, 3, 0, 1, 1);

        LE_PoissonRatio = new QLineEdit(layoutWidget_2);
        LE_PoissonRatio->setObjectName(QStringLiteral("LE_PoissonRatio"));

        gridLayout_2->addWidget(LE_PoissonRatio, 3, 1, 1, 1);

        L_ShearModulus = new QLabel(layoutWidget_2);
        L_ShearModulus->setObjectName(QStringLiteral("L_ShearModulus"));

        gridLayout_2->addWidget(L_ShearModulus, 4, 0, 1, 1);

        LE_ShearModulus = new QLineEdit(layoutWidget_2);
        LE_ShearModulus->setObjectName(QStringLiteral("LE_ShearModulus"));

        gridLayout_2->addWidget(LE_ShearModulus, 4, 1, 1, 1);

        layoutWidget7 = new QWidget(DLG_MakeParticle);
        layoutWidget7->setObjectName(QStringLiteral("layoutWidget7"));
        layoutWidget7->setGeometry(QRect(190, 450, 191, 25));
        horizontalLayout_4 = new QHBoxLayout(layoutWidget7);
        horizontalLayout_4->setObjectName(QStringLiteral("horizontalLayout_4"));
        horizontalLayout_4->setContentsMargins(0, 0, 0, 0);
        PB_Ok = new QPushButton(layoutWidget7);
        PB_Ok->setObjectName(QStringLiteral("PB_Ok"));

        horizontalLayout_4->addWidget(PB_Ok);

        PB_Cancle = new QPushButton(layoutWidget7);
        PB_Cancle->setObjectName(QStringLiteral("PB_Cancle"));

        horizontalLayout_4->addWidget(PB_Cancle);

        GB_BASIC = new QGroupBox(DLG_MakeParticle);
        GB_BASIC->setObjectName(QStringLiteral("GB_BASIC"));
        GB_BASIC->setGeometry(QRect(10, 10, 371, 53));
        L_Name = new QLabel(GB_BASIC);
        L_Name->setObjectName(QStringLiteral("L_Name"));
        L_Name->setGeometry(QRect(10, 21, 41, 20));
        LE_Name = new QLineEdit(GB_BASIC);
        LE_Name->setObjectName(QStringLiteral("LE_Name"));
        LE_Name->setGeometry(QRect(60, 21, 301, 20));
        QWidget::setTabOrder(TB_Method, LE_CUBE_NX);
        QWidget::setTabOrder(LE_CUBE_NX, LE_CUBE_NY);
        QWidget::setTabOrder(LE_CUBE_NY, LE_CUBE_NZ);
        QWidget::setTabOrder(LE_CUBE_NZ, LE_MIN_RADIUS);
        QWidget::setTabOrder(LE_MIN_RADIUS, LE_MAX_RADIUS);
        QWidget::setTabOrder(LE_MAX_RADIUS, LE_P_SPACING);
        QWidget::setTabOrder(LE_P_SPACING, LE_NUM_PARTICLE);
        QWidget::setTabOrder(LE_NUM_PARTICLE, PB_Ok);
        QWidget::setTabOrder(PB_Ok, PB_Cancle);
        QWidget::setTabOrder(PB_Cancle, LE_PLANE_NX);
        QWidget::setTabOrder(LE_PLANE_NX, LE_PLANE_NZ);
        QWidget::setTabOrder(LE_PLANE_NZ, LE_PLANE_DIR);

        retranslateUi(DLG_MakeParticle);

        TB_Method->setCurrentIndex(1);


        QMetaObject::connectSlotsByName(DLG_MakeParticle);
    } // setupUi

    void retranslateUi(QDialog *DLG_MakeParticle)
    {
        DLG_MakeParticle->setWindowTitle(QApplication::translate("DLG_MakeParticle", "Make particles", Q_NULLPTR));
        L_CUBE_NX->setText(QApplication::translate("DLG_MakeParticle", "Nx", Q_NULLPTR));
        L_CUBE_NY->setText(QApplication::translate("DLG_MakeParticle", "Ny", Q_NULLPTR));
        L_CUBE_NZ->setText(QApplication::translate("DLG_MakeParticle", "Nz", Q_NULLPTR));
        L_CUBE_LOC->setText(QApplication::translate("DLG_MakeParticle", "Start loc.", Q_NULLPTR));
        TB_Method->setTabText(TB_Method->indexOf(T_Cube), QApplication::translate("DLG_MakeParticle", "Cube", Q_NULLPTR));
        L_PLANE_LOC->setText(QApplication::translate("DLG_MakeParticle", "Start loc.", Q_NULLPTR));
        L_PLANE_NX->setText(QApplication::translate("DLG_MakeParticle", "Nx", Q_NULLPTR));
        L_PLANE_NZ->setText(QApplication::translate("DLG_MakeParticle", "Nz", Q_NULLPTR));
        L_PLANE_DIR->setText(QApplication::translate("DLG_MakeParticle", "Dir.", Q_NULLPTR));
        TB_Method->setTabText(TB_Method->indexOf(T_Plane), QApplication::translate("DLG_MakeParticle", "Plane", Q_NULLPTR));
        TB_Method->setTabText(TB_Method->indexOf(T_Sphere), QApplication::translate("DLG_MakeParticle", "Sphere", Q_NULLPTR));
        groupBox->setTitle(QApplication::translate("DLG_MakeParticle", "Particle property", Q_NULLPTR));
        L_MIN_RADIUS->setText(QApplication::translate("DLG_MakeParticle", "Min. radius", Q_NULLPTR));
        L_MAX_RADIUS->setText(QApplication::translate("DLG_MakeParticle", "Max. radius", Q_NULLPTR));
        L_P_SPACING->setText(QApplication::translate("DLG_MakeParticle", "Particle spacing", Q_NULLPTR));
        L_NUM_PARTICLE->setText(QApplication::translate("DLG_MakeParticle", "Num. particle", Q_NULLPTR));
        GB_MaterialProperty->setTitle(QApplication::translate("DLG_MakeParticle", "Material property", Q_NULLPTR));
        L_Type->setText(QApplication::translate("DLG_MakeParticle", "Type", Q_NULLPTR));
        L_YoungsModulus->setText(QApplication::translate("DLG_MakeParticle", "Youngs modulus", Q_NULLPTR));
        L_Density->setText(QApplication::translate("DLG_MakeParticle", "Density", Q_NULLPTR));
        L_PoissonRatio->setText(QApplication::translate("DLG_MakeParticle", "Poisson ratio", Q_NULLPTR));
        L_ShearModulus->setText(QApplication::translate("DLG_MakeParticle", "Shear modulus", Q_NULLPTR));
        PB_Ok->setText(QApplication::translate("DLG_MakeParticle", "Ok", Q_NULLPTR));
        PB_Cancle->setText(QApplication::translate("DLG_MakeParticle", "Cancle", Q_NULLPTR));
        GB_BASIC->setTitle(QApplication::translate("DLG_MakeParticle", "GroupBox", Q_NULLPTR));
        L_Name->setText(QApplication::translate("DLG_MakeParticle", "Name", Q_NULLPTR));
    } // retranslateUi

};

namespace Ui {
    class DLG_MakeParticle: public Ui_DLG_MakeParticle {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAKEPARTICLE_H
