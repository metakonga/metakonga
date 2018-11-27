/********************************************************************************
** Form generated from reading UI file 'makeParticleBnUhVd.ui'
**
** Created by: Qt User Interface Compiler version 5.10.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef MAKEPARTICLEBNUHVD_H
#define MAKEPARTICLEBNUHVD_H

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
#include <QtWidgets/QRadioButton>
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
    QLabel *L_CUBE_DX;
    QLineEdit *LE_CUBE_DX;
    QLabel *L_CUBE_DY;
    QLineEdit *LE_CUBE_DY;
    QLabel *L_CUBE_DZ;
    QLineEdit *LE_CUBE_DZ;
    QWidget *layoutWidget1;
    QHBoxLayout *horizontalLayout_6;
    QLabel *L_CUBE_LOC;
    QLineEdit *LE_CUBE_LOC;
    QWidget *T_Plane;
    QWidget *layoutWidget2;
    QHBoxLayout *horizontalLayout_7;
    QLabel *L_PLANE_LOC;
    QLineEdit *LE_PLANE_LOC;
    QWidget *widget;
    QHBoxLayout *horizontalLayout_3;
    QLabel *L_PLANE_DX;
    QLineEdit *LE_PLANE_DX;
    QLabel *L_PLANE_NY;
    QLineEdit *LE_PLANE_NY;
    QLabel *L_PLANE_DZ;
    QLineEdit *LE_PLANE_DZ;
    QWidget *widget1;
    QHBoxLayout *horizontalLayout_5;
    QLabel *L_PLANE_DIR;
    QLineEdit *LE_PLANE_DIR;
    QWidget *T_Circle;
    QWidget *layoutWidget3;
    QHBoxLayout *horizontalLayout_9;
    QLabel *L_CircleLocation;
    QLineEdit *LE_CircleLocation;
    QLabel *L_CircleDirection;
    QLineEdit *LE_CircleDirection;
    QWidget *layoutWidget4;
    QHBoxLayout *horizontalLayout_8;
    QLabel *L_CircleDiameter;
    QLineEdit *LE_CircleDiameter;
    QLabel *L_NumParticle;
    QLineEdit *LE_NumHeight;
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
    QGroupBox *GB_RealTime;
    QWidget *layoutWidget8;
    QHBoxLayout *horizontalLayout_10;
    QLabel *L_NumParticlesPer;
    QLineEdit *LE_NumParclesPer;
    QRadioButton *RB_OneByOne;
    QRadioButton *RB_OneByGroup;
    QPushButton *PB_CalNumParticles;

    void setupUi(QDialog *DLG_MakeParticle)
    {
        if (DLG_MakeParticle->objectName().isEmpty())
            DLG_MakeParticle->setObjectName(QStringLiteral("DLG_MakeParticle"));
        DLG_MakeParticle->resize(389, 565);
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
        L_CUBE_DX = new QLabel(layoutWidget);
        L_CUBE_DX->setObjectName(QStringLiteral("L_CUBE_DX"));

        horizontalLayout->addWidget(L_CUBE_DX);

        LE_CUBE_DX = new QLineEdit(layoutWidget);
        LE_CUBE_DX->setObjectName(QStringLiteral("LE_CUBE_DX"));

        horizontalLayout->addWidget(LE_CUBE_DX);

        L_CUBE_DY = new QLabel(layoutWidget);
        L_CUBE_DY->setObjectName(QStringLiteral("L_CUBE_DY"));

        horizontalLayout->addWidget(L_CUBE_DY);

        LE_CUBE_DY = new QLineEdit(layoutWidget);
        LE_CUBE_DY->setObjectName(QStringLiteral("LE_CUBE_DY"));

        horizontalLayout->addWidget(LE_CUBE_DY);

        L_CUBE_DZ = new QLabel(layoutWidget);
        L_CUBE_DZ->setObjectName(QStringLiteral("L_CUBE_DZ"));

        horizontalLayout->addWidget(L_CUBE_DZ);

        LE_CUBE_DZ = new QLineEdit(layoutWidget);
        LE_CUBE_DZ->setObjectName(QStringLiteral("LE_CUBE_DZ"));

        horizontalLayout->addWidget(LE_CUBE_DZ);

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

        widget = new QWidget(T_Plane);
        widget->setObjectName(QStringLiteral("widget"));
        widget->setGeometry(QRect(10, 10, 211, 22));
        horizontalLayout_3 = new QHBoxLayout(widget);
        horizontalLayout_3->setObjectName(QStringLiteral("horizontalLayout_3"));
        horizontalLayout_3->setContentsMargins(0, 0, 0, 0);
        L_PLANE_DX = new QLabel(widget);
        L_PLANE_DX->setObjectName(QStringLiteral("L_PLANE_DX"));

        horizontalLayout_3->addWidget(L_PLANE_DX);

        LE_PLANE_DX = new QLineEdit(widget);
        LE_PLANE_DX->setObjectName(QStringLiteral("LE_PLANE_DX"));

        horizontalLayout_3->addWidget(LE_PLANE_DX);

        L_PLANE_NY = new QLabel(widget);
        L_PLANE_NY->setObjectName(QStringLiteral("L_PLANE_NY"));

        horizontalLayout_3->addWidget(L_PLANE_NY);

        LE_PLANE_NY = new QLineEdit(widget);
        LE_PLANE_NY->setObjectName(QStringLiteral("LE_PLANE_NY"));

        horizontalLayout_3->addWidget(LE_PLANE_NY);

        L_PLANE_DZ = new QLabel(widget);
        L_PLANE_DZ->setObjectName(QStringLiteral("L_PLANE_DZ"));

        horizontalLayout_3->addWidget(L_PLANE_DZ);

        LE_PLANE_DZ = new QLineEdit(widget);
        LE_PLANE_DZ->setObjectName(QStringLiteral("LE_PLANE_DZ"));

        horizontalLayout_3->addWidget(LE_PLANE_DZ);

        widget1 = new QWidget(T_Plane);
        widget1->setObjectName(QStringLiteral("widget1"));
        widget1->setGeometry(QRect(230, 10, 131, 22));
        horizontalLayout_5 = new QHBoxLayout(widget1);
        horizontalLayout_5->setObjectName(QStringLiteral("horizontalLayout_5"));
        horizontalLayout_5->setContentsMargins(0, 0, 0, 0);
        L_PLANE_DIR = new QLabel(widget1);
        L_PLANE_DIR->setObjectName(QStringLiteral("L_PLANE_DIR"));

        horizontalLayout_5->addWidget(L_PLANE_DIR);

        LE_PLANE_DIR = new QLineEdit(widget1);
        LE_PLANE_DIR->setObjectName(QStringLiteral("LE_PLANE_DIR"));

        horizontalLayout_5->addWidget(LE_PLANE_DIR);

        TB_Method->addTab(T_Plane, QString());
        T_Circle = new QWidget();
        T_Circle->setObjectName(QStringLiteral("T_Circle"));
        layoutWidget3 = new QWidget(T_Circle);
        layoutWidget3->setObjectName(QStringLiteral("layoutWidget3"));
        layoutWidget3->setGeometry(QRect(10, 38, 341, 22));
        horizontalLayout_9 = new QHBoxLayout(layoutWidget3);
        horizontalLayout_9->setObjectName(QStringLiteral("horizontalLayout_9"));
        horizontalLayout_9->setContentsMargins(0, 0, 0, 0);
        L_CircleLocation = new QLabel(layoutWidget3);
        L_CircleLocation->setObjectName(QStringLiteral("L_CircleLocation"));

        horizontalLayout_9->addWidget(L_CircleLocation);

        LE_CircleLocation = new QLineEdit(layoutWidget3);
        LE_CircleLocation->setObjectName(QStringLiteral("LE_CircleLocation"));

        horizontalLayout_9->addWidget(LE_CircleLocation);

        L_CircleDirection = new QLabel(layoutWidget3);
        L_CircleDirection->setObjectName(QStringLiteral("L_CircleDirection"));

        horizontalLayout_9->addWidget(L_CircleDirection);

        LE_CircleDirection = new QLineEdit(layoutWidget3);
        LE_CircleDirection->setObjectName(QStringLiteral("LE_CircleDirection"));

        horizontalLayout_9->addWidget(LE_CircleDirection);

        layoutWidget4 = new QWidget(T_Circle);
        layoutWidget4->setObjectName(QStringLiteral("layoutWidget4"));
        layoutWidget4->setGeometry(QRect(10, 10, 341, 22));
        horizontalLayout_8 = new QHBoxLayout(layoutWidget4);
        horizontalLayout_8->setObjectName(QStringLiteral("horizontalLayout_8"));
        horizontalLayout_8->setContentsMargins(0, 0, 0, 0);
        L_CircleDiameter = new QLabel(layoutWidget4);
        L_CircleDiameter->setObjectName(QStringLiteral("L_CircleDiameter"));

        horizontalLayout_8->addWidget(L_CircleDiameter);

        LE_CircleDiameter = new QLineEdit(layoutWidget4);
        LE_CircleDiameter->setObjectName(QStringLiteral("LE_CircleDiameter"));

        horizontalLayout_8->addWidget(LE_CircleDiameter);

        L_NumParticle = new QLabel(layoutWidget4);
        L_NumParticle->setObjectName(QStringLiteral("L_NumParticle"));

        horizontalLayout_8->addWidget(L_NumParticle);

        LE_NumHeight = new QLineEdit(layoutWidget4);
        LE_NumHeight->setObjectName(QStringLiteral("LE_NumHeight"));

        horizontalLayout_8->addWidget(LE_NumHeight);

        TB_Method->addTab(T_Circle, QString());
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
        layoutWidget7->setGeometry(QRect(190, 530, 191, 25));
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
        GB_RealTime = new QGroupBox(DLG_MakeParticle);
        GB_RealTime->setObjectName(QStringLiteral("GB_RealTime"));
        GB_RealTime->setGeometry(QRect(10, 450, 371, 71));
        GB_RealTime->setCheckable(true);
        GB_RealTime->setChecked(false);
        layoutWidget8 = new QWidget(GB_RealTime);
        layoutWidget8->setObjectName(QStringLiteral("layoutWidget8"));
        layoutWidget8->setGeometry(QRect(10, 40, 351, 22));
        horizontalLayout_10 = new QHBoxLayout(layoutWidget8);
        horizontalLayout_10->setObjectName(QStringLiteral("horizontalLayout_10"));
        horizontalLayout_10->setContentsMargins(0, 0, 0, 0);
        L_NumParticlesPer = new QLabel(layoutWidget8);
        L_NumParticlesPer->setObjectName(QStringLiteral("L_NumParticlesPer"));

        horizontalLayout_10->addWidget(L_NumParticlesPer);

        LE_NumParclesPer = new QLineEdit(layoutWidget8);
        LE_NumParclesPer->setObjectName(QStringLiteral("LE_NumParclesPer"));

        horizontalLayout_10->addWidget(LE_NumParclesPer);

        RB_OneByOne = new QRadioButton(GB_RealTime);
        RB_OneByOne->setObjectName(QStringLiteral("RB_OneByOne"));
        RB_OneByOne->setGeometry(QRect(10, 20, 121, 16));
        RB_OneByOne->setChecked(true);
        RB_OneByGroup = new QRadioButton(GB_RealTime);
        RB_OneByGroup->setObjectName(QStringLiteral("RB_OneByGroup"));
        RB_OneByGroup->setGeometry(QRect(140, 20, 141, 16));
        PB_CalNumParticles = new QPushButton(DLG_MakeParticle);
        PB_CalNumParticles->setObjectName(QStringLiteral("PB_CalNumParticles"));
        PB_CalNumParticles->setGeometry(QRect(10, 530, 171, 25));
        QWidget::setTabOrder(LE_Name, TB_Method);
        QWidget::setTabOrder(TB_Method, LE_CUBE_DX);
        QWidget::setTabOrder(LE_CUBE_DX, LE_CUBE_DY);
        QWidget::setTabOrder(LE_CUBE_DY, LE_CUBE_DZ);
        QWidget::setTabOrder(LE_CUBE_DZ, LE_CUBE_LOC);
        QWidget::setTabOrder(LE_CUBE_LOC, LE_MIN_RADIUS);
        QWidget::setTabOrder(LE_MIN_RADIUS, LE_MAX_RADIUS);
        QWidget::setTabOrder(LE_MAX_RADIUS, LE_P_SPACING);
        QWidget::setTabOrder(LE_P_SPACING, LE_NUM_PARTICLE);
        QWidget::setTabOrder(LE_NUM_PARTICLE, CB_Type);
        QWidget::setTabOrder(CB_Type, LE_Youngs);
        QWidget::setTabOrder(LE_Youngs, LE_Density);
        QWidget::setTabOrder(LE_Density, LE_PoissonRatio);
        QWidget::setTabOrder(LE_PoissonRatio, LE_ShearModulus);
        QWidget::setTabOrder(LE_ShearModulus, GB_RealTime);
        QWidget::setTabOrder(GB_RealTime, RB_OneByOne);
        QWidget::setTabOrder(RB_OneByOne, RB_OneByGroup);
        QWidget::setTabOrder(RB_OneByGroup, LE_NumParclesPer);
        QWidget::setTabOrder(LE_NumParclesPer, PB_CalNumParticles);
        QWidget::setTabOrder(PB_CalNumParticles, PB_Ok);
        QWidget::setTabOrder(PB_Ok, PB_Cancle);
        QWidget::setTabOrder(PB_Cancle, LE_PLANE_DX);
        QWidget::setTabOrder(LE_PLANE_DX, LE_PLANE_DZ);
        QWidget::setTabOrder(LE_PLANE_DZ, LE_PLANE_DIR);
        QWidget::setTabOrder(LE_PLANE_DIR, LE_PLANE_LOC);
        QWidget::setTabOrder(LE_PLANE_LOC, LE_CircleDiameter);
        QWidget::setTabOrder(LE_CircleDiameter, LE_NumHeight);
        QWidget::setTabOrder(LE_NumHeight, LE_CircleLocation);
        QWidget::setTabOrder(LE_CircleLocation, LE_CircleDirection);

        retranslateUi(DLG_MakeParticle);

        TB_Method->setCurrentIndex(2);


        QMetaObject::connectSlotsByName(DLG_MakeParticle);
    } // setupUi

    void retranslateUi(QDialog *DLG_MakeParticle)
    {
        DLG_MakeParticle->setWindowTitle(QApplication::translate("DLG_MakeParticle", "Make particles", nullptr));
        L_CUBE_DX->setText(QApplication::translate("DLG_MakeParticle", "Dx", nullptr));
        L_CUBE_DY->setText(QApplication::translate("DLG_MakeParticle", "Dy", nullptr));
        L_CUBE_DZ->setText(QApplication::translate("DLG_MakeParticle", "Dz", nullptr));
        L_CUBE_LOC->setText(QApplication::translate("DLG_MakeParticle", "Start loc.", nullptr));
        TB_Method->setTabText(TB_Method->indexOf(T_Cube), QApplication::translate("DLG_MakeParticle", "Cube", nullptr));
        L_PLANE_LOC->setText(QApplication::translate("DLG_MakeParticle", "Start loc.", nullptr));
        L_PLANE_DX->setText(QApplication::translate("DLG_MakeParticle", "Dx", nullptr));
        L_PLANE_NY->setText(QApplication::translate("DLG_MakeParticle", "Ny", nullptr));
        L_PLANE_DZ->setText(QApplication::translate("DLG_MakeParticle", "Dz", nullptr));
        L_PLANE_DIR->setText(QApplication::translate("DLG_MakeParticle", "Dir.", nullptr));
        TB_Method->setTabText(TB_Method->indexOf(T_Plane), QApplication::translate("DLG_MakeParticle", "Plane", nullptr));
        L_CircleLocation->setText(QApplication::translate("DLG_MakeParticle", "Location", nullptr));
        L_CircleDirection->setText(QApplication::translate("DLG_MakeParticle", "Direction", nullptr));
        L_CircleDiameter->setText(QApplication::translate("DLG_MakeParticle", "Diameter", nullptr));
        L_NumParticle->setText(QApplication::translate("DLG_MakeParticle", "N.Height", nullptr));
        TB_Method->setTabText(TB_Method->indexOf(T_Circle), QApplication::translate("DLG_MakeParticle", "Circle", nullptr));
        groupBox->setTitle(QApplication::translate("DLG_MakeParticle", "Particle property", nullptr));
        L_MIN_RADIUS->setText(QApplication::translate("DLG_MakeParticle", "Min. radius", nullptr));
        L_MAX_RADIUS->setText(QApplication::translate("DLG_MakeParticle", "Max. radius", nullptr));
        L_P_SPACING->setText(QApplication::translate("DLG_MakeParticle", "Particle spacing", nullptr));
        L_NUM_PARTICLE->setText(QApplication::translate("DLG_MakeParticle", "Num. particle", nullptr));
        GB_MaterialProperty->setTitle(QApplication::translate("DLG_MakeParticle", "Material property", nullptr));
        L_Type->setText(QApplication::translate("DLG_MakeParticle", "Type", nullptr));
        L_YoungsModulus->setText(QApplication::translate("DLG_MakeParticle", "Youngs modulus", nullptr));
        L_Density->setText(QApplication::translate("DLG_MakeParticle", "Density", nullptr));
        L_PoissonRatio->setText(QApplication::translate("DLG_MakeParticle", "Poisson ratio", nullptr));
        L_ShearModulus->setText(QApplication::translate("DLG_MakeParticle", "Shear modulus", nullptr));
        PB_Ok->setText(QApplication::translate("DLG_MakeParticle", "Ok", nullptr));
        PB_Cancle->setText(QApplication::translate("DLG_MakeParticle", "Cancle", nullptr));
        GB_BASIC->setTitle(QApplication::translate("DLG_MakeParticle", "GroupBox", nullptr));
        L_Name->setText(QApplication::translate("DLG_MakeParticle", "Name", nullptr));
        GB_RealTime->setTitle(QApplication::translate("DLG_MakeParticle", "Real time creating", nullptr));
        L_NumParticlesPer->setText(QApplication::translate("DLG_MakeParticle", "The number of particles(per second)", nullptr));
        RB_OneByOne->setText(QApplication::translate("DLG_MakeParticle", "One by one drop", nullptr));
        RB_OneByGroup->setText(QApplication::translate("DLG_MakeParticle", "One by group drop", nullptr));
        PB_CalNumParticles->setText(QApplication::translate("DLG_MakeParticle", "Calculation Num. Particles", nullptr));
    } // retranslateUi

};

namespace Ui {
    class DLG_MakeParticle: public Ui_DLG_MakeParticle {};
} // namespace Ui

QT_END_NAMESPACE

#endif // MAKEPARTICLEBNUHVD_H
