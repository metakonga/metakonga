/********************************************************************************
** Form generated from reading UI file 'makePlane.ui'
**
** Created by: Qt User Interface Compiler version 5.7.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAKEPLANE_H
#define UI_MAKEPLANE_H

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
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_DLG_MAKEPLANE
{
public:
    QWidget *layoutWidget;
    QHBoxLayout *horizontalLayout;
    QPushButton *PB_Ok;
    QPushButton *PB_Cancle;
    QGroupBox *GB_Information;
    QWidget *widget;
    QGridLayout *gridLayout_2;
    QLabel *L_Name;
    QLineEdit *LE_Name;
    QLabel *L_Point_a;
    QLineEdit *LE_Point_a;
    QLabel *L_Point_b;
    QLineEdit *LE_Point_b;
    QLabel *L_Point_c;
    QLineEdit *LE_Point_c;
    QLabel *L_Point_d;
    QLineEdit *LE_Point_d;
    QGroupBox *GB_MaterialProperty;
    QWidget *layoutWidget_3;
    QGridLayout *gridLayout;
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

    void setupUi(QDialog *DLG_MAKEPLANE)
    {
        if (DLG_MAKEPLANE->objectName().isEmpty())
            DLG_MAKEPLANE->setObjectName(QStringLiteral("DLG_MAKEPLANE"));
        DLG_MAKEPLANE->resize(323, 385);
        layoutWidget = new QWidget(DLG_MAKEPLANE);
        layoutWidget->setObjectName(QStringLiteral("layoutWidget"));
        layoutWidget->setGeometry(QRect(137, 350, 171, 25));
        horizontalLayout = new QHBoxLayout(layoutWidget);
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        horizontalLayout->setContentsMargins(0, 0, 0, 0);
        PB_Ok = new QPushButton(layoutWidget);
        PB_Ok->setObjectName(QStringLiteral("PB_Ok"));

        horizontalLayout->addWidget(PB_Ok);

        PB_Cancle = new QPushButton(layoutWidget);
        PB_Cancle->setObjectName(QStringLiteral("PB_Cancle"));

        horizontalLayout->addWidget(PB_Cancle);

        GB_Information = new QGroupBox(DLG_MAKEPLANE);
        GB_Information->setObjectName(QStringLiteral("GB_Information"));
        GB_Information->setGeometry(QRect(10, 10, 301, 161));
        widget = new QWidget(GB_Information);
        widget->setObjectName(QStringLiteral("widget"));
        widget->setGeometry(QRect(10, 21, 281, 126));
        gridLayout_2 = new QGridLayout(widget);
        gridLayout_2->setObjectName(QStringLiteral("gridLayout_2"));
        gridLayout_2->setContentsMargins(0, 0, 0, 0);
        L_Name = new QLabel(widget);
        L_Name->setObjectName(QStringLiteral("L_Name"));

        gridLayout_2->addWidget(L_Name, 0, 0, 1, 1);

        LE_Name = new QLineEdit(widget);
        LE_Name->setObjectName(QStringLiteral("LE_Name"));

        gridLayout_2->addWidget(LE_Name, 0, 1, 1, 1);

        L_Point_a = new QLabel(widget);
        L_Point_a->setObjectName(QStringLiteral("L_Point_a"));

        gridLayout_2->addWidget(L_Point_a, 1, 0, 1, 1);

        LE_Point_a = new QLineEdit(widget);
        LE_Point_a->setObjectName(QStringLiteral("LE_Point_a"));

        gridLayout_2->addWidget(LE_Point_a, 1, 1, 1, 1);

        L_Point_b = new QLabel(widget);
        L_Point_b->setObjectName(QStringLiteral("L_Point_b"));

        gridLayout_2->addWidget(L_Point_b, 2, 0, 1, 1);

        LE_Point_b = new QLineEdit(widget);
        LE_Point_b->setObjectName(QStringLiteral("LE_Point_b"));

        gridLayout_2->addWidget(LE_Point_b, 2, 1, 1, 1);

        L_Point_c = new QLabel(widget);
        L_Point_c->setObjectName(QStringLiteral("L_Point_c"));

        gridLayout_2->addWidget(L_Point_c, 3, 0, 1, 1);

        LE_Point_c = new QLineEdit(widget);
        LE_Point_c->setObjectName(QStringLiteral("LE_Point_c"));

        gridLayout_2->addWidget(LE_Point_c, 3, 1, 1, 1);

        L_Point_d = new QLabel(widget);
        L_Point_d->setObjectName(QStringLiteral("L_Point_d"));

        gridLayout_2->addWidget(L_Point_d, 4, 0, 1, 1);

        LE_Point_d = new QLineEdit(widget);
        LE_Point_d->setObjectName(QStringLiteral("LE_Point_d"));

        gridLayout_2->addWidget(LE_Point_d, 4, 1, 1, 1);

        GB_MaterialProperty = new QGroupBox(DLG_MAKEPLANE);
        GB_MaterialProperty->setObjectName(QStringLiteral("GB_MaterialProperty"));
        GB_MaterialProperty->setGeometry(QRect(10, 180, 301, 161));
        layoutWidget_3 = new QWidget(GB_MaterialProperty);
        layoutWidget_3->setObjectName(QStringLiteral("layoutWidget_3"));
        layoutWidget_3->setGeometry(QRect(10, 21, 281, 126));
        gridLayout = new QGridLayout(layoutWidget_3);
        gridLayout->setObjectName(QStringLiteral("gridLayout"));
        gridLayout->setContentsMargins(0, 0, 0, 0);
        L_Type = new QLabel(layoutWidget_3);
        L_Type->setObjectName(QStringLiteral("L_Type"));

        gridLayout->addWidget(L_Type, 0, 0, 1, 1);

        CB_Type = new QComboBox(layoutWidget_3);
        CB_Type->setObjectName(QStringLiteral("CB_Type"));

        gridLayout->addWidget(CB_Type, 0, 1, 1, 1);

        L_YoungsModulus = new QLabel(layoutWidget_3);
        L_YoungsModulus->setObjectName(QStringLiteral("L_YoungsModulus"));

        gridLayout->addWidget(L_YoungsModulus, 1, 0, 1, 1);

        LE_Youngs = new QLineEdit(layoutWidget_3);
        LE_Youngs->setObjectName(QStringLiteral("LE_Youngs"));

        gridLayout->addWidget(LE_Youngs, 1, 1, 1, 1);

        L_Density = new QLabel(layoutWidget_3);
        L_Density->setObjectName(QStringLiteral("L_Density"));

        gridLayout->addWidget(L_Density, 2, 0, 1, 1);

        LE_Density = new QLineEdit(layoutWidget_3);
        LE_Density->setObjectName(QStringLiteral("LE_Density"));

        gridLayout->addWidget(LE_Density, 2, 1, 1, 1);

        L_PoissonRatio = new QLabel(layoutWidget_3);
        L_PoissonRatio->setObjectName(QStringLiteral("L_PoissonRatio"));

        gridLayout->addWidget(L_PoissonRatio, 3, 0, 1, 1);

        LE_PoissonRatio = new QLineEdit(layoutWidget_3);
        LE_PoissonRatio->setObjectName(QStringLiteral("LE_PoissonRatio"));

        gridLayout->addWidget(LE_PoissonRatio, 3, 1, 1, 1);

        L_ShearModulus = new QLabel(layoutWidget_3);
        L_ShearModulus->setObjectName(QStringLiteral("L_ShearModulus"));

        gridLayout->addWidget(L_ShearModulus, 4, 0, 1, 1);

        LE_ShearModulus = new QLineEdit(layoutWidget_3);
        LE_ShearModulus->setObjectName(QStringLiteral("LE_ShearModulus"));

        gridLayout->addWidget(LE_ShearModulus, 4, 1, 1, 1);

        QWidget::setTabOrder(LE_Name, LE_Point_a);
        QWidget::setTabOrder(LE_Point_a, LE_Point_b);
        QWidget::setTabOrder(LE_Point_b, LE_Point_c);
        QWidget::setTabOrder(LE_Point_c, LE_Point_d);
        QWidget::setTabOrder(LE_Point_d, CB_Type);
        QWidget::setTabOrder(CB_Type, LE_Youngs);
        QWidget::setTabOrder(LE_Youngs, LE_Density);
        QWidget::setTabOrder(LE_Density, LE_PoissonRatio);
        QWidget::setTabOrder(LE_PoissonRatio, LE_ShearModulus);
        QWidget::setTabOrder(LE_ShearModulus, PB_Ok);
        QWidget::setTabOrder(PB_Ok, PB_Cancle);

        retranslateUi(DLG_MAKEPLANE);

        QMetaObject::connectSlotsByName(DLG_MAKEPLANE);
    } // setupUi

    void retranslateUi(QDialog *DLG_MAKEPLANE)
    {
        DLG_MAKEPLANE->setWindowTitle(QApplication::translate("DLG_MAKEPLANE", "Make plane object", Q_NULLPTR));
        PB_Ok->setText(QApplication::translate("DLG_MAKEPLANE", "Ok", Q_NULLPTR));
        PB_Cancle->setText(QApplication::translate("DLG_MAKEPLANE", "Cancle", Q_NULLPTR));
        GB_Information->setTitle(QApplication::translate("DLG_MAKEPLANE", "Information", Q_NULLPTR));
        L_Name->setText(QApplication::translate("DLG_MAKEPLANE", "Name", Q_NULLPTR));
        L_Point_a->setText(QApplication::translate("DLG_MAKEPLANE", "Point a", Q_NULLPTR));
        L_Point_b->setText(QApplication::translate("DLG_MAKEPLANE", "Point b", Q_NULLPTR));
        L_Point_c->setText(QApplication::translate("DLG_MAKEPLANE", "Point c", Q_NULLPTR));
        L_Point_d->setText(QApplication::translate("DLG_MAKEPLANE", "Point d", Q_NULLPTR));
        GB_MaterialProperty->setTitle(QApplication::translate("DLG_MAKEPLANE", "Material property", Q_NULLPTR));
        L_Type->setText(QApplication::translate("DLG_MAKEPLANE", "Type", Q_NULLPTR));
        L_YoungsModulus->setText(QApplication::translate("DLG_MAKEPLANE", "Youngs modulus", Q_NULLPTR));
        L_Density->setText(QApplication::translate("DLG_MAKEPLANE", "Density", Q_NULLPTR));
        L_PoissonRatio->setText(QApplication::translate("DLG_MAKEPLANE", "Poisson ratio", Q_NULLPTR));
        L_ShearModulus->setText(QApplication::translate("DLG_MAKEPLANE", "Shear modulus", Q_NULLPTR));
    } // retranslateUi

};

namespace Ui {
    class DLG_MAKEPLANE: public Ui_DLG_MAKEPLANE {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAKEPLANE_H
