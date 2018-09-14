/********************************************************************************
** Form generated from reading UI file 'bodyinfo.ui'
**
** Created by: Qt User Interface Compiler version 5.10.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_BODYINFO_H
#define UI_BODYINFO_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
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

class Ui_DLG_BODYINFO
{
public:
    QGroupBox *GB_Property;
    QWidget *widget;
    QHBoxLayout *horizontalLayout;
    QLabel *L_Mass;
    QLineEdit *LE_Mass;
    QWidget *widget1;
    QHBoxLayout *horizontalLayout_2;
    QVBoxLayout *verticalLayout;
    QLabel *L_Ixx;
    QLabel *L_Iyy;
    QLabel *L_Izz;
    QVBoxLayout *verticalLayout_2;
    QLineEdit *LE_Ixx;
    QLineEdit *LE_Iyy;
    QLineEdit *LE_Izz;
    QVBoxLayout *verticalLayout_3;
    QLabel *L_Ixy;
    QLabel *L_Iyz;
    QLabel *L_Izx;
    QVBoxLayout *verticalLayout_4;
    QLineEdit *LE_Ixy;
    QLineEdit *LE_Iyz;
    QLineEdit *LE_Izx;
    QComboBox *CB_Material_Input_Type;
    QComboBox *CB_Material_Type;
    QPushButton *PB_Ok;
    QPushButton *PB_Cancle;
    QWidget *widget2;
    QVBoxLayout *verticalLayout_5;
    QLabel *L_Material_Input_Type;
    QLabel *L_Material_Type;
    QWidget *widget3;
    QHBoxLayout *horizontalLayout_3;
    QLabel *L_Position;
    QLineEdit *LE_Position;
    QWidget *widget4;
    QHBoxLayout *horizontalLayout_4;
    QLabel *L_Volume;
    QLineEdit *LE_Volume;

    void setupUi(QDialog *DLG_BODYINFO)
    {
        if (DLG_BODYINFO->objectName().isEmpty())
            DLG_BODYINFO->setObjectName(QStringLiteral("DLG_BODYINFO"));
        DLG_BODYINFO->resize(353, 302);
        GB_Property = new QGroupBox(DLG_BODYINFO);
        GB_Property->setObjectName(QStringLiteral("GB_Property"));
        GB_Property->setGeometry(QRect(10, 100, 331, 135));
        widget = new QWidget(GB_Property);
        widget->setObjectName(QStringLiteral("widget"));
        widget->setGeometry(QRect(10, 20, 311, 22));
        horizontalLayout = new QHBoxLayout(widget);
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        horizontalLayout->setContentsMargins(0, 0, 0, 0);
        L_Mass = new QLabel(widget);
        L_Mass->setObjectName(QStringLiteral("L_Mass"));

        horizontalLayout->addWidget(L_Mass);

        LE_Mass = new QLineEdit(widget);
        LE_Mass->setObjectName(QStringLiteral("LE_Mass"));

        horizontalLayout->addWidget(LE_Mass);

        widget1 = new QWidget(GB_Property);
        widget1->setObjectName(QStringLiteral("widget1"));
        widget1->setGeometry(QRect(10, 50, 311, 76));
        horizontalLayout_2 = new QHBoxLayout(widget1);
        horizontalLayout_2->setObjectName(QStringLiteral("horizontalLayout_2"));
        horizontalLayout_2->setContentsMargins(0, 0, 0, 0);
        verticalLayout = new QVBoxLayout();
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        L_Ixx = new QLabel(widget1);
        L_Ixx->setObjectName(QStringLiteral("L_Ixx"));

        verticalLayout->addWidget(L_Ixx);

        L_Iyy = new QLabel(widget1);
        L_Iyy->setObjectName(QStringLiteral("L_Iyy"));

        verticalLayout->addWidget(L_Iyy);

        L_Izz = new QLabel(widget1);
        L_Izz->setObjectName(QStringLiteral("L_Izz"));

        verticalLayout->addWidget(L_Izz);


        horizontalLayout_2->addLayout(verticalLayout);

        verticalLayout_2 = new QVBoxLayout();
        verticalLayout_2->setObjectName(QStringLiteral("verticalLayout_2"));
        LE_Ixx = new QLineEdit(widget1);
        LE_Ixx->setObjectName(QStringLiteral("LE_Ixx"));

        verticalLayout_2->addWidget(LE_Ixx);

        LE_Iyy = new QLineEdit(widget1);
        LE_Iyy->setObjectName(QStringLiteral("LE_Iyy"));

        verticalLayout_2->addWidget(LE_Iyy);

        LE_Izz = new QLineEdit(widget1);
        LE_Izz->setObjectName(QStringLiteral("LE_Izz"));

        verticalLayout_2->addWidget(LE_Izz);


        horizontalLayout_2->addLayout(verticalLayout_2);

        verticalLayout_3 = new QVBoxLayout();
        verticalLayout_3->setObjectName(QStringLiteral("verticalLayout_3"));
        L_Ixy = new QLabel(widget1);
        L_Ixy->setObjectName(QStringLiteral("L_Ixy"));

        verticalLayout_3->addWidget(L_Ixy);

        L_Iyz = new QLabel(widget1);
        L_Iyz->setObjectName(QStringLiteral("L_Iyz"));

        verticalLayout_3->addWidget(L_Iyz);

        L_Izx = new QLabel(widget1);
        L_Izx->setObjectName(QStringLiteral("L_Izx"));

        verticalLayout_3->addWidget(L_Izx);


        horizontalLayout_2->addLayout(verticalLayout_3);

        verticalLayout_4 = new QVBoxLayout();
        verticalLayout_4->setObjectName(QStringLiteral("verticalLayout_4"));
        LE_Ixy = new QLineEdit(widget1);
        LE_Ixy->setObjectName(QStringLiteral("LE_Ixy"));

        verticalLayout_4->addWidget(LE_Ixy);

        LE_Iyz = new QLineEdit(widget1);
        LE_Iyz->setObjectName(QStringLiteral("LE_Iyz"));

        verticalLayout_4->addWidget(LE_Iyz);

        LE_Izx = new QLineEdit(widget1);
        LE_Izx->setObjectName(QStringLiteral("LE_Izx"));

        verticalLayout_4->addWidget(LE_Izx);


        horizontalLayout_2->addLayout(verticalLayout_4);

        CB_Material_Input_Type = new QComboBox(DLG_BODYINFO);
        CB_Material_Input_Type->addItem(QString());
        CB_Material_Input_Type->addItem(QString());
        CB_Material_Input_Type->setObjectName(QStringLiteral("CB_Material_Input_Type"));
        CB_Material_Input_Type->setGeometry(QRect(170, 10, 171, 22));
        CB_Material_Type = new QComboBox(DLG_BODYINFO);
        CB_Material_Type->setObjectName(QStringLiteral("CB_Material_Type"));
        CB_Material_Type->setGeometry(QRect(170, 40, 171, 22));
        PB_Ok = new QPushButton(DLG_BODYINFO);
        PB_Ok->setObjectName(QStringLiteral("PB_Ok"));
        PB_Ok->setGeometry(QRect(180, 270, 75, 23));
        PB_Cancle = new QPushButton(DLG_BODYINFO);
        PB_Cancle->setObjectName(QStringLiteral("PB_Cancle"));
        PB_Cancle->setGeometry(QRect(267, 270, 75, 23));
        widget2 = new QWidget(DLG_BODYINFO);
        widget2->setObjectName(QStringLiteral("widget2"));
        widget2->setGeometry(QRect(20, 11, 111, 51));
        verticalLayout_5 = new QVBoxLayout(widget2);
        verticalLayout_5->setObjectName(QStringLiteral("verticalLayout_5"));
        verticalLayout_5->setContentsMargins(0, 0, 0, 0);
        L_Material_Input_Type = new QLabel(widget2);
        L_Material_Input_Type->setObjectName(QStringLiteral("L_Material_Input_Type"));

        verticalLayout_5->addWidget(L_Material_Input_Type);

        L_Material_Type = new QLabel(widget2);
        L_Material_Type->setObjectName(QStringLiteral("L_Material_Type"));

        verticalLayout_5->addWidget(L_Material_Type);

        widget3 = new QWidget(DLG_BODYINFO);
        widget3->setObjectName(QStringLiteral("widget3"));
        widget3->setGeometry(QRect(20, 70, 321, 22));
        horizontalLayout_3 = new QHBoxLayout(widget3);
        horizontalLayout_3->setObjectName(QStringLiteral("horizontalLayout_3"));
        horizontalLayout_3->setContentsMargins(0, 0, 0, 0);
        L_Position = new QLabel(widget3);
        L_Position->setObjectName(QStringLiteral("L_Position"));

        horizontalLayout_3->addWidget(L_Position);

        LE_Position = new QLineEdit(widget3);
        LE_Position->setObjectName(QStringLiteral("LE_Position"));

        horizontalLayout_3->addWidget(LE_Position);

        widget4 = new QWidget(DLG_BODYINFO);
        widget4->setObjectName(QStringLiteral("widget4"));
        widget4->setGeometry(QRect(20, 240, 321, 22));
        horizontalLayout_4 = new QHBoxLayout(widget4);
        horizontalLayout_4->setObjectName(QStringLiteral("horizontalLayout_4"));
        horizontalLayout_4->setContentsMargins(0, 0, 0, 0);
        L_Volume = new QLabel(widget4);
        L_Volume->setObjectName(QStringLiteral("L_Volume"));

        horizontalLayout_4->addWidget(L_Volume);

        LE_Volume = new QLineEdit(widget4);
        LE_Volume->setObjectName(QStringLiteral("LE_Volume"));

        horizontalLayout_4->addWidget(LE_Volume);

        CB_Material_Input_Type->raise();
        GB_Property->raise();
        L_Material_Input_Type->raise();
        L_Material_Type->raise();
        CB_Material_Type->raise();
        L_Volume->raise();
        LE_Volume->raise();
        L_Position->raise();
        LE_Position->raise();
        PB_Ok->raise();
        PB_Cancle->raise();
        QWidget::setTabOrder(CB_Material_Input_Type, CB_Material_Type);
        QWidget::setTabOrder(CB_Material_Type, LE_Position);
        QWidget::setTabOrder(LE_Position, LE_Mass);
        QWidget::setTabOrder(LE_Mass, LE_Ixx);
        QWidget::setTabOrder(LE_Ixx, LE_Iyy);
        QWidget::setTabOrder(LE_Iyy, LE_Izz);
        QWidget::setTabOrder(LE_Izz, LE_Ixy);
        QWidget::setTabOrder(LE_Ixy, LE_Iyz);
        QWidget::setTabOrder(LE_Iyz, LE_Izx);
        QWidget::setTabOrder(LE_Izx, LE_Volume);
        QWidget::setTabOrder(LE_Volume, PB_Ok);
        QWidget::setTabOrder(PB_Ok, PB_Cancle);

        retranslateUi(DLG_BODYINFO);

        QMetaObject::connectSlotsByName(DLG_BODYINFO);
    } // setupUi

    void retranslateUi(QDialog *DLG_BODYINFO)
    {
        DLG_BODYINFO->setWindowTitle(QApplication::translate("DLG_BODYINFO", "Dialog", nullptr));
        GB_Property->setTitle(QApplication::translate("DLG_BODYINFO", "Property", nullptr));
        L_Mass->setText(QApplication::translate("DLG_BODYINFO", "Mass", nullptr));
        L_Ixx->setText(QApplication::translate("DLG_BODYINFO", "Ixx", nullptr));
        L_Iyy->setText(QApplication::translate("DLG_BODYINFO", "Iyy", nullptr));
        L_Izz->setText(QApplication::translate("DLG_BODYINFO", "Izz", nullptr));
        L_Ixy->setText(QApplication::translate("DLG_BODYINFO", "Ixy", nullptr));
        L_Iyz->setText(QApplication::translate("DLG_BODYINFO", "Iyz", nullptr));
        L_Izx->setText(QApplication::translate("DLG_BODYINFO", "Izx", nullptr));
        CB_Material_Input_Type->setItemText(0, QApplication::translate("DLG_BODYINFO", "Library", nullptr));
        CB_Material_Input_Type->setItemText(1, QApplication::translate("DLG_BODYINFO", "User input", nullptr));

        PB_Ok->setText(QApplication::translate("DLG_BODYINFO", "Ok", nullptr));
        PB_Cancle->setText(QApplication::translate("DLG_BODYINFO", "Cancle", nullptr));
        L_Material_Input_Type->setText(QApplication::translate("DLG_BODYINFO", "Material Input Type", nullptr));
        L_Material_Type->setText(QApplication::translate("DLG_BODYINFO", "Material Type", nullptr));
        L_Position->setText(QApplication::translate("DLG_BODYINFO", "Position", nullptr));
        L_Volume->setText(QApplication::translate("DLG_BODYINFO", "Volume", nullptr));
    } // retranslateUi

};

namespace Ui {
    class DLG_BODYINFO: public Ui_DLG_BODYINFO {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_BODYINFO_H
