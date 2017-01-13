/********************************************************************************
** Form generated from reading UI file 'HMCM.ui'
**
** Created by: Qt User Interface Compiler version 5.7.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_HMCM_H
#define UI_HMCM_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QDialog>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_DLG_HMCM
{
public:
    QGroupBox *GBContactElements;
    QWidget *widget;
    QGridLayout *gridLayout;
    QLabel *LName;
    QVBoxLayout *verticalLayout;
    QLineEdit *LEName;
    QComboBox *CBFirstObject;
    QComboBox *CBSecondObject;
    QLabel *LFirstObject;
    QLabel *LSecondObject;
    QGroupBox *GBContactParameter;
    QCheckBox *CHBEableCohesion;
    QWidget *widget1;
    QGridLayout *gridLayout_2;
    QLabel *LRestitution;
    QVBoxLayout *verticalLayout_2;
    QLineEdit *LERestitution;
    QLineEdit *LEFriction;
    QLineEdit *LERollingFriction;
    QLabel *LFriction;
    QLabel *LRollingFriction;
    QWidget *widget2;
    QHBoxLayout *horizontalLayout;
    QPushButton *PBOk;
    QPushButton *PBCancle;

    void setupUi(QDialog *DLG_HMCM)
    {
        if (DLG_HMCM->objectName().isEmpty())
            DLG_HMCM->setObjectName(QStringLiteral("DLG_HMCM"));
        DLG_HMCM->resize(331, 332);
        DLG_HMCM->setModal(true);
        GBContactElements = new QGroupBox(DLG_HMCM);
        GBContactElements->setObjectName(QStringLiteral("GBContactElements"));
        GBContactElements->setGeometry(QRect(10, 10, 311, 111));
        widget = new QWidget(GBContactElements);
        widget->setObjectName(QStringLiteral("widget"));
        widget->setGeometry(QRect(11, 20, 291, 76));
        gridLayout = new QGridLayout(widget);
        gridLayout->setObjectName(QStringLiteral("gridLayout"));
        gridLayout->setContentsMargins(0, 0, 0, 0);
        LName = new QLabel(widget);
        LName->setObjectName(QStringLiteral("LName"));

        gridLayout->addWidget(LName, 0, 0, 1, 1);

        verticalLayout = new QVBoxLayout();
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        LEName = new QLineEdit(widget);
        LEName->setObjectName(QStringLiteral("LEName"));

        verticalLayout->addWidget(LEName);

        CBFirstObject = new QComboBox(widget);
        CBFirstObject->setObjectName(QStringLiteral("CBFirstObject"));

        verticalLayout->addWidget(CBFirstObject);

        CBSecondObject = new QComboBox(widget);
        CBSecondObject->setObjectName(QStringLiteral("CBSecondObject"));

        verticalLayout->addWidget(CBSecondObject);


        gridLayout->addLayout(verticalLayout, 0, 1, 3, 1);

        LFirstObject = new QLabel(widget);
        LFirstObject->setObjectName(QStringLiteral("LFirstObject"));

        gridLayout->addWidget(LFirstObject, 1, 0, 1, 1);

        LSecondObject = new QLabel(widget);
        LSecondObject->setObjectName(QStringLiteral("LSecondObject"));

        gridLayout->addWidget(LSecondObject, 2, 0, 1, 1);

        LName->raise();
        LEName->raise();
        GBContactParameter = new QGroupBox(DLG_HMCM);
        GBContactParameter->setObjectName(QStringLiteral("GBContactParameter"));
        GBContactParameter->setGeometry(QRect(10, 130, 311, 161));
        CHBEableCohesion = new QCheckBox(GBContactParameter);
        CHBEableCohesion->setObjectName(QStringLiteral("CHBEableCohesion"));
        CHBEableCohesion->setGeometry(QRect(11, 104, 121, 16));
        widget1 = new QWidget(GBContactParameter);
        widget1->setObjectName(QStringLiteral("widget1"));
        widget1->setGeometry(QRect(10, 20, 291, 76));
        gridLayout_2 = new QGridLayout(widget1);
        gridLayout_2->setObjectName(QStringLiteral("gridLayout_2"));
        gridLayout_2->setContentsMargins(0, 0, 0, 0);
        LRestitution = new QLabel(widget1);
        LRestitution->setObjectName(QStringLiteral("LRestitution"));

        gridLayout_2->addWidget(LRestitution, 0, 0, 1, 1);

        verticalLayout_2 = new QVBoxLayout();
        verticalLayout_2->setObjectName(QStringLiteral("verticalLayout_2"));
        LERestitution = new QLineEdit(widget1);
        LERestitution->setObjectName(QStringLiteral("LERestitution"));

        verticalLayout_2->addWidget(LERestitution);

        LEFriction = new QLineEdit(widget1);
        LEFriction->setObjectName(QStringLiteral("LEFriction"));

        verticalLayout_2->addWidget(LEFriction);

        LERollingFriction = new QLineEdit(widget1);
        LERollingFriction->setObjectName(QStringLiteral("LERollingFriction"));

        verticalLayout_2->addWidget(LERollingFriction);


        gridLayout_2->addLayout(verticalLayout_2, 0, 1, 3, 1);

        LFriction = new QLabel(widget1);
        LFriction->setObjectName(QStringLiteral("LFriction"));

        gridLayout_2->addWidget(LFriction, 1, 0, 1, 1);

        LRollingFriction = new QLabel(widget1);
        LRollingFriction->setObjectName(QStringLiteral("LRollingFriction"));

        gridLayout_2->addWidget(LRollingFriction, 2, 0, 1, 1);

        widget2 = new QWidget(DLG_HMCM);
        widget2->setObjectName(QStringLiteral("widget2"));
        widget2->setGeometry(QRect(160, 300, 158, 25));
        horizontalLayout = new QHBoxLayout(widget2);
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        horizontalLayout->setContentsMargins(0, 0, 0, 0);
        PBOk = new QPushButton(widget2);
        PBOk->setObjectName(QStringLiteral("PBOk"));

        horizontalLayout->addWidget(PBOk);

        PBCancle = new QPushButton(widget2);
        PBCancle->setObjectName(QStringLiteral("PBCancle"));

        horizontalLayout->addWidget(PBCancle);


        retranslateUi(DLG_HMCM);

        QMetaObject::connectSlotsByName(DLG_HMCM);
    } // setupUi

    void retranslateUi(QDialog *DLG_HMCM)
    {
        DLG_HMCM->setWindowTitle(QApplication::translate("DLG_HMCM", "Hertz-Mindlin Contact Model", 0));
        GBContactElements->setTitle(QApplication::translate("DLG_HMCM", "Contact elements", 0));
        LName->setText(QApplication::translate("DLG_HMCM", "Name", 0));
        LFirstObject->setText(QApplication::translate("DLG_HMCM", "First object", 0));
        LSecondObject->setText(QApplication::translate("DLG_HMCM", "Second object", 0));
        GBContactParameter->setTitle(QApplication::translate("DLG_HMCM", "Contact parameter", 0));
        CHBEableCohesion->setText(QApplication::translate("DLG_HMCM", "Enable cohesion", 0));
        LRestitution->setText(QApplication::translate("DLG_HMCM", "Restitution", 0));
        LFriction->setText(QApplication::translate("DLG_HMCM", "Friction", 0));
        LRollingFriction->setText(QApplication::translate("DLG_HMCM", "Rolling friction", 0));
        PBOk->setText(QApplication::translate("DLG_HMCM", "Ok", 0));
        PBCancle->setText(QApplication::translate("DLG_HMCM", "Cancle", 0));
    } // retranslateUi

};

namespace Ui {
    class DLG_HMCM: public Ui_DLG_HMCM {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_HMCM_H
