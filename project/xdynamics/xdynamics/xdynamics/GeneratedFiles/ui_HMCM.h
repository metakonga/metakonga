/********************************************************************************
** Form generated from reading UI file 'HMCM.ui'
**
** Created by: Qt User Interface Compiler version 5.7.1
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
    QWidget *layoutWidget;
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
    QWidget *layoutWidget1;
    QGridLayout *gridLayout_2;
    QLabel *LRestitution;
    QVBoxLayout *verticalLayout_2;
    QLineEdit *LERestitution;
    QLineEdit *LEFriction;
    QLineEdit *LERollingFriction;
    QLabel *LFriction;
    QLabel *LRollingFriction;
    QLabel *LCohesion;
    QLineEdit *LECohesion;
    QCheckBox *CHBUSESTIFFNESSRATIO;
    QLabel *LRatio;
    QLineEdit *LERatio;
    QWidget *layoutWidget2;
    QHBoxLayout *horizontalLayout;
    QPushButton *PBOk;
    QPushButton *PBCancle;

    void setupUi(QDialog *DLG_HMCM)
    {
        if (DLG_HMCM->objectName().isEmpty())
            DLG_HMCM->setObjectName(QStringLiteral("DLG_HMCM"));
        DLG_HMCM->resize(332, 386);
        DLG_HMCM->setModal(true);
        GBContactElements = new QGroupBox(DLG_HMCM);
        GBContactElements->setObjectName(QStringLiteral("GBContactElements"));
        GBContactElements->setGeometry(QRect(10, 10, 311, 111));
        layoutWidget = new QWidget(GBContactElements);
        layoutWidget->setObjectName(QStringLiteral("layoutWidget"));
        layoutWidget->setGeometry(QRect(11, 20, 291, 76));
        gridLayout = new QGridLayout(layoutWidget);
        gridLayout->setObjectName(QStringLiteral("gridLayout"));
        gridLayout->setContentsMargins(0, 0, 0, 0);
        LName = new QLabel(layoutWidget);
        LName->setObjectName(QStringLiteral("LName"));

        gridLayout->addWidget(LName, 0, 0, 1, 1);

        verticalLayout = new QVBoxLayout();
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        LEName = new QLineEdit(layoutWidget);
        LEName->setObjectName(QStringLiteral("LEName"));

        verticalLayout->addWidget(LEName);

        CBFirstObject = new QComboBox(layoutWidget);
        CBFirstObject->setObjectName(QStringLiteral("CBFirstObject"));

        verticalLayout->addWidget(CBFirstObject);

        CBSecondObject = new QComboBox(layoutWidget);
        CBSecondObject->setObjectName(QStringLiteral("CBSecondObject"));

        verticalLayout->addWidget(CBSecondObject);


        gridLayout->addLayout(verticalLayout, 0, 1, 3, 1);

        LFirstObject = new QLabel(layoutWidget);
        LFirstObject->setObjectName(QStringLiteral("LFirstObject"));

        gridLayout->addWidget(LFirstObject, 1, 0, 1, 1);

        LSecondObject = new QLabel(layoutWidget);
        LSecondObject->setObjectName(QStringLiteral("LSecondObject"));

        gridLayout->addWidget(LSecondObject, 2, 0, 1, 1);

        GBContactParameter = new QGroupBox(DLG_HMCM);
        GBContactParameter->setObjectName(QStringLiteral("GBContactParameter"));
        GBContactParameter->setGeometry(QRect(10, 130, 311, 211));
        CHBEableCohesion = new QCheckBox(GBContactParameter);
        CHBEableCohesion->setObjectName(QStringLiteral("CHBEableCohesion"));
        CHBEableCohesion->setGeometry(QRect(11, 104, 121, 16));
        layoutWidget1 = new QWidget(GBContactParameter);
        layoutWidget1->setObjectName(QStringLiteral("layoutWidget1"));
        layoutWidget1->setGeometry(QRect(10, 20, 291, 76));
        gridLayout_2 = new QGridLayout(layoutWidget1);
        gridLayout_2->setObjectName(QStringLiteral("gridLayout_2"));
        gridLayout_2->setContentsMargins(0, 0, 0, 0);
        LRestitution = new QLabel(layoutWidget1);
        LRestitution->setObjectName(QStringLiteral("LRestitution"));

        gridLayout_2->addWidget(LRestitution, 0, 0, 1, 1);

        verticalLayout_2 = new QVBoxLayout();
        verticalLayout_2->setObjectName(QStringLiteral("verticalLayout_2"));
        LERestitution = new QLineEdit(layoutWidget1);
        LERestitution->setObjectName(QStringLiteral("LERestitution"));

        verticalLayout_2->addWidget(LERestitution);

        LEFriction = new QLineEdit(layoutWidget1);
        LEFriction->setObjectName(QStringLiteral("LEFriction"));

        verticalLayout_2->addWidget(LEFriction);

        LERollingFriction = new QLineEdit(layoutWidget1);
        LERollingFriction->setObjectName(QStringLiteral("LERollingFriction"));

        verticalLayout_2->addWidget(LERollingFriction);


        gridLayout_2->addLayout(verticalLayout_2, 0, 1, 3, 1);

        LFriction = new QLabel(layoutWidget1);
        LFriction->setObjectName(QStringLiteral("LFriction"));

        gridLayout_2->addWidget(LFriction, 1, 0, 1, 1);

        LRollingFriction = new QLabel(layoutWidget1);
        LRollingFriction->setObjectName(QStringLiteral("LRollingFriction"));

        gridLayout_2->addWidget(LRollingFriction, 2, 0, 1, 1);

        LCohesion = new QLabel(GBContactParameter);
        LCohesion->setObjectName(QStringLiteral("LCohesion"));
        LCohesion->setGeometry(QRect(10, 132, 71, 16));
        LECohesion = new QLineEdit(GBContactParameter);
        LECohesion->setObjectName(QStringLiteral("LECohesion"));
        LECohesion->setGeometry(QRect(95, 130, 202, 20));
        CHBUSESTIFFNESSRATIO = new QCheckBox(GBContactParameter);
        CHBUSESTIFFNESSRATIO->setObjectName(QStringLiteral("CHBUSESTIFFNESSRATIO"));
        CHBUSESTIFFNESSRATIO->setGeometry(QRect(10, 160, 131, 16));
        LRatio = new QLabel(GBContactParameter);
        LRatio->setObjectName(QStringLiteral("LRatio"));
        LRatio->setGeometry(QRect(10, 185, 71, 16));
        LERatio = new QLineEdit(GBContactParameter);
        LERatio->setObjectName(QStringLiteral("LERatio"));
        LERatio->setGeometry(QRect(94, 182, 202, 20));
        layoutWidget2 = new QWidget(DLG_HMCM);
        layoutWidget2->setObjectName(QStringLiteral("layoutWidget2"));
        layoutWidget2->setGeometry(QRect(160, 350, 158, 25));
        horizontalLayout = new QHBoxLayout(layoutWidget2);
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        horizontalLayout->setContentsMargins(0, 0, 0, 0);
        PBOk = new QPushButton(layoutWidget2);
        PBOk->setObjectName(QStringLiteral("PBOk"));

        horizontalLayout->addWidget(PBOk);

        PBCancle = new QPushButton(layoutWidget2);
        PBCancle->setObjectName(QStringLiteral("PBCancle"));

        horizontalLayout->addWidget(PBCancle);

        QWidget::setTabOrder(LEName, CBFirstObject);
        QWidget::setTabOrder(CBFirstObject, CBSecondObject);
        QWidget::setTabOrder(CBSecondObject, LERestitution);
        QWidget::setTabOrder(LERestitution, LEFriction);
        QWidget::setTabOrder(LEFriction, LERollingFriction);
        QWidget::setTabOrder(LERollingFriction, CHBEableCohesion);
        QWidget::setTabOrder(CHBEableCohesion, LECohesion);
        QWidget::setTabOrder(LECohesion, CHBUSESTIFFNESSRATIO);
        QWidget::setTabOrder(CHBUSESTIFFNESSRATIO, LERatio);
        QWidget::setTabOrder(LERatio, PBOk);
        QWidget::setTabOrder(PBOk, PBCancle);

        retranslateUi(DLG_HMCM);

        QMetaObject::connectSlotsByName(DLG_HMCM);
    } // setupUi

    void retranslateUi(QDialog *DLG_HMCM)
    {
        DLG_HMCM->setWindowTitle(QApplication::translate("DLG_HMCM", "Contact Definition Dialog", Q_NULLPTR));
        GBContactElements->setTitle(QApplication::translate("DLG_HMCM", "Contact elements", Q_NULLPTR));
        LName->setText(QApplication::translate("DLG_HMCM", "Name", Q_NULLPTR));
        LFirstObject->setText(QApplication::translate("DLG_HMCM", "First object", Q_NULLPTR));
        LSecondObject->setText(QApplication::translate("DLG_HMCM", "Second object", Q_NULLPTR));
        GBContactParameter->setTitle(QApplication::translate("DLG_HMCM", "Contact parameter", Q_NULLPTR));
        CHBEableCohesion->setText(QApplication::translate("DLG_HMCM", "Enable cohesion", Q_NULLPTR));
        LRestitution->setText(QApplication::translate("DLG_HMCM", "Restitution", Q_NULLPTR));
        LFriction->setText(QApplication::translate("DLG_HMCM", "Friction", Q_NULLPTR));
        LRollingFriction->setText(QApplication::translate("DLG_HMCM", "Rolling friction", Q_NULLPTR));
        LCohesion->setText(QApplication::translate("DLG_HMCM", "Cohesion", Q_NULLPTR));
        CHBUSESTIFFNESSRATIO->setText(QApplication::translate("DLG_HMCM", "Use stiffness ratio", Q_NULLPTR));
        LRatio->setText(QApplication::translate("DLG_HMCM", "Ratio", Q_NULLPTR));
        PBOk->setText(QApplication::translate("DLG_HMCM", "Ok", Q_NULLPTR));
        PBCancle->setText(QApplication::translate("DLG_HMCM", "Cancle", Q_NULLPTR));
    } // retranslateUi

};

namespace Ui {
    class DLG_HMCM: public Ui_DLG_HMCM {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_HMCM_H
