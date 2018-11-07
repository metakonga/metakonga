/****************************************************************************
** Meta object code from reading C++ file 'xDynamicsSolver.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.7.1)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../../../../include/xDynamicsSolver.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'xDynamicsSolver.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.7.1. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
struct qt_meta_stringdata_xDynamicsSolver_t {
    QByteArrayData data[7];
    char stringdata0[84];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_xDynamicsSolver_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_xDynamicsSolver_t qt_meta_stringdata_xDynamicsSolver = {
    {
QT_MOC_LITERAL(0, 0, 15), // "xDynamicsSolver"
QT_MOC_LITERAL(1, 16, 14), // "finishedThread"
QT_MOC_LITERAL(2, 31, 0), // ""
QT_MOC_LITERAL(3, 32, 12), // "sendProgress"
QT_MOC_LITERAL(4, 45, 4), // "info"
QT_MOC_LITERAL(5, 50, 16), // "excuteMessageBox"
QT_MOC_LITERAL(6, 67, 16) // "setStopCondition"

    },
    "xDynamicsSolver\0finishedThread\0\0"
    "sendProgress\0info\0excuteMessageBox\0"
    "setStopCondition"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_xDynamicsSolver[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
       5,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       4,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    0,   39,    2, 0x06 /* Public */,
       3,    3,   40,    2, 0x06 /* Public */,
       3,    2,   47,    2, 0x26 /* Public | MethodCloned */,
       5,    0,   52,    2, 0x06 /* Public */,

 // slots: name, argc, parameters, tag, flags
       6,    0,   53,    2, 0x0a /* Public */,

 // signals: parameters
    QMetaType::Void,
    QMetaType::Void, QMetaType::Int, QMetaType::QString, QMetaType::QString,    2,    2,    4,
    QMetaType::Void, QMetaType::Int, QMetaType::QString,    2,    2,
    QMetaType::Void,

 // slots: parameters
    QMetaType::Void,

       0        // eod
};

void xDynamicsSolver::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        xDynamicsSolver *_t = static_cast<xDynamicsSolver *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->finishedThread(); break;
        case 1: _t->sendProgress((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< QString(*)>(_a[2])),(*reinterpret_cast< QString(*)>(_a[3]))); break;
        case 2: _t->sendProgress((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< QString(*)>(_a[2]))); break;
        case 3: _t->excuteMessageBox(); break;
        case 4: _t->setStopCondition(); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        void **func = reinterpret_cast<void **>(_a[1]);
        {
            typedef void (xDynamicsSolver::*_t)();
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&xDynamicsSolver::finishedThread)) {
                *result = 0;
                return;
            }
        }
        {
            typedef void (xDynamicsSolver::*_t)(int , QString , QString );
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&xDynamicsSolver::sendProgress)) {
                *result = 1;
                return;
            }
        }
        {
            typedef void (xDynamicsSolver::*_t)();
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&xDynamicsSolver::excuteMessageBox)) {
                *result = 3;
                return;
            }
        }
    }
}

const QMetaObject xDynamicsSolver::staticMetaObject = {
    { &QThread::staticMetaObject, qt_meta_stringdata_xDynamicsSolver.data,
      qt_meta_data_xDynamicsSolver,  qt_static_metacall, Q_NULLPTR, Q_NULLPTR}
};


const QMetaObject *xDynamicsSolver::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *xDynamicsSolver::qt_metacast(const char *_clname)
{
    if (!_clname) return Q_NULLPTR;
    if (!strcmp(_clname, qt_meta_stringdata_xDynamicsSolver.stringdata0))
        return static_cast<void*>(const_cast< xDynamicsSolver*>(this));
    return QThread::qt_metacast(_clname);
}

int xDynamicsSolver::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QThread::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 5)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 5;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 5)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 5;
    }
    return _id;
}

// SIGNAL 0
void xDynamicsSolver::finishedThread()
{
    QMetaObject::activate(this, &staticMetaObject, 0, Q_NULLPTR);
}

// SIGNAL 1
void xDynamicsSolver::sendProgress(int _t1, QString _t2, QString _t3)
{
    void *_a[] = { Q_NULLPTR, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)), const_cast<void*>(reinterpret_cast<const void*>(&_t3)) };
    QMetaObject::activate(this, &staticMetaObject, 1, _a);
}

// SIGNAL 3
void xDynamicsSolver::excuteMessageBox()
{
    QMetaObject::activate(this, &staticMetaObject, 3, Q_NULLPTR);
}
QT_END_MOC_NAMESPACE
