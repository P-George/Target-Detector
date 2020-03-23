#-------------------------------------------------
#
# Project created by QtCreator 2018-12-25T10:27:01
#
#-------------------------------------------------

#config for dlib---
INCLUDEPATH += /home/jerry/Dlib-19.17/dlib-19.17 \
LIBS += -L/home/jerry/Dlib-19.17/dlib-19.17 \
CONFIG += link_pkgconfig
PKGCONFIG += x11
SOURCES += /home/jerry/Dlib-19.17/dlib-19.17/dlib/all/source.cpp

#LIBS += -L/opt/X11/lib \
LIBS += -lX11
#-----------------------

#config for Xenomai---------------
#QMAKE_CC += $(shell $(XENO_CONFIG) --cc)
#QMAKE_CFLAGS += $(shell $(XENO_CONFIG) --alchemy --cflags) \
QMAKE_LFLAGS += $(shell $(XENO_CONFIG) --alchemy --ldflags) \

INCLUDEPATH += /usr/xenomai/include/cobalt\
/usr/xenomai/include\
/usr/xenomai/include/alchemy\
LIBS += -lalchemy -lcopperplate\
/usr/xenomai/lib \
-lcobalt -lmodechk -lpthread -lrt \
QMAKE_LFLAGS += -Wl,--no-as-needed \
-Wl,@/usr/xenomai/lib/modechk.wrappers -lalchemy -lcopperplate \
/usr/xenomai/lib/xenomai/bootstrap.o \
-Wl,--wrap=main \
-Wl,--dynamic-list=/usr/xenomai/lib/dynlist.ld \
-L/usr/xenomai/lib \
-lcobalt -lmodechk -lpthread -lrt \
#INCLUDEPATH += -I/usr/xenomai/include/cobalt -I/usr/xenomai/include -D_GNU_SOURCE -D_REENTRANT -D__COBALT__ -I/usr/xenomai/include/alchemy \
#LIBS += -Wl,--no-as-needed -Wl,@/usr/xenomai/lib/modechk.wrappers -lalchemy -lcopperplate /usr/xenomai/lib/xenomai/bootstrap.o -Wl,--wrap=main -Wl,--dynamic-list=/usr/xenomai/lib/dynlist.ld -L/usr/xenomai/lib -lcobalt -lmodechk -lpthread -lrt \
#------------------------------

#config for opencv---
INCLUDEPATH += /usr/local/include \
    /usr/local/include/opencv \
    /usr/local/include/opencv2 \

LIBS += /usr/local/lib/libopencv_calib3d.so \
 /usr/local/lib/libopencv_core.so \
 /usr/local/lib/libopencv_features2d.so \
 /usr/local/lib/libopencv_flann.so \
 /usr/local/lib/libopencv_highgui.so \
/usr/local/lib/libopencv_imgcodecs.so \
 /usr/local/lib/libopencv_imgproc.so \
/usr/local/lib/libopencv_ml.so \
/usr/local/lib/libopencv_objdetect.so \
 /usr/local/lib/libopencv_photo.so \
/usr/local/lib/libopencv_shape.so \
/usr/local/lib/libopencv_stitching.so \
 /usr/local/lib/libopencv_superres.so \
/usr/local/lib/libopencv_videoio.so \
/usr/local/lib/libopencv_video.so \
/usr/local/lib/libopencv_videostab.so \
#-----------------------

#config for GX---------
LIBS += -lgxiapi -ldximageproc -lpthread \
          -L$(GENICAM_ROOT_V2_3)/bin/Linux64_x64 \
          -L$(GENICAM_ROOT_V2_3)/bin/Linux64_x64/GenApi/Generic \
          -L/usr/local/lib \
         -lGCBase_gcc40_v2_3 -lGenApi_gcc40_v2_3 -llog4cpp_gcc40_v2_3 -lLog_gcc40_v2_3 -lMathParser_gcc40_v2_3\

INCLUDEPATH += $(GENICAM_ROOT_V2_3)/library/CPP/include \
INCLUDEPATH += $(GENICAM_ROOT_V2_3)/../../sdk/include \
#-----------------------------



QT       += core gui
#QT       += core gui sql charts

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets printsupport

TARGET = Target_Detector
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp \
    qcustomplot.cpp

HEADERS  += mainwindow.h\
    GxIAPI.h \
    DxImageProc.h \
    qcustomplot.h

FORMS    += mainwindow.ui

CONFIG   += c++11
