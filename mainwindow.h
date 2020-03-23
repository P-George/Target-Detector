#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <pthread.h>
#include "qcustomplot.h"

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();
    //void *ProcGetImage(void* param);
    void InitPlot(QCustomPlot *mycustomplot);
    void initCheckBoxGroup();

private slots:
    void on_ButtonStart_clicked();

    void on_ButtonStop_clicked();

    //void *ProcGetImage(void* param);

    void on_ButtonGain_clicked();

    void on_ButtonWhiteBalance_clicked();

    void on_ButtonCapture_clicked();

    void on_ButtonReplot_clicked();

    void on_ButtonReset_clicked();

    void on_ButtonChooseBG_clicked();

    void on_checkBD_stateChanged(int arg1);

    void on_ButtonSave_clicked();

    void on_checkMoveDetect_stateChanged(int arg1);

private:
    Ui::MainWindow *ui;
};

#endif // MAINWINDOW_H
