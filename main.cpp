#include "mainwindow.h"
#include <QApplication>

/*#include<cstdlib>
#include<iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace std;
using namespace cv;*/
#include <unistd.h>


int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    //WJN--------------------
    uid_t user = 0;
        user = geteuid();
        if(user != 0)
        {
            /*printf("\n");
            printf("Please run this application with 'sudo -E ./GxAcquireContinuous' or"
                                  " Start with root !\n");
            printf("\n");*/
            return 0;
        }
    //WJN--------------------

    MainWindow w;
    w.show();

    //cv::Mat img; //定义一个Mat变量
    //img = cv::imread("/home/jerry/1.jpg"); //读取图片
    //cv::imshow("test",img);        //显示图片
    //cv::waitKey(0);



    return a.exec();
}


