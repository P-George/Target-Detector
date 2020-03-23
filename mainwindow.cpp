#include "mainwindow.h"
#include "ui_mainwindow.h"

//Dlib----------------------
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/opencv.h>
#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>

//#include "opaque_types.h"
//#include <dlib/python.h>
#include <dlib/matrix.h>
#include <dlib/image_transforms.h>
//#include "indexing.h"
//#include <pybind11/stl_bind.h>

//using namespace dlib;

/*/ ----------------------------------------------------------------------------------------
template <long num_filters, typename SUBNET> using con5d = dlib::con<num_filters,5,5,3,3,SUBNET>;
template <long num_filters, typename SUBNET> using con3  = dlib::con<num_filters,3,3,1,1,SUBNET>;
template <typename SUBNET> using downsampler  = dlib::relu<dlib::affine<con5d<10, dlib::relu<dlib::affine<con5d<10,SUBNET>>>>>>;
template <typename SUBNET> using rcon3  = dlib::relu<dlib::affine<con3<10,SUBNET>>>;
template <typename SUBNET> using rcon4  = dlib::relu<dlib::affine<con3<10,SUBNET>>>;
using net_type  = dlib::loss_mmod<dlib::con<1,9,9,1,1,rcon4<rcon3<downsampler<dlib::input_rgb_image_pyramid<dlib::pyramid_down<6>>>>>>>;
// ----------------------------------------------------------------------------------------trainer19:55ms with iterations 500 with good precision limit*/
// ----------------------------------------------------------------------------------------
template <long num_filters, typename SUBNET> using con5d = dlib::con<num_filters,5,5,3,3,SUBNET>;
template <long num_filters, typename SUBNET> using con3  = dlib::con<num_filters,3,3,1,1,SUBNET>;
template <typename SUBNET> using downsampler  = dlib::relu<dlib::affine<con5d<10, dlib::relu<dlib::affine<con5d<10,SUBNET>>>>>>;
template <typename SUBNET> using rcon3  = dlib::relu<dlib::affine<con3<10,SUBNET>>>;
template <typename SUBNET> using rcon4  = dlib::relu<dlib::affine<con3<10,SUBNET>>>;
using net_type  = dlib::loss_mmod<dlib::con<1,9,9,1,1,rcon4<rcon3<downsampler<dlib::input_rgb_image_pyramid<dlib::pyramid_down<6>>>>>>>;
// ----------------------------------------------------------------------------------------trainer4:60*/
//----------------------------

//Xenomai---------------
#include <stdio.h>
#include <signal.h>
#include <unistd.h>
#include <sys/mman.h>
#include <alchemy/task.h>
#include <alchemy/timer.h>
#include <math.h>
#include <time.h>


#define CLOCK_RES 1e-9 //Clock resolution is 1 ns by default
#define LOOP_PERIOD 1e7 //Expressed in ticks
//RTIME period = 1000000000;
RT_TASK GetimageRT_task;
//RT_TASK CaptureImageRT_task;

//---------------------------

//WJN-----------------------------
#include "GxIAPI.h"
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>

#define MEMORY_ALLOT_ERROR -1

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "DxImageProc.h"

#include "opencv2/opencv.hpp"
#include <iostream>

#include <vector>
//#include <iterator>
using namespace std;

using namespace cv;
Mat m_image;
Mat move_result;
//Mat move_temp;
bool is_implemented = false;
int64_t m_pixel_color = 0;
///< Bayer格?式o?
char* m_rgb_image = NULL;

GX_DEV_HANDLE g_device = NULL;              ///< 设备句柄
GX_FRAME_DATA g_frame_data = { 0 };         ///< 采集图像参数
pthread_t g_acquire_thread = 0;             ///< 采集线程ID
bool g_get_image = false;                   ///< 采集线程是否结束的标志：true 运行；false 退出

//API接口函数返回值
GX_STATUS status = GX_STATUS_SUCCESS;

uint32_t device_num = 0;
uint32_t ret = 0;
GX_OPEN_PARAM open_param;

Ui::MainWindow *ui1;
double text_imagegettime,text_imageprocesstime,text_totaltime;
int Target_LocationX=0,Target_LocationY=0,Target_LocationW=0,Target_LocationH=0;
int Box_Times=4;
//获取图像大小并申请图像数据空间
int PreForImage();

//释放资源
int UnPreForImage();

//采集线程函数
void *ProcGetImage_BD(void* param);
void ProcGetImage_RT_BD(void *arg);
Mat MoveDetect_BD(Mat temp, Mat frame);

void *ProcGetImage_FD(void* param);
void ProcGetImage_RT_FD(void *arg);
Mat MoveDetect_FD(Mat temp, Mat frame);

void *ProcGetImage_TriFD(void* param);
void ProcGetImage_RT_TriFD(void *arg);
Mat MoveDetect_TriFD(Mat &diff_thresh1,Mat framepre,Mat frame);

void *ProcGetImage_CNN(void* param);
void ProcGetImage_RT_CNN(void *arg);

//获取错误信息描述
void GetErrorString(GX_STATUS error_status);

void MainWindow::InitPlot(QCustomPlot *mycustomplot){
    mycustomplot->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom);      //可拖拽+可滚轮缩放
    //mycustomplot->setInteractions(QCP::iRangeZoom);

    mycustomplot->addGraph();
    mycustomplot->addGraph();
    mycustomplot->legend->setVisible(true);                                //右上角指示曲线的缩略框
    mycustomplot->graph(0)->setPen(QPen(Qt::red));
    mycustomplot->graph(1)->setPen(QPen(Qt::blue));
    mycustomplot->graph(0)->setName("非实时");
    mycustomplot->graph(1)->setName("实时");

    mycustomplot->xAxis->setLabel("Frame");
    mycustomplot->yAxis->setLabel("Time");

    mycustomplot->xAxis->setRange(0,2000);
    mycustomplot->yAxis->setRange(0,80);
}
int FrameCount=0,FrameCountRT=0;
double Averagetimetotal=0,AveragetimetotalRT=0;
vector<double> Timeforstatistics,RTTimeforstatistics;
double Maxtime=0,MaxtimeRT=0;

bool flag_choosebackground=false;
Mat move_temp;

void MainWindow::initCheckBoxGroup()
{
   QButtonGroup* pButtonGroup = new QButtonGroup(this);
   pButtonGroup->addButton(ui->checkBD,1);
   pButtonGroup->addButton(ui->checkFD, 2);
   pButtonGroup->addButton(ui->checkTriFD, 3);
   pButtonGroup->addButton(ui->checkCNN, 4);
   ui->checkBD->setChecked(true);
   ui->ButtonSave->setEnabled(false);
}

QString Location_tobesaved="";
//WJN------------------------------

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    ui->ImageView->setScaledContents(true);
    ui->ImageCapture->setScaledContents(true);
    ui1=ui;

    //ui->widget->replot();
    InitPlot(ui->mycustomplot);

    //initiative checkboxgroup
    initCheckBoxGroup();
}

MainWindow::~MainWindow()
{
    delete ui;
}


//WJN--------------------------
//-------------------------------------------------
/**
\brief 获取图像大小并申请图像数据空间
\return void
*/
//-------------------------------------------------
int PreForImage()
{
    GX_STATUS status = GX_STATUS_SUCCESS;
    int64_t payload_size = 0;

    status = GXGetInt(g_device, GX_INT_PAYLOAD_SIZE, &payload_size);
    if(status != GX_STATUS_SUCCESS)
    {
        GetErrorString(status);
        return status;
    }

    g_frame_data.pImgBuf = malloc(payload_size);
    if(g_frame_data.pImgBuf == NULL)
    {
        printf("<Failed to allot memory>\n");
        return MEMORY_ALLOT_ERROR;
    }

    return 0;
}
//----------------------------------------------------------------------------------
/**
\brief  获取错误信息描述
\param  emErrorStatus  错误码

\return void
*/
//----------------------------------------------------------------------------------
void GetErrorString(GX_STATUS error_status)
{
    char *error_info = NULL;
    size_t size = 0;
    GX_STATUS status = GX_STATUS_SUCCESS;

    // 获取错误描述信息长度
    status = GXGetLastError(&error_status, NULL, &size);
    if(status != GX_STATUS_SUCCESS)
    {
           GetErrorString(status);
       return;
    }

    error_info = new char[size];
    if (error_info == NULL)
    {
        printf("<Failed to allocate memory>\n");
        return ;
    }

    // 获取错误信息描述
    status = GXGetLastError(&error_status, error_info, &size);
    if (status != GX_STATUS_SUCCESS)
    {
        printf("<GXGetLastError call fail>\n");
    }
    else
    {
        printf("%s\n", (char*)error_info);
    }

    // 释放资源
    if (error_info != NULL)
    {
        delete []error_info;
        error_info = NULL;
    }
}
//-------------------------------------------------
/**
\brief 释放资源
\return void
*/
//-------------------------------------------------
int UnPreForImage()
{
    GX_STATUS status = GX_STATUS_SUCCESS;
    uint32_t ret = 0;

    //发送停采命令
    /*status = GXSendCommand(g_device, GX_COMMAND_ACQUISITION_STOP);
    if(status != GX_STATUS_SUCCESS)
    {
        GetErrorString(status);
        return status;
    }*/

    g_get_image = false;

    //release thread resources
    if(ui1->checkRT->isChecked())
    {
        //xenomai-----------
        munlockall();
        //rt_task_join(&GetimageRT_task);
        rt_task_delete(&GetimageRT_task);
        //-----------------------
    }else{
         ret = pthread_join(g_acquire_thread,NULL);
         if(ret != 0)
         {
             printf("<Failed to release resources>\n");
             return ret;
         }
    }

    //发送停采命令
    status = GXSendCommand(g_device, GX_COMMAND_ACQUISITION_STOP);
    if(status != GX_STATUS_SUCCESS)
    {
        GetErrorString(status);
        return status;
    }

    //释放buffer
    if(g_frame_data.pImgBuf != NULL)
    {
        free(g_frame_data.pImgBuf);
        g_frame_data.pImgBuf = NULL;
    }

    return 0;
}

//-------------------------------------------------
/**
\brief 采集线程函数
\param pParam 线程传入参数
\return void*
*/
//-------------------------------------------------
void *ProcGetImage_BD(void* pParam)
{
    GX_STATUS status = GX_STATUS_SUCCESS;

    //get frame storage cover state
    //bool frame_storage_flag;
    //status=GXGetBool(g_device,GX_BOOL_FRAMESTORE_COVER_ACTIVE, &frame_storage_flag);

    //接收线程启动标志
    g_get_image = true;

    //发送开采命令
    status = GXSendCommand(g_device, GX_COMMAND_ACQUISITION_START);
    if(status != GX_STATUS_SUCCESS)
    {
        GetErrorString(status);
    }

    //int flag_count=0;
    Mat move_frame;
    //Mat move_result;
    //Mat move_temp;
    while(g_get_image)
    {
        //RTIME tstart=rt_timer_read();
        clock_t t_start,t_finish,t_imageget,t_imageprocess;
        t_start=clock();
        if(g_frame_data.pImgBuf == NULL)
        {
            continue;
        }

        //set latency
        //usleep(10000);

        status = GXGetImage(g_device, &g_frame_data, 100);
        if(status == GX_STATUS_SUCCESS)
        {
            if(g_frame_data.nStatus == 0)
            {
                //printf("<Successful acquisition : Width: %d Height: %d >\n", g_frame_data.nWidth, g_frame_data.nHeight);
                if(is_implemented)
                {
                    DxRaw8toRGB24(g_frame_data.pImgBuf,m_rgb_image,g_frame_data.nWidth,
                    g_frame_data.nHeight,RAW2RGB_NEIGHBOUR,DX_PIXEL_COLOR_FILTER(BAYERRG),false);
                    memcpy(m_image.data,m_rgb_image,g_frame_data.nHeight*g_frame_data.nWidth*3);
                }else{
                    memcpy(m_image.data,g_frame_data.pImgBuf,g_frame_data.nHeight*g_frame_data.nWidth);
                }
                t_imageget=clock();
                double imagegettime=(double)((t_imageget-t_start)/1000);
                cout<<"Image get time: "<<imagegettime<<" ms"<<endl;
                //QImage m_q_image1=QImage((const unsigned char*)m_image.data, m_image.cols, m_image.rows, QImage::Format_RGB888);
                //ui1->ImageCapture->setPixmap(QPixmap::fromImage(m_image));
                //ui1->ImageCapture->show();

                //flag_count++;
                if(ui1->checkMoveDetect->isChecked())
                {
                    move_frame=m_image;
                    if(flag_choosebackground){
                        move_result=MoveDetect_BD(move_temp,move_frame);
                    }else{
                        move_result=MoveDetect_BD(move_frame,move_frame);
                    }
                    //move_temp=move_frame.clone();
                }else{
                    move_result=m_image;
                }
                t_imageprocess=clock();
                double imageprocesstime=(double)((t_imageprocess-t_imageget)/1000);
                cout<<"Image process time: "<<imageprocesstime<<" ms"<<endl;
                //imshow("move_result",move_result);

                QImage m_q_image=QImage((const unsigned char*)move_result.data, move_result.cols, move_result.rows,move_result.step, QImage::Format_RGB888);
                ui1->ImageView->setPixmap(QPixmap::fromImage(m_q_image));
                ui1->ImageView->show();

                //t_imageview=clock();
               // cout<<"Image view time: "<<(double)((t_imageview-t_imageprocess)/1000)<<" ms"<<endl;

                //cout<<"Loop time: "<<(rt_timer_read()-tstart)/1000000<<" ms"<<endl;
                t_finish=clock();
                double t_total=(double)((t_finish-t_start)/1000);
                //cout<<CLOCKS_PER_SEC<<endl;
                cout<<"Total time: "<<t_total<<" ms"<<endl;
                cout<<move_result.cols<<"  "<<move_result.rows<<endl;
                text_imagegettime=imagegettime;
                text_imageprocesstime=imageprocesstime;
                text_totaltime=t_total;

                //if(flag_choosebackground)
                //{
                ui1->mycustomplot->graph(0)->addData(FrameCount,t_total);
                Timeforstatistics.push_back(t_total);
                Averagetimetotal+=t_total;
                //if(FrameCount%500==0){
                    //ui1->mycustomplot->replot();
                //}
                FrameCount++;
                //}
            }
        }
    }
}
//xenomai----------version
void ProcGetImage_RT_BD(void *arg)
{
    GX_STATUS status = GX_STATUS_SUCCESS;

    //get frame storage cover state
    //bool frame_storage_flag;
    //status=GXGetBool(g_device,GX_BOOL_FRAMESTORE_COVER_ACTIVE, &frame_storage_flag);

    //接收线程启动标志
    g_get_image = true;

    //发送开采命令
    status = GXSendCommand(g_device, GX_COMMAND_ACQUISITION_START);
    if(status != GX_STATUS_SUCCESS)
    {
        GetErrorString(status);
    }

    //int flag_count=0;
    Mat move_frame;
    //Mat move_result;
    //Mat move_temp;
    while(g_get_image)
    {
        //RTIME tstart=rt_timer_read();
        clock_t t_start,t_finish,t_imageget,t_imageprocess;
        t_start=clock();
        if(g_frame_data.pImgBuf == NULL)
        {
            continue;
        }

        //set latency
        //usleep(10000);

        status = GXGetImage(g_device, &g_frame_data, 100);
        if(status == GX_STATUS_SUCCESS)
        {
            if(g_frame_data.nStatus == 0)
            {
                //printf("<Successful acquisition : Width: %d Height: %d >\n", g_frame_data.nWidth, g_frame_data.nHeight);
                if(is_implemented)
                {
                    DxRaw8toRGB24(g_frame_data.pImgBuf,m_rgb_image,g_frame_data.nWidth,
                    g_frame_data.nHeight,RAW2RGB_NEIGHBOUR,DX_PIXEL_COLOR_FILTER(BAYERRG),false);
                    memcpy(m_image.data,m_rgb_image,g_frame_data.nHeight*g_frame_data.nWidth*3);
                }else{
                    memcpy(m_image.data,g_frame_data.pImgBuf,g_frame_data.nHeight*g_frame_data.nWidth);
                }
                t_imageget=clock();
                double imagegettime=(double)((t_imageget-t_start)/1000);
                cout<<"Image get time: "<<imagegettime<<" ms"<<endl;
                //QImage m_q_image1=QImage((const unsigned char*)m_image.data, m_image.cols, m_image.rows, QImage::Format_RGB888);
                //ui1->ImageCapture->setPixmap(QPixmap::fromImage(m_image));
                //ui1->ImageCapture->show();

                //flag_count++;
                if(ui1->checkMoveDetect->isChecked())
                {
                    move_frame=m_image;
                    if(flag_choosebackground){
                        move_result=MoveDetect_BD(move_temp,move_frame);
                    }else{
                        move_result=MoveDetect_BD(move_frame,move_frame);
                    }
                    //move_temp=move_frame.clone();
                }else{
                    move_result=m_image;
                }
                t_imageprocess=clock();
                double imageprocesstime=(double)((t_imageprocess-t_imageget)/1000);
                cout<<"Image process time: "<<imageprocesstime<<" ms"<<endl;
                //imshow("move_result",move_result);

                QImage m_q_image=QImage((const unsigned char*)move_result.data, move_result.cols, move_result.rows,move_result.step, QImage::Format_RGB888);
                ui1->ImageView->setPixmap(QPixmap::fromImage(m_q_image));
                ui1->ImageView->show();

                //t_imageview=clock();
               // cout<<"Image view time: "<<(double)((t_imageview-t_imageprocess)/1000)<<" ms"<<endl;

                //cout<<"Loop time: "<<(rt_timer_read()-tstart)/1000000<<" ms"<<endl;
                t_finish=clock();
                double t_total=(double)((t_finish-t_start)/1000);
                //cout<<CLOCKS_PER_SEC<<endl;
                cout<<"Total time: "<<t_total<<" ms"<<endl;
                cout<<move_result.cols<<"  "<<move_result.rows<<endl;
                text_imagegettime=imagegettime;
                text_imageprocesstime=imageprocesstime;
                text_totaltime=t_total;

                //if(flag_choosebackground)
                //{
                ui1->mycustomplot->graph(1)->addData(FrameCountRT,t_total);
                RTTimeforstatistics.push_back(t_total);
                AveragetimetotalRT+=t_total;
                //if(FrameCountRT%100==0){
                    //ui1->mycustomplot->replot();
                //}
                FrameCountRT++;
                //}
            }
        }
    }
}

Mat MoveDetect_BD(Mat temp, Mat frame)
{
    //Mat result = frame.clone();
    Mat result = frame;// make an optimization
    //1.将background和frame转为灰度图

    int Box_X=0,Box_Y=0,Box_W=0,Box_H=0;
    if(ui1->checkSA->isChecked()&&(Target_LocationX!=0||Target_LocationY!=0)){
        Box_X=(Target_LocationX+Target_LocationW/2)-(Target_LocationW*Box_Times/2);
        Box_Y=(Target_LocationY+Target_LocationH/2)-(Target_LocationH*Box_Times/2);
        Box_W=Target_LocationW*Box_Times;
        Box_H=Target_LocationH*Box_Times;

        if(Box_X<0){
            Box_X=0;
        }
        if(Box_Y<0){
            Box_Y=0;
        }
        if(Box_X+Box_W>frame.cols){
            Box_W=frame.cols-Box_X;
        }
        if(Box_Y+Box_H>frame.rows){
            Box_H=frame.rows-Box_Y;
        }

        Rect Box_Region(Box_X,Box_Y,Box_W,Box_H);
        temp=temp(Box_Region);
        frame=frame(Box_Region);
    }

    Mat gray1, gray2;
    cvtColor(temp, gray1, CV_BGR2GRAY);
    cvtColor(frame, gray2, CV_BGR2GRAY);
    //2.将background和frame做差
    Mat diff;
    absdiff(gray1, gray2, diff);

    //3.对差值图diff_thresh进行阈值化处理
    Mat diff_thresh;
    threshold(diff, diff_thresh, 50, 255, CV_THRESH_BINARY);
    //imshow("diff_thresh", diff_thresh);

    //GaussianBlur(diff_thresh, diff_thresh, Size(3, 3), 0, 0);

    //4.腐蚀
    Mat kernel_erode = getStructuringElement(MORPH_RECT, Size(3, 3));
    Mat kernel_dilate = getStructuringElement(MORPH_RECT, Size(18, 18));
    erode(diff_thresh, diff_thresh, kernel_erode);
    //imshow("erode", diff_thresh);
    //5.膨胀
    dilate(diff_thresh, diff_thresh, kernel_dilate);
    //imshow("dilate", diff_thresh);
    //6.查找轮廓并绘制轮廓
    vector<vector<Point> > contours;
    findContours(diff_thresh, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    //drawContours(result, contours, -1, Scalar(0, 0, 255), 2);//在result上绘制轮廓
    //7.查找正外接矩形
    /*Target_LocationX=0;
    Target_LocationY=0;
    Target_LocationW=0;
    Target_LocationH=0;*/
    vector<Rect> boundRect(contours.size());
    for (int i = 0; i < contours.size(); i++)
    {
        boundRect[i] = boundingRect(contours[i]);

        if(ui1->checkSA->isChecked()&&(Target_LocationX!=0||Target_LocationY!=0)){
            boundRect[i].x+=Box_X;
            boundRect[i].y+=Box_Y;
        }

        rectangle(result, boundRect[i], Scalar(0, 255, 0), 2);//在result上绘制正外接矩形
        if(i==0){
            Target_LocationX=boundRect[i].x;
            Target_LocationY=boundRect[i].y;
            Target_LocationW=boundRect[i].width;
            Target_LocationH=boundRect[i].height;
        }
    }

    if(contours.size()==0){
        Target_LocationX=0;
        Target_LocationY=0;
        Target_LocationW=0;
        Target_LocationH=0;
    }

    Location_tobesaved+=QString::number(Target_LocationX,10);
    Location_tobesaved+=" ";
    Location_tobesaved+=QString::number(Target_LocationY,10);
    Location_tobesaved+=" ";
    QTime current_time =QTime::currentTime();
    Location_tobesaved+=current_time.toString("hh:mm:ss.zzz");
    Location_tobesaved+="\n";

    //cv::namedWindow("diff", CV_WINDOW_NORMAL);
    //cv::imshow("diff", diff);
    //cv::waitKey(1);
    //QImage m_q_image=QImage((const unsigned char*)gray1.data, gray1.cols, gray1.rows, QImage::Format_RGB888);
    //ui1->ImageCapture->setPixmap(QPixmap::fromImage(m_q_image));
    //ui1->ImageCapture->show();
    //cout<<"done!"<<endl;
    return result;//返回result
}

void *ProcGetImage_FD(void* pParam)
{
    GX_STATUS status = GX_STATUS_SUCCESS;

    //get frame storage cover state
    //bool frame_storage_flag;
    //status=GXGetBool(g_device,GX_BOOL_FRAMESTORE_COVER_ACTIVE, &frame_storage_flag);

    //接收线程启动标志
    g_get_image = true;

    //发送开采命令
    status = GXSendCommand(g_device, GX_COMMAND_ACQUISITION_START);
    if(status != GX_STATUS_SUCCESS)
    {
        GetErrorString(status);
    }

    int flag_count=0;
    Mat move_frame;
    //Mat move_result;
    //Mat move_temp;
    while(g_get_image)
    {
        //RTIME tstart=rt_timer_read();
        clock_t t_start,t_finish,t_imageget,t_imageprocess;
        t_start=clock();
        if(g_frame_data.pImgBuf == NULL)
        {
            continue;
        }

        //set latency
        //usleep(10000);

        status = GXGetImage(g_device, &g_frame_data, 100);
        if(status == GX_STATUS_SUCCESS)
        {
            if(g_frame_data.nStatus == 0)
            {
                //printf("<Successful acquisition : Width: %d Height: %d >\n", g_frame_data.nWidth, g_frame_data.nHeight);
                if(is_implemented)
                {
                    DxRaw8toRGB24(g_frame_data.pImgBuf,m_rgb_image,g_frame_data.nWidth,
                    g_frame_data.nHeight,RAW2RGB_NEIGHBOUR,DX_PIXEL_COLOR_FILTER(BAYERRG),false);
                    memcpy(m_image.data,m_rgb_image,g_frame_data.nHeight*g_frame_data.nWidth*3);
                }else{
                    memcpy(m_image.data,g_frame_data.pImgBuf,g_frame_data.nHeight*g_frame_data.nWidth);
                }
                t_imageget=clock();
                double imagegettime=(double)((t_imageget-t_start)/1000);
                cout<<"Image get time: "<<imagegettime<<" ms"<<endl;
                //QImage m_q_image1=QImage((const unsigned char*)m_image.data, m_image.cols, m_image.rows, QImage::Format_RGB888);
                //ui1->ImageCapture->setPixmap(QPixmap::fromImage(m_image));
                //ui1->ImageCapture->show();

                flag_count++;
                if(ui1->checkMoveDetect->isChecked())
                {
                    move_frame=m_image;
                    if(flag_count>1){
                        move_result=MoveDetect_FD(move_temp,move_frame);
                    }else{
                        move_result=MoveDetect_FD(move_frame,move_frame);
                    }
                    move_temp=move_frame.clone();
                }else{
                    move_result=m_image;
                }
                t_imageprocess=clock();
                double imageprocesstime=(double)((t_imageprocess-t_imageget)/1000);
                cout<<"Image process time: "<<imageprocesstime<<" ms"<<endl;
                //imshow("move_result",move_result);

                QImage m_q_image=QImage((const unsigned char*)move_result.data, move_result.cols, move_result.rows,move_result.step, QImage::Format_RGB888);
                ui1->ImageView->setPixmap(QPixmap::fromImage(m_q_image));
                ui1->ImageView->show();

                //t_imageview=clock();
               // cout<<"Image view time: "<<(double)((t_imageview-t_imageprocess)/1000)<<" ms"<<endl;

                //cout<<"Loop time: "<<(rt_timer_read()-tstart)/1000000<<" ms"<<endl;
                t_finish=clock();
                double t_total=(double)((t_finish-t_start)/1000);
                //cout<<CLOCKS_PER_SEC<<endl;
                cout<<"Total time: "<<t_total<<" ms"<<endl;
                cout<<move_result.cols<<"  "<<move_result.rows<<endl;
                text_imagegettime=imagegettime;
                text_imageprocesstime=imageprocesstime;
                text_totaltime=t_total;

                //if(flag_choosebackground)
                //{
                ui1->mycustomplot->graph(0)->addData(FrameCount,t_total);
                Timeforstatistics.push_back(t_total);
                Averagetimetotal+=t_total;
                //if(FrameCount%500==0){
                    //ui1->mycustomplot->replot();
                //}
                FrameCount++;
                //}
            }
        }
    }
}
//xenomai----------version
void ProcGetImage_RT_FD(void *arg)
{
    GX_STATUS status = GX_STATUS_SUCCESS;

    //get frame storage cover state
    //bool frame_storage_flag;
    //status=GXGetBool(g_device,GX_BOOL_FRAMESTORE_COVER_ACTIVE, &frame_storage_flag);

    //接收线程启动标志
    g_get_image = true;

    //发送开采命令
    status = GXSendCommand(g_device, GX_COMMAND_ACQUISITION_START);
    if(status != GX_STATUS_SUCCESS)
    {
        GetErrorString(status);
    }

    int flag_count=0;
    Mat move_frame;
    //Mat move_result;
    //Mat move_temp;
    while(g_get_image)
    {
        //RTIME tstart=rt_timer_read();
        clock_t t_start,t_finish,t_imageget,t_imageprocess;
        t_start=clock();
        if(g_frame_data.pImgBuf == NULL)
        {
            continue;
        }

        //set latency
        //usleep(10000);

        status = GXGetImage(g_device, &g_frame_data, 100);
        if(status == GX_STATUS_SUCCESS)
        {
            if(g_frame_data.nStatus == 0)
            {
                //printf("<Successful acquisition : Width: %d Height: %d >\n", g_frame_data.nWidth, g_frame_data.nHeight);
                if(is_implemented)
                {
                    DxRaw8toRGB24(g_frame_data.pImgBuf,m_rgb_image,g_frame_data.nWidth,
                    g_frame_data.nHeight,RAW2RGB_NEIGHBOUR,DX_PIXEL_COLOR_FILTER(BAYERRG),false);
                    memcpy(m_image.data,m_rgb_image,g_frame_data.nHeight*g_frame_data.nWidth*3);
                }else{
                    memcpy(m_image.data,g_frame_data.pImgBuf,g_frame_data.nHeight*g_frame_data.nWidth);
                }
                t_imageget=clock();
                double imagegettime=(double)((t_imageget-t_start)/1000);
                cout<<"Image get time: "<<imagegettime<<" ms"<<endl;
                //QImage m_q_image1=QImage((const unsigned char*)m_image.data, m_image.cols, m_image.rows, QImage::Format_RGB888);
                //ui1->ImageCapture->setPixmap(QPixmap::fromImage(m_image));
                //ui1->ImageCapture->show();

                flag_count++;
                if(ui1->checkMoveDetect->isChecked())
                {
                    move_frame=m_image;
                    if(flag_count>1){
                        move_result=MoveDetect_FD(move_temp,move_frame);
                    }else{
                        move_result=MoveDetect_FD(move_frame,move_frame);
                    }
                    move_temp=move_frame.clone();
                }else{
                    move_result=m_image;
                }
                t_imageprocess=clock();
                double imageprocesstime=(double)((t_imageprocess-t_imageget)/1000);
                cout<<"Image process time: "<<imageprocesstime<<" ms"<<endl;
                //imshow("move_result",move_result);

                QImage m_q_image=QImage((const unsigned char*)move_result.data, move_result.cols, move_result.rows,move_result.step, QImage::Format_RGB888);
                ui1->ImageView->setPixmap(QPixmap::fromImage(m_q_image));
                ui1->ImageView->show();

                //t_imageview=clock();
               // cout<<"Image view time: "<<(double)((t_imageview-t_imageprocess)/1000)<<" ms"<<endl;

                //cout<<"Loop time: "<<(rt_timer_read()-tstart)/1000000<<" ms"<<endl;
                t_finish=clock();
                double t_total=(double)((t_finish-t_start)/1000);
                //cout<<CLOCKS_PER_SEC<<endl;
                cout<<"Total time: "<<t_total<<" ms"<<endl;
                cout<<move_result.cols<<"  "<<move_result.rows<<endl;
                text_imagegettime=imagegettime;
                text_imageprocesstime=imageprocesstime;
                text_totaltime=t_total;

                //if(flag_choosebackground)
                //{
                ui1->mycustomplot->graph(1)->addData(FrameCountRT,t_total);
                RTTimeforstatistics.push_back(t_total);
                AveragetimetotalRT+=t_total;
                //if(FrameCountRT%100==0){
                    //ui1->mycustomplot->replot();
                //}
                FrameCountRT++;
                //}
            }
        }
    }
}

Mat MoveDetect_FD(Mat temp, Mat frame)
{
    Mat result = frame.clone();
    //Mat result = frame;// make an optimization
    //1.将background和frame转为灰度图

    int Box_X=0,Box_Y=0,Box_W=0,Box_H=0;
    if(ui1->checkSA->isChecked()&&(Target_LocationX!=0||Target_LocationY!=0)){
        Box_X=(Target_LocationX+Target_LocationW/2)-(Target_LocationW*Box_Times/2);
        Box_Y=(Target_LocationY+Target_LocationH/2)-(Target_LocationH*Box_Times/2);
        Box_W=Target_LocationW*Box_Times;
        Box_H=Target_LocationH*Box_Times;

        if(Box_X<0){
            Box_X=0;
        }
        if(Box_Y<0){
            Box_Y=0;
        }
        if(Box_X+Box_W>frame.cols){
            Box_W=frame.cols-Box_X;
        }
        if(Box_Y+Box_H>frame.rows){
            Box_H=frame.rows-Box_Y;
        }

        Rect Box_Region(Box_X,Box_Y,Box_W,Box_H);
        temp=temp(Box_Region);
        frame=frame(Box_Region);
    }

    Mat gray1, gray2;
    cvtColor(temp, gray1, CV_BGR2GRAY);
    cvtColor(frame, gray2, CV_BGR2GRAY);
    //2.将background和frame做差
    Mat diff;
    absdiff(gray1, gray2, diff);

    //3.对差值图diff_thresh进行阈值化处理
    Mat diff_thresh;
    threshold(diff, diff_thresh, 50, 255, CV_THRESH_BINARY);
    //imshow("diff_thresh", diff_thresh);

    //GaussianBlur(diff_thresh, diff_thresh, Size(3, 3), 0, 0);

    //4.腐蚀
    Mat kernel_erode = getStructuringElement(MORPH_RECT, Size(3, 3));
    Mat kernel_dilate = getStructuringElement(MORPH_RECT, Size(18, 18));
    erode(diff_thresh, diff_thresh, kernel_erode);
    //imshow("erode", diff_thresh);
    //5.膨胀
    dilate(diff_thresh, diff_thresh, kernel_dilate);
    //imshow("dilate", diff_thresh);
    //6.查找轮廓并绘制轮廓
    vector<vector<Point> > contours;
    findContours(diff_thresh, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    //drawContours(result, contours, -1, Scalar(0, 0, 255), 2);//在result上绘制轮廓
    //7.查找正外接矩形
    /*Target_LocationX=0;
    Target_LocationY=0;
    Target_LocationW=0;
    Target_LocationH=0;*/
    vector<Rect> boundRect(contours.size());
    for (int i = 0; i < contours.size(); i++)
    {
        boundRect[i] = boundingRect(contours[i]);

        if(ui1->checkSA->isChecked()&&(Target_LocationX!=0||Target_LocationY!=0)){
            boundRect[i].x+=Box_X;
            boundRect[i].y+=Box_Y;
        }

        rectangle(result, boundRect[i], Scalar(0, 255, 0), 2);//在result上绘制正外接矩形
        if(i==0){
            Target_LocationX=boundRect[i].x;
            Target_LocationY=boundRect[i].y;
            Target_LocationW=boundRect[i].width;
            Target_LocationH=boundRect[i].height;
        }
    }

    if(contours.size()==0){
        Target_LocationX=0;
        Target_LocationY=0;
        Target_LocationW=0;
        Target_LocationH=0;
    }

    Location_tobesaved+=QString::number(Target_LocationX,10);
    Location_tobesaved+=" ";
    Location_tobesaved+=QString::number(Target_LocationY,10);
    Location_tobesaved+=" ";
    QTime current_time =QTime::currentTime();
    Location_tobesaved+=current_time.toString("hh:mm:ss.zzz");
    Location_tobesaved+="\n";

    //cv::namedWindow("diff", CV_WINDOW_NORMAL);
    //cv::imshow("diff", diff);
    //cv::waitKey(1);
    //QImage m_q_image=QImage((const unsigned char*)gray1.data, gray1.cols, gray1.rows, QImage::Format_RGB888);
    //ui1->ImageCapture->setPixmap(QPixmap::fromImage(m_q_image));
    //ui1->ImageCapture->show();
    //cout<<"done!"<<endl;
    return result;//返回result
}

void *ProcGetImage_TriFD(void* pParam)
{
    GX_STATUS status = GX_STATUS_SUCCESS;

    //get frame storage cover state
    //bool frame_storage_flag;
    //status=GXGetBool(g_device,GX_BOOL_FRAMESTORE_COVER_ACTIVE, &frame_storage_flag);

    //接收线程启动标志
    g_get_image = true;

    //发送开采命令
    status = GXSendCommand(g_device, GX_COMMAND_ACQUISITION_START);
    if(status != GX_STATUS_SUCCESS)
    {
        GetErrorString(status);
    }

    int flag_count=0;
    Mat move_frame;
    //Mat move_result;
    //Mat move_temp;
    Mat frameprepre;
    Mat framepre;
    Mat diff_thresh1;
    while(g_get_image)
    {
        //RTIME tstart=rt_timer_read();
        clock_t t_start,t_finish,t_imageget,t_imageprocess;
        t_start=clock();
        if(g_frame_data.pImgBuf == NULL)
        {
            continue;
        }

        //set latency
        //usleep(10000);

        status = GXGetImage(g_device, &g_frame_data, 100);
        if(status == GX_STATUS_SUCCESS)
        {
            if(g_frame_data.nStatus == 0)
            {
                //printf("<Successful acquisition : Width: %d Height: %d >\n", g_frame_data.nWidth, g_frame_data.nHeight);
                if(is_implemented)
                {
                    DxRaw8toRGB24(g_frame_data.pImgBuf,m_rgb_image,g_frame_data.nWidth,
                    g_frame_data.nHeight,RAW2RGB_NEIGHBOUR,DX_PIXEL_COLOR_FILTER(BAYERRG),false);
                    memcpy(m_image.data,m_rgb_image,g_frame_data.nHeight*g_frame_data.nWidth*3);
                }else{
                    memcpy(m_image.data,g_frame_data.pImgBuf,g_frame_data.nHeight*g_frame_data.nWidth);
                }
                t_imageget=clock();
                double imagegettime=(double)((t_imageget-t_start)/1000);
                cout<<"Image get time: "<<imagegettime<<" ms"<<endl;
                //QImage m_q_image1=QImage((const unsigned char*)m_image.data, m_image.cols, m_image.rows, QImage::Format_RGB888);
                //ui1->ImageCapture->setPixmap(QPixmap::fromImage(m_image));
                //ui1->ImageCapture->show();

                flag_count++;
                if(ui1->checkMoveDetect->isChecked())
                {
                    move_frame=m_image;
                    if(flag_count>2){
                        move_result=MoveDetect_TriFD(diff_thresh1,framepre,move_frame);
                    }else if(flag_count==2){
                        framepre=move_frame.clone();

                        Mat gray1, gray2;
                        cvtColor(frameprepre, gray1, CV_BGR2GRAY);
                        cvtColor(framepre, gray2, CV_BGR2GRAY);
                        //2.将background和frame做差
                        Mat diff;
                        absdiff(gray1, gray2, diff);
                        threshold(diff, diff_thresh1, 50, 255, CV_THRESH_BINARY);
                    }else{
                        frameprepre=move_frame.clone();
                    }
                    framepre=move_frame.clone();
                }else{
                    move_result=m_image;
                }
                t_imageprocess=clock();
                double imageprocesstime=(double)((t_imageprocess-t_imageget)/1000);
                cout<<"Image process time: "<<imageprocesstime<<" ms"<<endl;
                //imshow("move_result",move_result);

                QImage m_q_image=QImage((const unsigned char*)move_result.data, move_result.cols, move_result.rows,move_result.step, QImage::Format_RGB888);
                ui1->ImageView->setPixmap(QPixmap::fromImage(m_q_image));
                ui1->ImageView->show();

                //t_imageview=clock();
               // cout<<"Image view time: "<<(double)((t_imageview-t_imageprocess)/1000)<<" ms"<<endl;

                //cout<<"Loop time: "<<(rt_timer_read()-tstart)/1000000<<" ms"<<endl;
                t_finish=clock();
                double t_total=(double)((t_finish-t_start)/1000);
                //cout<<CLOCKS_PER_SEC<<endl;
                cout<<"Total time: "<<t_total<<" ms"<<endl;
                cout<<move_result.cols<<"  "<<move_result.rows<<endl;
                text_imagegettime=imagegettime;
                text_imageprocesstime=imageprocesstime;
                text_totaltime=t_total;

                //if(flag_choosebackground)
                //{
                ui1->mycustomplot->graph(0)->addData(FrameCount,t_total);
                Timeforstatistics.push_back(t_total);
                Averagetimetotal+=t_total;
                //if(FrameCount%500==0){
                    //ui1->mycustomplot->replot();
                //}
                FrameCount++;
                //}
            }
        }
    }
}
//xenomai----------version
void ProcGetImage_RT_TriFD(void *arg)
{
    GX_STATUS status = GX_STATUS_SUCCESS;

    //get frame storage cover state
    //bool frame_storage_flag;
    //status=GXGetBool(g_device,GX_BOOL_FRAMESTORE_COVER_ACTIVE, &frame_storage_flag);

    //接收线程启动标志
    g_get_image = true;

    //发送开采命令
    status = GXSendCommand(g_device, GX_COMMAND_ACQUISITION_START);
    if(status != GX_STATUS_SUCCESS)
    {
        GetErrorString(status);
    }

    int flag_count=0;
    Mat move_frame;
    //Mat move_result;
    //Mat move_temp;
    Mat frameprepre;
    Mat framepre;
    Mat diff_thresh1;
    while(g_get_image)
    {
        //RTIME tstart=rt_timer_read();
        clock_t t_start,t_finish,t_imageget,t_imageprocess;
        t_start=clock();
        if(g_frame_data.pImgBuf == NULL)
        {
            continue;
        }

        //set latency
        //usleep(10000);

        status = GXGetImage(g_device, &g_frame_data, 100);
        if(status == GX_STATUS_SUCCESS)
        {
            if(g_frame_data.nStatus == 0)
            {
                //printf("<Successful acquisition : Width: %d Height: %d >\n", g_frame_data.nWidth, g_frame_data.nHeight);
                if(is_implemented)
                {
                    DxRaw8toRGB24(g_frame_data.pImgBuf,m_rgb_image,g_frame_data.nWidth,
                    g_frame_data.nHeight,RAW2RGB_NEIGHBOUR,DX_PIXEL_COLOR_FILTER(BAYERRG),false);
                    memcpy(m_image.data,m_rgb_image,g_frame_data.nHeight*g_frame_data.nWidth*3);
                }else{
                    memcpy(m_image.data,g_frame_data.pImgBuf,g_frame_data.nHeight*g_frame_data.nWidth);
                }
                t_imageget=clock();
                double imagegettime=(double)((t_imageget-t_start)/1000);
                cout<<"Image get time: "<<imagegettime<<" ms"<<endl;
                //QImage m_q_image1=QImage((const unsigned char*)m_image.data, m_image.cols, m_image.rows, QImage::Format_RGB888);
                //ui1->ImageCapture->setPixmap(QPixmap::fromImage(m_image));
                //ui1->ImageCapture->show();

                flag_count++;
                if(ui1->checkMoveDetect->isChecked())
                {
                    move_frame=m_image;
                    if(flag_count>2){
                        move_result=MoveDetect_TriFD(diff_thresh1,framepre,move_frame);
                    }else if(flag_count==2){
                        framepre=move_frame.clone();

                        Mat gray1, gray2;
                        cvtColor(frameprepre, gray1, CV_BGR2GRAY);
                        cvtColor(framepre, gray2, CV_BGR2GRAY);
                        //2.将background和frame做差
                        Mat diff;
                        absdiff(gray1, gray2, diff);
                        threshold(diff, diff_thresh1, 50, 255, CV_THRESH_BINARY);
                    }else{
                        frameprepre=move_frame.clone();
                    }
                    framepre=move_frame.clone();
                }else{
                    move_result=m_image;
                }
                t_imageprocess=clock();
                double imageprocesstime=(double)((t_imageprocess-t_imageget)/1000);
                cout<<"Image process time: "<<imageprocesstime<<" ms"<<endl;
                //imshow("move_result",move_result);

                QImage m_q_image=QImage((const unsigned char*)move_result.data, move_result.cols, move_result.rows,move_result.step, QImage::Format_RGB888);
                ui1->ImageView->setPixmap(QPixmap::fromImage(m_q_image));
                ui1->ImageView->show();

                //t_imageview=clock();
               // cout<<"Image view time: "<<(double)((t_imageview-t_imageprocess)/1000)<<" ms"<<endl;

                //cout<<"Loop time: "<<(rt_timer_read()-tstart)/1000000<<" ms"<<endl;
                t_finish=clock();
                double t_total=(double)((t_finish-t_start)/1000);
                //cout<<CLOCKS_PER_SEC<<endl;
                cout<<"Total time: "<<t_total<<" ms"<<endl;
                cout<<move_result.cols<<"  "<<move_result.rows<<endl;
                text_imagegettime=imagegettime;
                text_imageprocesstime=imageprocesstime;
                text_totaltime=t_total;

                //if(flag_choosebackground)
                //{
                ui1->mycustomplot->graph(1)->addData(FrameCountRT,t_total);
                RTTimeforstatistics.push_back(t_total);
                AveragetimetotalRT+=t_total;
                //if(FrameCountRT%100==0){
                    //ui1->mycustomplot->replot();
                //}
                FrameCountRT++;
                //}
            }
        }
    }
}
Mat MoveDetect_TriFD(Mat &diff_thresh1,Mat framepre,Mat frame)
{
    Mat result = frame.clone();
    //Mat result = frame;// make an optimization
    //1.将background和frame转为灰度图
    Mat gray1, gray2;
    cvtColor(framepre, gray1, CV_BGR2GRAY);
    cvtColor(frame, gray2, CV_BGR2GRAY);
    //2.将background和frame做差
    Mat diff;
    absdiff(gray1, gray2, diff);

    //3.对差值图diff_thresh进行阈值化处理
    Mat diff_thresh2;
    threshold(diff, diff_thresh2, 50, 255, CV_THRESH_BINARY);
    //imshow("diff_thresh", diff_thresh);

    //GaussianBlur(diff_thresh, diff_thresh, Size(3, 3), 0, 0);

    Mat region_diff_thresh1=diff_thresh1,region_diff_thresh2=diff_thresh2;
    int Box_X=0,Box_Y=0,Box_W=0,Box_H=0;
    if(ui1->checkSA->isChecked()&&(Target_LocationX!=0||Target_LocationY!=0)){
        Box_X=(Target_LocationX+Target_LocationW/2)-(Target_LocationW*Box_Times/2);
        Box_Y=(Target_LocationY+Target_LocationH/2)-(Target_LocationH*Box_Times/2);
        Box_W=Target_LocationW*Box_Times;
        Box_H=Target_LocationH*Box_Times;

        if(Box_X<0){
            Box_X=0;
        }
        if(Box_Y<0){
            Box_Y=0;
        }
        if(Box_X+Box_W>frame.cols){
            Box_W=frame.cols-Box_X;
        }
        if(Box_Y+Box_H>frame.rows){
            Box_H=frame.rows-Box_Y;
        }

        Rect Box_Region(Box_X,Box_Y,Box_W,Box_H);
        region_diff_thresh1=diff_thresh1(Box_Region);
        region_diff_thresh2=diff_thresh2(Box_Region);
    }
    //for TriFrame algorithm;
    Mat diff_thresh;
    bitwise_and(region_diff_thresh1, region_diff_thresh2, diff_thresh);
    diff_thresh1=diff_thresh2;

    //4.腐蚀
    Mat kernel_erode = getStructuringElement(MORPH_RECT, Size(3, 3));
    Mat kernel_dilate = getStructuringElement(MORPH_RECT, Size(18, 18));
    erode(diff_thresh, diff_thresh, kernel_erode);
    //imshow("erode", diff_thresh);
    //5.膨胀
    dilate(diff_thresh, diff_thresh, kernel_dilate);
    //imshow("dilate", diff_thresh);
    //6.查找轮廓并绘制轮廓
    vector<vector<Point> > contours;
    findContours(diff_thresh, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    //drawContours(result, contours, -1, Scalar(0, 0, 255), 2);//在result上绘制轮廓
    //7.查找正外接矩形
    /*Target_LocationX=0;
    Target_LocationY=0;
    Target_LocationW=0;
    Target_LocationH=0;*/
    vector<Rect> boundRect(contours.size());
    for (int i = 0; i < contours.size(); i++)
    {
        boundRect[i] = boundingRect(contours[i]);

        if(ui1->checkSA->isChecked()&&(Target_LocationX!=0||Target_LocationY!=0)){
            boundRect[i].x+=Box_X;
            boundRect[i].y+=Box_Y;
        }

        rectangle(result, boundRect[i], Scalar(0, 255, 0), 2);//在result上绘制正外接矩形
        if(i==0){
            Target_LocationX=boundRect[i].x;
            Target_LocationY=boundRect[i].y;
            Target_LocationW=boundRect[i].width;
            Target_LocationH=boundRect[i].height;
        }
    }

    if(contours.size()==0){
        Target_LocationX=0;
        Target_LocationY=0;
        Target_LocationW=0;
        Target_LocationH=0;
    }

    Location_tobesaved+=QString::number(Target_LocationX,10);
    Location_tobesaved+=" ";
    Location_tobesaved+=QString::number(Target_LocationY,10);
    Location_tobesaved+=" ";
    QTime current_time =QTime::currentTime();
    Location_tobesaved+=current_time.toString("hh:mm:ss.zzz");
    Location_tobesaved+="\n";

    //cv::namedWindow("diff", CV_WINDOW_NORMAL);
    //cv::imshow("diff", diff);
    //cv::waitKey(1);
    //QImage m_q_image=QImage((const unsigned char*)gray1.data, gray1.cols, gray1.rows, QImage::Format_RGB888);
    //ui1->ImageCapture->setPixmap(QPixmap::fromImage(m_q_image));
    //ui1->ImageCapture->show();
    //cout<<"done!"<<endl;
    return result;//返回result
}

void *ProcGetImage_CNN(void* pParam)
{
    //dlib---------------
    Mat src_img;
    //string model_path = "/home/jerry/Dlib-19.17/mmod_human_face_detector.dat";
    //string model_path = "/home/jerry/Dlib-19.17/mmod_network.dat";
    string model_path = "/home/jerry/Dlib-19.17/mmod_rear_end_vehicle_detector.dat";
    net_type net;
    dlib::deserialize(model_path) >> net;
    dlib::matrix<dlib::rgb_pixel> dlib_img;
    //--------------------

    GX_STATUS status = GX_STATUS_SUCCESS;

    //get frame storage cover state
    //bool frame_storage_flag;
    //status=GXGetBool(g_device,GX_BOOL_FRAMESTORE_COVER_ACTIVE, &frame_storage_flag);

    //接收线程启动标志
    g_get_image = true;

    //发送开采命令
    status = GXSendCommand(g_device, GX_COMMAND_ACQUISITION_START);
    if(status != GX_STATUS_SUCCESS)
    {
        GetErrorString(status);
    }

    //int flag_count=0;
    //Mat move_frame;

    //Mat move_result;
    //Mat move_temp;
    while(g_get_image)
    {
        //RTIME tstart=rt_timer_read();
        clock_t t_start,t_finish,t_imageget,t_imageprocess;
        t_start=clock();
        if(g_frame_data.pImgBuf == NULL)
        {
            continue;
        }

        //set latency
        //usleep(10000);

        status = GXGetImage(g_device, &g_frame_data, 100);
        if(status == GX_STATUS_SUCCESS)
        {
            if(g_frame_data.nStatus == 0)
            {
                //printf("<Successful acquisition : Width: %d Height: %d >\n", g_frame_data.nWidth, g_frame_data.nHeight);
                if(is_implemented)
                {
                    DxRaw8toRGB24(g_frame_data.pImgBuf,m_rgb_image,g_frame_data.nWidth,
                    g_frame_data.nHeight,RAW2RGB_NEIGHBOUR,DX_PIXEL_COLOR_FILTER(BAYERRG),false);
                    memcpy(m_image.data,m_rgb_image,g_frame_data.nHeight*g_frame_data.nWidth*3);
                }else{
                    memcpy(m_image.data,g_frame_data.pImgBuf,g_frame_data.nHeight*g_frame_data.nWidth);
                }
                t_imageget=clock();
                double imagegettime=(double)((t_imageget-t_start)/1000);
                cout<<"Image get time: "<<imagegettime<<" ms"<<endl;
                //QImage m_q_image1=QImage((const unsigned char*)m_image.data, m_image.cols, m_image.rows, QImage::Format_RGB888);
                //ui1->ImageCapture->setPixmap(QPixmap::fromImage(m_image));
                //ui1->ImageCapture->show();

                //flag_count++;
                if(ui1->checkMoveDetect->isChecked())
                {
                    src_img=m_image;
                    dlib::assign_image(dlib_img, dlib::cv_image<dlib::rgb_pixel>(src_img));
                    vector<Rect> faces;
                    auto dets = net(dlib_img);
                    for (auto&& det : dets){//type(rt):mmod_rect
                       dlib::rectangle rt=det.rect;
                       long tl_x, tl_y;
                       unsigned long h,w;
                       tl_x=rt.left();
                       tl_y=rt.top();
                       h=rt.height();
                       w=rt.width();
                       //tl_x, tl_y = rt.left(), rt.top();
                       //h ,w = rt.height(),rt.width();
                       faces.push_back(Rect(tl_x,tl_y,w,h));
                    }
                    Target_LocationX=0;
                    Target_LocationY=0;
                    for (auto t = 0; t < faces.size(); ++t)
                    {
                        rectangle(src_img, faces[t], Scalar(0, 0, 255), 1, 8, 0);            // 用红色矩形框出人脸
                        if(t==0){
                            Target_LocationX=faces[t].x;
                            Target_LocationY=faces[t].y;
                        }
                    }

                    Location_tobesaved+=QString::number(Target_LocationX,10);
                    Location_tobesaved+=" ";
                    Location_tobesaved+=QString::number(Target_LocationY,10);
                    Location_tobesaved+=" ";
                    QTime current_time =QTime::currentTime();
                    Location_tobesaved+=current_time.toString("hh:mm:ss.zzz");
                    Location_tobesaved+="\n";

                    move_result=src_img.clone();
                    /*if(flag_count>1){
                        move_result=MoveDetect_FD(move_temp,move_frame);
                    }else{
                        move_result=MoveDetect_FD(move_frame,move_frame);
                    }
                    move_temp=move_frame.clone();*/
                }else{
                    move_result=m_image;
                }
                t_imageprocess=clock();
                double imageprocesstime=(double)((t_imageprocess-t_imageget)/1000);
                cout<<"Image process time: "<<imageprocesstime<<" ms"<<endl;
                //imshow("move_result",move_result);

                QImage m_q_image=QImage((const unsigned char*)move_result.data, move_result.cols, move_result.rows,move_result.step, QImage::Format_RGB888);
                ui1->ImageView->setPixmap(QPixmap::fromImage(m_q_image));
                ui1->ImageView->show();

                //t_imageview=clock();
               // cout<<"Image view time: "<<(double)((t_imageview-t_imageprocess)/1000)<<" ms"<<endl;

                //cout<<"Loop time: "<<(rt_timer_read()-tstart)/1000000<<" ms"<<endl;
                t_finish=clock();
                double t_total=(double)((t_finish-t_start)/1000);
                //cout<<CLOCKS_PER_SEC<<endl;
                cout<<"Total time: "<<t_total<<" ms"<<endl;
                cout<<move_result.cols<<"  "<<move_result.rows<<endl;
                text_imagegettime=imagegettime;
                text_imageprocesstime=imageprocesstime;
                text_totaltime=t_total;

                //if(flag_choosebackground)
                //{
                ui1->mycustomplot->graph(0)->addData(FrameCount,t_total);
                Timeforstatistics.push_back(t_total);
                Averagetimetotal+=t_total;
                //if(FrameCount%500==0){
                    //ui1->mycustomplot->replot();
                //}
                FrameCount++;
                //}
            }
        }
    }
}
//xenomai----------version
void ProcGetImage_RT_CNN(void *arg)
{
    //dlib---------------
    Mat src_img;
    //string model_path = "/home/jerry/Dlib-19.17/mmod_human_face_detector.dat";
    //string model_path = "/home/jerry/Dlib-19.17/mmod_network.dat";
    string model_path = "/home/jerry/Dlib-19.17/mmod_rear_end_vehicle_detector.dat";
    net_type net;
    dlib::deserialize(model_path) >> net;
    dlib::matrix<dlib::rgb_pixel> dlib_img;
    //--------------------

    GX_STATUS status = GX_STATUS_SUCCESS;

    //get frame storage cover state
    //bool frame_storage_flag;
    //status=GXGetBool(g_device,GX_BOOL_FRAMESTORE_COVER_ACTIVE, &frame_storage_flag);

    //接收线程启动标志
    g_get_image = true;

    //发送开采命令
    status = GXSendCommand(g_device, GX_COMMAND_ACQUISITION_START);
    if(status != GX_STATUS_SUCCESS)
    {
        GetErrorString(status);
    }

    //int flag_count=0;
    //Mat move_frame;

    //Mat move_result;
    //Mat move_temp;
    while(g_get_image)
    {
        //RTIME tstart=rt_timer_read();
        clock_t t_start,t_finish,t_imageget,t_imageprocess;
        t_start=clock();
        if(g_frame_data.pImgBuf == NULL)
        {
            continue;
        }

        //set latency
        //usleep(10000);

        status = GXGetImage(g_device, &g_frame_data, 100);
        if(status == GX_STATUS_SUCCESS)
        {
            if(g_frame_data.nStatus == 0)
            {
                //printf("<Successful acquisition : Width: %d Height: %d >\n", g_frame_data.nWidth, g_frame_data.nHeight);
                if(is_implemented)
                {
                    DxRaw8toRGB24(g_frame_data.pImgBuf,m_rgb_image,g_frame_data.nWidth,
                    g_frame_data.nHeight,RAW2RGB_NEIGHBOUR,DX_PIXEL_COLOR_FILTER(BAYERRG),false);
                    memcpy(m_image.data,m_rgb_image,g_frame_data.nHeight*g_frame_data.nWidth*3);
                }else{
                    memcpy(m_image.data,g_frame_data.pImgBuf,g_frame_data.nHeight*g_frame_data.nWidth);
                }
                t_imageget=clock();
                double imagegettime=(double)((t_imageget-t_start)/1000);
                cout<<"Image get time: "<<imagegettime<<" ms"<<endl;
                //QImage m_q_image1=QImage((const unsigned char*)m_image.data, m_image.cols, m_image.rows, QImage::Format_RGB888);
                //ui1->ImageCapture->setPixmap(QPixmap::fromImage(m_image));
                //ui1->ImageCapture->show();

                //flag_count++;
                if(ui1->checkMoveDetect->isChecked())
                {
                    src_img=m_image;
                    dlib::assign_image(dlib_img, dlib::cv_image<dlib::rgb_pixel>(src_img));
                    vector<Rect> faces;
                    auto dets = net(dlib_img);
                    for (auto&& det : dets){//type(rt):mmod_rect
                       dlib::rectangle rt=det.rect;
                       long tl_x, tl_y;
                       unsigned long h,w;
                       tl_x=rt.left();
                       tl_y=rt.top();
                       h=rt.height();
                       w=rt.width();
                       //tl_x, tl_y = rt.left(), rt.top();
                       //h ,w = rt.height(),rt.width();
                       faces.push_back(Rect(tl_x,tl_y,w,h));
                    }
                    Target_LocationX=0;
                    Target_LocationY=0;
                    for (auto t = 0; t < faces.size(); ++t)
                    {
                        rectangle(src_img, faces[t], Scalar(0, 0, 255), 1, 8, 0);            // 用红色矩形框出人脸
                        if(t==0){
                            Target_LocationX=faces[t].x;
                            Target_LocationY=faces[t].y;
                        }
                    }

                    Location_tobesaved+=QString::number(Target_LocationX,10);
                    Location_tobesaved+=" ";
                    Location_tobesaved+=QString::number(Target_LocationY,10);
                    Location_tobesaved+=" ";
                    QTime current_time =QTime::currentTime();
                    Location_tobesaved+=current_time.toString("hh:mm:ss.zzz");
                    Location_tobesaved+="\n";

                    move_result=src_img.clone();
                    /*if(flag_count>1){
                        move_result=MoveDetect_FD(move_temp,move_frame);
                    }else{
                        move_result=MoveDetect_FD(move_frame,move_frame);
                    }
                    move_temp=move_frame.clone();*/
                }else{
                    move_result=m_image;
                }
                t_imageprocess=clock();
                double imageprocesstime=(double)((t_imageprocess-t_imageget)/1000);
                cout<<"Image process time: "<<imageprocesstime<<" ms"<<endl;
                //imshow("move_result",move_result);

                QImage m_q_image=QImage((const unsigned char*)move_result.data, move_result.cols, move_result.rows,move_result.step, QImage::Format_RGB888);
                ui1->ImageView->setPixmap(QPixmap::fromImage(m_q_image));
                ui1->ImageView->show();

                //t_imageview=clock();
               // cout<<"Image view time: "<<(double)((t_imageview-t_imageprocess)/1000)<<" ms"<<endl;

                //cout<<"Loop time: "<<(rt_timer_read()-tstart)/1000000<<" ms"<<endl;
                t_finish=clock();
                double t_total=(double)((t_finish-t_start)/1000);
                //cout<<CLOCKS_PER_SEC<<endl;
                cout<<"Total time: "<<t_total<<" ms"<<endl;
                cout<<move_result.cols<<"  "<<move_result.rows<<endl;
                text_imagegettime=imagegettime;
                text_imageprocesstime=imageprocesstime;
                text_totaltime=t_total;

                //if(flag_choosebackground)
                //{
                ui1->mycustomplot->graph(1)->addData(FrameCountRT,t_total);
                RTTimeforstatistics.push_back(t_total);
                AveragetimetotalRT+=t_total;
                //if(FrameCountRT%100==0){
                    //ui1->mycustomplot->replot();
                //}
                FrameCountRT++;
                //}
            }
        }
    }
}

//WJN--------------------------
void MainWindow::on_ButtonStart_clicked()
{
    //WJN-------------------


        /*printf("\n");
        printf("-------------------------------------------------------------\n");
        printf("sample to show how to acquire image continuously.\n");
        #ifdef __x86_64__
        printf("version: 1.0.1605.8041\n");
        #elif __i386__
        printf("version: 1.0.1605.9041\n");
        #endif
        printf("-------------------------------------------------------------\n");
        printf("\n");

        printf("Press [x] or [X] and then press [Enter] to Exit the Program\n");
        printf("Initializing......");
        printf("\n\n");


        usleep(2000000);*/



        //初始化设备打开参数，默认打开序号为1的设备
        open_param.accessMode = GX_ACCESS_EXCLUSIVE;
        open_param.openMode = GX_OPEN_INDEX;
        open_param.pszContent = "1";

        //初始化库
        status = GXInitLib();
        if(status != GX_STATUS_SUCCESS)
        {
            GetErrorString(status);
            exit(0);
        }

        //获取枚举设备个数
        status = GXUpdateDeviceList(&device_num, 1000);
        if(status != GX_STATUS_SUCCESS)
        {
            GetErrorString(status);
            status = GXCloseLib();
            exit(0);
        }

        if(device_num <= 0)
        {
            //printf("<No device>\n");
            status = GXCloseLib();
            exit(0);
        }
        else
        {
            //默认打开第1个设备
            status = GXOpenDevice(&open_param, &g_device);
            if(status == GX_STATUS_SUCCESS)
        {
                //get the initial resolution
                QString tempresW=ui1->SetInitialResWlineEdit->text();
                QString tempresH=ui1->SetInitialResHlineEdit->text();
                if(tempresW!=""&&tempresH!=""){
                    int64_t nWidth = tempresW.toInt();
                    int64_t nHeight = tempresH.toInt();
                    int64_t nOffsetX =1024-nWidth/2;
                    int64_t nOffsetY =768-nHeight/2;
                    status = GXSetInt(g_device,GX_INT_WIDTH,nWidth);
                    status = GXSetInt(g_device,GX_INT_HEIGHT,nHeight);
                    status = GXSetInt(g_device,GX_INT_OFFSET_X,nOffsetX);
                    status = GXSetInt(g_device,GX_INT_OFFSET_Y,nOffsetY);
                }

                //printf("<Open device success>\n");
            int64_t width,height;
            status = GXGetInt(g_device,GX_INT_WIDTH,&width);
            status = GXGetInt(g_device,GX_INT_HEIGHT,&height);
            // 查询当前相机是否支持 GX_ENUM_PIXEL_COLOR_FILTER
            status=GXIsImplemented(g_device,GX_ENUM_PIXEL_COLOR_FILTER,
            &is_implemented);
            //支持彩色图像
            if(is_implemented)
            {
                status= GXGetEnum(g_device, GX_ENUM_PIXEL_COLOR_FILTER, &m_pixel_color);
                m_image.create(height,width,CV_8UC3);//彩色相机
                m_rgb_image = new char[width*height*3];
            }else{
                m_image.create(height,width,CV_8UC1);//黑白相机
            }
            }
            else
            {
                //printf("<Open device fail>\n");
                status = GXCloseLib();
                exit(0);
            }
        }

        //设置采集模式为连续采集
        status = GXSetEnum(g_device, GX_ENUM_ACQUISITION_MODE, GX_ACQ_MODE_CONTINUOUS);
        if(status != GX_STATUS_SUCCESS)
        {
            GetErrorString(status);
            status = GXCloseDevice(g_device);
            if(g_device != NULL)
            {
                g_device = NULL;
            }
            status = GXCloseLib();
            exit(0);
        }

        //设置触发开关为OFF
        status = GXSetEnum(g_device, GX_ENUM_TRIGGER_MODE, GX_TRIGGER_MODE_OFF);
        if(status != GX_STATUS_SUCCESS)
        {
            GetErrorString(status);
            status = GXCloseDevice(g_device);
            if(g_device != NULL)
            {
                g_device = NULL;
            }
            status = GXCloseLib();
            exit(0);
        }

        //为采集做准备
        ret = PreForImage();
        if(ret != 0)
        {
            //printf("<Failed to prepare for acquire image>\n");
            status = GXCloseDevice(g_device);
            if(g_device != NULL)
            {
                g_device = NULL;
            }
            status = GXCloseLib();
            exit(0);
        }

        //set the size of image
        /*int64_t nWidth = 600;
        int64_t nHeight = 400;
        int64_t nOffsetX =724;
        int64_t nOffsetY =568;
        status = GXSetInt(g_device,GX_INT_WIDTH,nWidth);
        status = GXSetInt(g_device,GX_INT_HEIGHT,nHeight);
        status = GXSetInt(g_device,GX_INT_OFFSET_X,nOffsetX);
        status = GXSetInt(g_device,GX_INT_OFFSET_Y,nOffsetY);*/

        //initialization for target location
        Target_LocationX=0;
        Target_LocationY=0;
        Target_LocationW=0;
        Target_LocationH=0;


        if(ui1->checkRT->isChecked())
        {
            //RT_task_version---------------------
            mlockall(MCL_CURRENT | MCL_FUTURE);
            char str[20];
            sprintf(str, "MygetimageRT_task");
            rt_task_create(&GetimageRT_task, str, 0, 50, 0);
            if(ui1->checkBD->isChecked()){
                rt_task_start(&GetimageRT_task, &ProcGetImage_RT_BD, 0);
            }
            if(ui1->checkFD->isChecked()){
                rt_task_start(&GetimageRT_task, &ProcGetImage_RT_FD, 0);
            }
            if(ui1->checkTriFD->isChecked()){
                rt_task_start(&GetimageRT_task, &ProcGetImage_RT_TriFD, 0);
            }
            if(ui1->checkCNN->isChecked()){
                rt_task_start(&GetimageRT_task, &ProcGetImage_RT_CNN, 0);
            }
        }else{
            //pthread_version----------------
            //启动接收线程
            if(ui1->checkBD->isChecked()){
                ret = pthread_create(&g_acquire_thread, 0, ProcGetImage_BD, 0);
            }
            if(ui1->checkFD->isChecked()){
                ret = pthread_create(&g_acquire_thread, 0, ProcGetImage_FD, 0);
            }
            if(ui1->checkTriFD->isChecked()){
                ret = pthread_create(&g_acquire_thread, 0, ProcGetImage_TriFD, 0);
            }
            if(ui1->checkCNN->isChecked()){
                ret = pthread_create(&g_acquire_thread, 0, ProcGetImage_CNN, 0);
            }

            if(ret != 0)
            {
                //printf("<Failed to create the collection thread>\n");
                status = GXCloseDevice(g_device);
                if(g_device != NULL)
                {
                    g_device = NULL;
                }
                status = GXCloseLib();
                exit(0);
            }
        }

        /*bool run = true;
        while(run == true)
        {
            int c = getchar();

            switch(c)
            {
                //退出程序
                case 'X':
                case 'x':
                    run = false;
                    break;
                default:
                    break;
            }
        }*/

    //for location save function initialization.
    Location_tobesaved="";
    //WJN-------------------
}
void MainWindow::on_ButtonStop_clicked()
{
    //munlockall();
    //为停止采集做准备
    ret = UnPreForImage();
    if(ret != 0)
    {
        status = GXCloseDevice(g_device);
        if(g_device != NULL)
        {
            g_device = NULL;
        }
        status = GXCloseLib();
        exit(0);
    }

    //关闭设备
    status = GXCloseDevice(g_device);
    if(status != GX_STATUS_SUCCESS)
    {
        GetErrorString(status);
        if(g_device != NULL)
        {
            g_device = NULL;
        }
        status = GXCloseLib();
        exit(0);
    }

    //释放库
    status = GXCloseLib();

    flag_choosebackground=false;
}

void MainWindow::on_ButtonGain_clicked()
{
    status=GXSetEnum(g_device,GX_ENUM_GAIN_AUTO,GX_GAIN_AUTO_ONCE);
}

void MainWindow::on_ButtonWhiteBalance_clicked()
{
    status=GXSetEnum(g_device,GX_ENUM_BALANCE_WHITE_AUTO,GX_BALANCE_WHITE_AUTO_ONCE);
}

/*void CaptureImage_RT(void *arg)
{
    QImage m_q_image=QImage((const unsigned char*)move_result.data, move_result.cols, move_result.rows, QImage::Format_RGB888);
    ui1->ImageCapture->setPixmap(QPixmap::fromImage(m_q_image));
    ui1->ImageCapture->show();

    QString tempstr;
    //set the time statistics
    tempstr = QString::number(text_imagegettime, 10,1);
    ui1->ImageGetTimelineEdit->setText(tempstr);

    tempstr = QString::number(text_imageprocesstime, 10,1);
    ui1->ImageProcessTimelineEdit->setText(tempstr);

    tempstr = QString::number(text_totaltime, 10,1);
    ui1->TotalTimelineEdit->setText(tempstr);

    //set the Current resolutions
    tempstr = QString::number(move_result.cols,10);
    ui1->CurrentResXlineEdit->setText(tempstr);

    tempstr = QString::number(move_result.rows,10);
    ui1->CurrentResYlineEdit->setText(tempstr);

    //set the target location
    tempstr = QString::number(Target_LocationX,10);
    ui1->TargetLocationXlineEdit->setText(tempstr);

    tempstr = QString::number(Target_LocationY,10);
    ui1->TargetLocationYlineEdit->setText(tempstr);
}*/

void MainWindow::on_ButtonCapture_clicked()
{
    /*status = GXGetImage(g_device, &g_frame_data, 100);
    if(status == GX_STATUS_SUCCESS)
    {
        if(g_frame_data.nStatus == 0)
        {
            //printf("<Successful acquisition : Width: %d Height: %d >\n", g_frame_data.nWidth, g_frame_data.nHeight);
            if(is_implemented)
            {
                DxRaw8toRGB24(g_frame_data.pImgBuf,m_rgb_image,g_frame_data.nWidth,
                g_frame_data.nHeight,RAW2RGB_NEIGHBOUR,DX_PIXEL_COLOR_FILTER(BAYERRG),false);
                memcpy(m_image.data,m_rgb_image,g_frame_data.nHeight*g_frame_data.nWidth*3);
            }else{
                memcpy(m_image.data,g_frame_data.pImgBuf,g_frame_data.nHeight*g_frame_data.nWidth);
            }

            //imshow("move_result",move_result);
            QImage m_q_image=QImage((const unsigned char*)m_image.data, m_image.cols, m_image.rows, QImage::Format_RGB888);
            ui1->ImageCapture->setPixmap(QPixmap::fromImage(m_q_image));
            ui1->ImageCapture->show();
        }
    }*/

   /* char str[20];
    sprintf(str, "MyCaptureImageRT_task");
    mlockall(MCL_CURRENT | MCL_FUTURE);
    rt_task_create(&CaptureImageRT_task, str, 0, 50, 0);
    rt_task_start(&CaptureImageRT_task, &CaptureImage_RT, 0);
    usleep(1000);
    munlockall();
    rt_task_delete(&CaptureImageRT_task);*/

    QImage m_q_image=QImage((const unsigned char*)move_result.data, move_result.cols, move_result.rows, QImage::Format_RGB888);
    ui1->ImageCapture->setPixmap(QPixmap::fromImage(m_q_image));
    ui1->ImageCapture->show();

    QString tempstr;
    //set the time statistics
    tempstr = QString::number(text_imagegettime, 10,1);
    ui1->ImageGetTimelineEdit->setText(tempstr);

    tempstr = QString::number(text_imageprocesstime, 10,1);
    ui1->ImageProcessTimelineEdit->setText(tempstr);

    tempstr = QString::number(text_totaltime, 10,1);
    ui1->TotalTimelineEdit->setText(tempstr);

    tempstr = QString::number(1000/text_totaltime, 10,1);
    ui1->FPSlineEdit->setText(tempstr);

    //set the Current resolutions
    tempstr = QString::number(move_result.cols,10);
    ui1->CurrentResXlineEdit->setText(tempstr);

    tempstr = QString::number(move_result.rows,10);
    ui1->CurrentResYlineEdit->setText(tempstr);

    //set the target location
    tempstr = QString::number(Target_LocationX,10);
    ui1->TargetLocationXlineEdit->setText(tempstr);

    tempstr = QString::number(Target_LocationY,10);
    ui1->TargetLocationYlineEdit->setText(tempstr);

}

void MainWindow::on_ButtonReplot_clicked()
{
    ui1->mycustomplot->replot();

    QString tempstr;
    if(!ui1->checkRT->isChecked())
    {
        double mean=Averagetimetotal/FrameCount;
        tempstr = QString::number(mean, 10,1);
        ui1->AverageTimelineEdit->setText(tempstr);

        tempstr = QString::number(1000/mean, 10,1);
        ui1->AverageFPSlineEdit->setText(tempstr);

        double accum=0;
        for_each(begin(Timeforstatistics),end(Timeforstatistics),[&](const double d){
            accum += (d-mean)*(d-mean);
        });
        double Variancetime = sqrt(accum/(FrameCount-1));
        tempstr = QString::number(Variancetime, 10,1);
        ui1->VarianceTimelineEdit->setText(tempstr);

        int maxcountflag=0;
        for_each(begin(Timeforstatistics),end(Timeforstatistics),[&](const double d){
            maxcountflag++;
            if(maxcountflag>100&&d>Maxtime){
                Maxtime=d;
            }
        });
        tempstr = QString::number(Maxtime, 10,1);
        ui1->MaxTimelineEdit->setText(tempstr);
    }else{
        double mean=AveragetimetotalRT/FrameCountRT;
        tempstr = QString::number(mean, 10,1);
        ui1->RTAverageTimelineEdit->setText(tempstr);

        tempstr = QString::number(1000/mean, 10,1);
        ui1->RTAverageFPSlineEdit->setText(tempstr);

        double accum=0;
        for_each(begin(RTTimeforstatistics),end(RTTimeforstatistics),[&](const double d){
            accum += (d-mean)*(d-mean);
        });
        double VariancetimeRT = sqrt(accum/(FrameCountRT-1));
        tempstr = QString::number(VariancetimeRT, 10,1);
        ui1->RTVarianceTimelineEdit->setText(tempstr);

        int maxcountflag=0;
        for_each(begin(RTTimeforstatistics),end(RTTimeforstatistics),[&](const double d){
            maxcountflag++;
            if(maxcountflag>100&&d>MaxtimeRT){
                MaxtimeRT=d;
            }
        });
        tempstr = QString::number(MaxtimeRT, 10,1);
        ui1->RTMaxTimelineEdit->setText(tempstr);
    }
}

void MainWindow::on_ButtonReset_clicked()
{
    ui1->mycustomplot->graph(0)->data().data()->clear();
    ui1->mycustomplot->graph(1)->data().data()->clear();
    ui1->mycustomplot->replot();

    ui1->AverageTimelineEdit->setText("");
    ui1->RTAverageTimelineEdit->setText("");
    ui1->VarianceTimelineEdit->setText("");
    ui1->RTVarianceTimelineEdit->setText("");
    ui1->AverageFPSlineEdit->setText("");
    ui1->RTAverageFPSlineEdit->setText("");
    ui1->MaxTimelineEdit->setText("");
    ui1->RTMaxTimelineEdit->setText("");

    FrameCount=0;
    FrameCountRT=0;

    Averagetimetotal=0;
    AveragetimetotalRT=0;

    Timeforstatistics.clear();
    RTTimeforstatistics.clear();

    Maxtime=0;
    MaxtimeRT=0;
}

void MainWindow::on_ButtonChooseBG_clicked()
{
    move_temp=m_image.clone();
    flag_choosebackground=true;
}

void MainWindow::on_checkBD_stateChanged(int arg1)
{
    if(arg1){
        ui->ButtonChooseBG->setEnabled(true);
    }else{
        ui->ButtonChooseBG->setEnabled(false);
    }
}

void MainWindow::on_ButtonSave_clicked()
{
    QString FilePath="/home/jerry/daheng/dhcam_install_20180302/dh_camera/daheng-sdk-x64/sample_cn/Target_Detector/result/";

    QDateTime current_date_time =QDateTime::currentDateTime();
    QString current_date =current_date_time.toString("yyyy.MM.dd_hh:mm:ss");
    QString FileName=FilePath+current_date;
    if(ui->checkBD->isChecked()){
        FileName+="_BD.txt";
    }else if(ui->checkFD->isChecked()){
        FileName+="_FD.txt";
    }else if(ui->checkTriFD->isChecked()){
        FileName+="_TriFD.txt";
    }else if(ui->checkCNN->isChecked()){
        FileName+="_CNN.txt";
    }

    QFile myfile(FileName);
    if(!myfile.open(QIODevice::WriteOnly|QIODevice::Text))
    {
        QMessageBox::critical(this,"提示","无法创建文件！");
        return;
    }
    QTextStream txtOutput(&myfile);
    txtOutput<<Location_tobesaved;
    myfile.close();
    QMessageBox::critical(this,"提示","文件保存成功！");
}

void MainWindow::on_checkMoveDetect_stateChanged(int arg1)
{
    if(arg1){
        ui->ButtonSave->setEnabled(true);
    }else{
        ui->ButtonSave->setEnabled(false);
    }
}
