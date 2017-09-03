#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "qxtspanslider.h"
#include <opencv2/opencv.hpp>
#include <QDirIterator>
#include <QLabel>
struct colorSpace {
    char const *name;
    int numColors;
    int range[3];
    int cvFlag;
};

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();
    QxtSpanSlider *slider1 ;
    QxtSpanSlider *slider2 ;
    QxtSpanSlider *slider3 ;
    QLabel *lbFileName;
    cv::Mat imgTransformMatrix,imgTransformMatrixInv;
    //cv::Mat cvImgRGB;
    cv::Mat cvImg; //Original image
    cv::Mat img; //Converted Image
    void setupDialer(colorSpace *cs);
    void changeFilter();
    QDirIterator *dirIter;
    void loadImg(QString fileName);
    void loadNext();
    colorSpace cs;
    void regenChannelsView();
public slots:
private slots:
    void on_slider_spanChanged(int lower, int upper);
    void on_actionOpen_triggered();

    void on_cbColors_activated(const QString &arg1);

    void on_cbColors_activated(int index);

    void on_pushButton_clicked();

private:
    Ui::MainWindow *ui;
};

#endif // MAINWINDOW_H
