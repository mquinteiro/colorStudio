#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QFileDialog>
#include <opencv2/opencv.hpp>
#include "qxtspanslider.h"
#include <QDirIterator>
cv::Mat get_hogdescriptor_visu(const cv::Mat& color_origImg, std::vector<float>& descriptorValues, const cv::Size & size )
{
    const int DIMX = size.width;
    const int DIMY = size.height;
    float zoomFac = 3;
    cv::Mat visu;
    cv::resize(color_origImg, visu, cv::Size( (int)(color_origImg.cols*zoomFac), (int)(color_origImg.rows*zoomFac) ) );

    int cellSize        = 8;
    int gradientBinSize = 9;
    float radRangeForOneBin = (float)(CV_PI/(float)gradientBinSize); // dividing 180 into 9 bins, how large (in rad) is one bin?

    // prepare data structure: 9 orientation / gradient strenghts for each cell
    int cells_in_x_dir = DIMX / cellSize;
    int cells_in_y_dir = DIMY / cellSize;
    float*** gradientStrengths = new float**[cells_in_y_dir];
    int** cellUpdateCounter   = new int*[cells_in_y_dir];
    for (int y=0; y<cells_in_y_dir; y++)
    {
        gradientStrengths[y] = new float*[cells_in_x_dir];
        cellUpdateCounter[y] = new int[cells_in_x_dir];
        for (int x=0; x<cells_in_x_dir; x++)
        {
            gradientStrengths[y][x] = new float[gradientBinSize];
            cellUpdateCounter[y][x] = 0;

            for (int bin=0; bin<gradientBinSize; bin++)
                gradientStrengths[y][x][bin] = 0.0;
        }
    }

    // nr of blocks = nr of cells - 1
    // since there is a new block on each cell (overlapping blocks!) but the last one
    int blocks_in_x_dir = cells_in_x_dir - 1;
    int blocks_in_y_dir = cells_in_y_dir - 1;

    // compute gradient strengths per cell
    int descriptorDataIdx = 0;
    int cellx = 0;
    int celly = 0;

    for (int blockx=0; blockx<blocks_in_x_dir; blockx++)
    {
        for (int blocky=0; blocky<blocks_in_y_dir; blocky++)
        {
            // 4 cells per block ...
            for (int cellNr=0; cellNr<4; cellNr++)
            {
                // compute corresponding cell nr
                cellx = blockx;
                celly = blocky;
                if (cellNr==1) celly++;
                if (cellNr==2) cellx++;
                if (cellNr==3)
                {
                    cellx++;
                    celly++;
                }

                for (int bin=0; bin<gradientBinSize; bin++)
                {
                    float gradientStrength = descriptorValues[ descriptorDataIdx ];
                    descriptorDataIdx++;

                    gradientStrengths[celly][cellx][bin] += gradientStrength;

                } // for (all bins)


                // note: overlapping blocks lead to multiple updates of this sum!
                // we therefore keep track how often a cell was updated,
                // to compute average gradient strengths
                cellUpdateCounter[celly][cellx]++;

            } // for (all cells)


        } // for (all block x pos)
    } // for (all block y pos)


    // compute average gradient strengths
    for (celly=0; celly<cells_in_y_dir; celly++)
    {
        for (cellx=0; cellx<cells_in_x_dir; cellx++)
        {

            float NrUpdatesForThisCell = (float)cellUpdateCounter[celly][cellx];

            // compute average gradient strenghts for each gradient bin direction
            for (int bin=0; bin<gradientBinSize; bin++)
            {
                gradientStrengths[celly][cellx][bin] /= NrUpdatesForThisCell;
            }
        }
    }

    // draw cells
    for (celly=0; celly<cells_in_y_dir; celly++)
    {
        for (cellx=0; cellx<cells_in_x_dir; cellx++)
        {
            int drawX = cellx * cellSize;
            int drawY = celly * cellSize;

            int mx = drawX + cellSize/2;
            int my = drawY + cellSize/2;

            cv::rectangle(visu, cv::Point((int)(drawX*zoomFac), (int)(drawY*zoomFac)), cv::Point((int)((drawX+cellSize)*zoomFac), (int)((drawY+cellSize)*zoomFac)), cv::Scalar(100,100,100), 1);

            // draw in each cell all 9 gradient strengths
            for (int bin=0; bin<gradientBinSize; bin++)
            {
                float currentGradStrength = gradientStrengths[celly][cellx][bin];

                // no line to draw?
                if (currentGradStrength==0)
                    continue;

                float currRad = bin * radRangeForOneBin + radRangeForOneBin/2;

                float dirVecX = cos( currRad );
                float dirVecY = sin( currRad );
                float maxVecLen = (float)(cellSize/2.f);
                float scale = 2.5; // just a visualization scale, to see the lines better

                // compute line coordinates
                float x1 = mx - dirVecX * currentGradStrength * maxVecLen * scale;
                float y1 = my - dirVecY * currentGradStrength * maxVecLen * scale;
                float x2 = mx + dirVecX * currentGradStrength * maxVecLen * scale;
                float y2 = my + dirVecY * currentGradStrength * maxVecLen * scale;

                // draw gradient visualization
                line(visu, cv::Point((int)(x1*zoomFac),(int)(y1*zoomFac)), cv::Point((int)(x2*zoomFac),(int)(y2*zoomFac)), cv::Scalar(0,255,0), 1);

            } // for (all bins)

        } // for (cellx)
    } // for (celly)


    // don't forget to free memory allocated by helper data structures!
    for (int y=0; y<cells_in_y_dir; y++)
    {
        for (int x=0; x<cells_in_x_dir; x++)
        {
            delete[] gradientStrengths[y][x];
        }
        delete[] gradientStrengths[y];
        delete[] cellUpdateCounter[y];
    }
    delete[] gradientStrengths;
    delete[] cellUpdateCounter;

    return visu;

} // get_hogdescriptor_visu

//["RGB", "HSV", "LUV", "HLS", "YUV", "YCrCb"]
colorSpace colorSpaces[]={{"BRG",3,{256,256,256},0},{"Lab",3,{256,256,256},CV_BGR2Lab},{"HSV",3,{256,256,256},CV_BGR2HSV},{"LUV",3,{256,256,256},CV_BGR2Luv},
    {"HLS",3,{256,256,256},CV_BGR2HLS},{"YUV",3,{256,256,256},CV_BGR2YUV},{"YCrCb",3,{256,256,256},CV_BGR2YCrCb},
    {"Gray",1,{256,0,0},CV_BGR2GRAY}};
void MainWindow::setupDialer(colorSpace *cs) {
    slider1->setRange(0,cs->range[0]-1);
    slider2->setRange(0,cs->range[1]-1);
    slider3->setRange(0,cs->range[2]-1);
    slider1->setUpperValue(cs->range[0]-1);
    slider2->setUpperValue(cs->range[1]-1);
    slider3->setUpperValue(cs->range[2]-1);
    slider1->setLowerValue(0);
    slider2->setLowerValue(0);
    slider3->setLowerValue(0);
}

void MainWindow::changeFilter()
{
    cv::Scalar low = cv::Scalar(slider1->lowerValue(),slider2->lowerValue(),slider3->lowerValue());
    cv::Scalar hi = cv::Scalar(slider1->upperValue(),slider2->upperValue(),slider3->upperValue());
    cv::Mat frame_threshold;
    cv::inRange(img,low, hi,frame_threshold);
    cv::imshow("Object filtered",frame_threshold);
}

void MainWindow::on_slider_spanChanged(int lower,int upper)
{
   changeFilter();
}

cv::Point2f orgPoints[]={{160,96},{245,96},{245,162},{160,162}};

cv::Point2f dstPoints[]={{42,42},{83,42},{83,83},{42,83}};

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    int element =0;
    ui->setupUi(this);
    ui->frame->hide();
    for(colorSpace &color : colorSpaces ) {
        ui->cbColors->addItem(color.name,element++);
    }
    cv::namedWindow("Original", cv::WINDOW_AUTOSIZE  );
    cs = colorSpaces[0];
    slider1 = new QxtSpanSlider(ui->frame);
    slider2 = new QxtSpanSlider(ui->frame);
    slider3 = new QxtSpanSlider(ui->frame);
    ui->gridLayout_2->addWidget(slider1, 2, 0, 1, 1);
    ui->gridLayout_2->addWidget(slider2, 2, 1, 1, 1);
    ui->gridLayout_2->addWidget(slider3, 2, 2, 1, 1);
    setupDialer(&colorSpaces[0]);
    connect(slider1,SIGNAL(spanChanged(int,int)),this,SLOT(on_slider_spanChanged(int,int)));
    connect(slider2,SIGNAL(spanChanged(int,int)),this,SLOT(on_slider_spanChanged(int,int)));
    connect(slider3,SIGNAL(spanChanged(int,int)),this,SLOT(on_slider_spanChanged(int,int)));
    lbFileName = new QLabel(ui->frame);
    lbFileName->setTextInteractionFlags(Qt::TextSelectableByMouse);
    ui->gridLayout_2->addWidget(lbFileName, 1, 2, 1, 1);
    imgTransformMatrix= cv::getPerspectiveTransform(orgPoints,dstPoints);

}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_actionOpen_triggered()
{
    QString fileName;
    QString dir = QFileDialog::getExistingDirectory(this, tr("Open Directory"),
                                                    "/home",
                                                    QFileDialog::ShowDirsOnly
                                                    | QFileDialog::DontResolveSymlinks);
    QStringList filters;
    filters << "*.png";
    dirIter = new QDirIterator(dir,filters);
    //fileName = QFileDialog::getOpenFileName(this, tr("Open Image"), "./", tr("Image Files (*.png *.jpg *.bmp)"));
    if(!dirIter->hasNext())
        return;

    loadNext();
}
void MainWindow::loadNext()
{
    if(!dirIter->hasNext())
        return;
    loadImg(dirIter->next());
}


void MainWindow::loadImg(QString fileName)
{
    if(fileName.length()>0) {
        lbFileName->setText(fileName);
        cv::Mat cvImgOring = cv::imread(fileName.toLatin1().data());
        cv::warpPerspective(cvImgOring,cvImg,imgTransformMatrix,cv::Size(84,84));
        cv::imshow("Original",cvImg);
        cv::HOGDescriptor d(cv::Size(84, 84),cv::Size(16,16), cv::Size(8,8),
                            cv::Size(8,8), 9);
        std::vector<float> descriptorsValues;
        std::vector<cv::Point> locations;
        d.compute(cvImg,descriptorsValues, cv::Size(0,0), cv::Size(0,0), locations);
        cv::waitKey(1);
        //        QImage img((uchar*)cvImgRGB.data, cvImgRGB.cols, cvImgRGB.rows, QImage::Format_RGB32);
        ui->frame->show();
        regenChannelsView();
    }
}

void MainWindow::on_cbColors_activated(const QString &arg1)
{

}

void MainWindow::regenChannelsView()
{
    cv::Mat treeImg;
    if(cs.cvFlag==0) {

        img = cvImg.clone();
    }
    else {
        cv::cvtColor(cvImg,img,cs.cvFlag);
    }
    cv::Mat bgr[3];   //destination array
    split(img,bgr);//split source
    treeImg.push_back(bgr[0]);
    treeImg.push_back(bgr[1]);
    treeImg.push_back(bgr[2]);
    cv::imshow("ColorSpace",treeImg);
    changeFilter();
}

void MainWindow::on_cbColors_activated(int index)
{
    cs = colorSpaces[index];
    regenChannelsView();

}

void MainWindow::on_pushButton_clicked()
{
    loadNext();

}
