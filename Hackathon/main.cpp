#include <iostream>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <fstream>
#include "opencv2/imgproc/imgproc.hpp"
#include "constants.h"
#include <sys/stat.h>

using namespace std;
using namespace cv;


#define DEBUG 0
#define CLUSTERING 1

Rect screen;

typedef struct {
    Point CenterPointOfEyes;
    Point OffsetFromEyeCenter;
    int eyeLeftMax=0;//13;
    int eyeRightMax=0;//13;
    int eyeTopMax=0;//11;
    int eyeBottomMax=0;//11;
    int count = 0;
} EyeSettingsSt;
EyeSettingsSt EyeSettings;

vector<string> &split(const string &s, char delim, vector<string> &elems) {
    stringstream ss(s);
    string item;
    while (getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}

vector<string> split(const string &s, char delim) {
    vector<string> elems;
    split(s, delim, elems);
    return elems;
}

void scale(const Mat &src,Mat &dst) {
    cv::resize(src, dst, cv::Size(kFastEyeWidth,(((float)kFastEyeWidth)/src.cols) * src.rows));
}

Point unscale_point(Point p, Rect origSize) {
    float ratio = (((float)kFastEyeWidth)/origSize.width);
    int x = round(p.x / ratio);
    int y = round(p.y / ratio);
    return Point(x,y);
}

Mat matrix_magnitude(Mat mat_x, Mat mat_y) {
    Mat mag(mat_x.rows, mat_x.cols, CV_64F);

    for (int y = 0; y < mat_x.rows; y++) {
        const double *x_row = mat_x.ptr<double>(y), *y_row = mat_y.ptr<double>(y);
        double *mag_row = mag.ptr<double>(y);
        for (int x = 0; x < mat_x.cols; x++) {
            double gx = x_row[x], gy = y_row[x];
            double magnitude = sqrt((gx * gx) + (gy * gy));
            mag_row[x] = magnitude;
        }
    }
    return mag;
}


/*
 * Find possible center in gradient location
 * doesn't use the postprocessing weight of color (section 2.1)
 */
void possible_centers(int x, int y, const Mat &blurred, double gx, double gy, Mat &output) {

    for (int cy = 0; cy < output.rows; cy++) {
        double *output_row = output.ptr<double>(cy);
        for (int cx = 0; cx < output.cols; cx++) {
            if (x == cx && y == cy) {
                continue;
            }
            // equation (2)
            double dx = x - cx;
            double dy = y - cy;
            double magnitude = sqrt((dx * dx) + (dy * dy));
            dx = dx / magnitude;
            dy = dy / magnitude;

            double dotProduct = (dx*gx + dy*gy);
            // ignores vectors pointing in opposite direction/negative dot products
            dotProduct = max(0.0, dotProduct);

            // summation
            output_row[cx] += dotProduct * dotProduct;
        }
    }
}

/*
 * Imitate Matlab gradiant function, to better match results from paper
 */
Mat computeMatXGradient(const Mat &mat) {
    Mat out(mat.rows,mat.cols,CV_64F);

    for (int y = 0; y < mat.rows; ++y) {
        const uchar *Mr = mat.ptr<uchar>(y);
        double *Or = out.ptr<double>(y);

        Or[0] = Mr[1] - Mr[0];
        for (int x = 1; x < mat.cols - 1; ++x) {
            Or[x] = (Mr[x+1] - Mr[x-1])/2.0;
        }
        Or[mat.cols-1] = Mr[mat.cols-1] - Mr[mat.cols-2];
    }

    return out;
}

/*
 * Finds the pupils within the given eye region
 * returns points of where pupil is calculated to be
 *
 * face_image: image of face region from frame
 * eye_region: dimensions of eye region
 * window_name: display window name
 */
Point find_centers(Mat face_image, Rect eye_region) {

    Mat eye_unscaled = face_image(eye_region);

    // scale and grey image
    Mat eye_scaled_gray;
    scale(eye_unscaled, eye_scaled_gray);
    cvtColor(eye_scaled_gray, eye_scaled_gray, COLOR_BGRA2GRAY);

    // get the gradient of eye regions
    Mat gradient_x, gradient_y;
    gradient_x = computeMatXGradient(eye_scaled_gray);
    //Sobel(eye_scaled_gray, gradient_x, CV_64F, 1, 0, 5);
    gradient_y = computeMatXGradient(eye_scaled_gray.t()).t();
    //Sobel(eye_scaled_gray, gradient_y, CV_64F, 0, 1, 5);

    //Mat magnitude = matrix_magnitude(gradient_x, gradient_y);

    // normalized displacement vectors
    normalize(gradient_x, gradient_x);
    normalize(gradient_y, gradient_y);

    // blur and invert the image
    Mat blurred;
    GaussianBlur(eye_scaled_gray, blurred, Size(5, 5), 0, 0);
    bitwise_not(blurred, blurred);
    //increase contrast and decrease brightness
    for( int y = 0; y < blurred.rows; y++ )
    {
        for( int x = 0; x < blurred.cols; x++ )
        {
            for( int c = 0; c < 3; c++ )
            {
                blurred.at<Vec3b>(y,x)[c] = saturate_cast<uchar>(1.01*( blurred.at<Vec3b>(y,x)[c] ) - 10 );
            }
        }
    }

    //imshow("window", blurred);

    Mat outSum = Mat::zeros(eye_scaled_gray.rows, eye_scaled_gray.cols, CV_64F);

    for (int y = 0; y < blurred.rows; y++) {
        const double *x_row = gradient_x.ptr<double>(y), *y_row = gradient_y.ptr<double>(y);
        for (int x = 0; x < blurred.cols; x++) {
            double gx = x_row[x], gy = y_row[x];
            if (gx == 0.0 && gy == 0.0) {
                continue;
            }
            possible_centers(x, y, blurred, gx, gy, outSum);
        }
    }

    double numGradients = (blurred.rows*blurred.cols);
    Mat out;
    outSum.convertTo(out, CV_32F, 1.0/numGradients);

    Point max_point;
    double max_value;
    minMaxLoc(out, NULL, &max_value, NULL, &max_point);

    Point pupil = unscale_point(max_point, eye_region);
    return pupil;
}

/*
 * returns an array of points of the pupils
 * [left pupil, right pupil]
 *
 * color_image: image of the whole frame
 * face: dimensions of face in color_image
 */
void find_eyes(Mat color_image, Rect face, Point &left_pupil_dst, Point &right_pupil_dst, Rect &left_eye_region_dst, Rect &right_eye_region_dst) {
    // image of face
    Mat face_image = color_image(face);

    int eye_width = face.width * (kEyePercentWidth/100.0);
    int eye_height = face.height * (kEyePercentHeight/100.0);
    int eye_top = face.height * (kEyePercentTop/100.0);
    int eye_side = face.width * (kEyePercentSide/100.0);
    int right_eye_x = face.width - eye_width -  eye_side;

    // eye regions
    Rect left_eye_region(eye_side, eye_top, eye_width, eye_height);
    Rect right_eye_region(right_eye_x, eye_top, eye_width, eye_height);

    // get points of pupils within eye region
    Point left_pupil = find_centers(face_image, left_eye_region);
    Point right_pupil = find_centers(face_image, right_eye_region);

    // convert points to fit on frame image
    right_pupil.x += right_eye_region.x;
    right_pupil.y += right_eye_region.y;
    left_pupil.x += left_eye_region.x;
    left_pupil.y += left_eye_region.y;


    left_pupil_dst = left_pupil;
    right_pupil_dst = right_pupil;
    left_eye_region_dst = left_eye_region;
    right_eye_region_dst = right_eye_region;
}

void display_eyes(Mat color_image, Rect face, Point left_pupil, Point right_pupil, Rect left_eye_region, Rect right_eye_region, int record = 0, bool doCalibration = false) {
    Mat face_image = color_image(face);

    // draw eye regions
    rectangle(face_image, left_eye_region, Scalar(0, 0, 255));
    rectangle(face_image, right_eye_region, Scalar(0, 0, 255));

    //find eye center
    Point center;
    center.x = (right_pupil.x - left_pupil.x)/2 + left_pupil.x;
    center.y = (right_pupil.y + left_pupil.y)/2;

    // draw pupils
    circle(face_image, right_pupil, 3, Scalar(0, 255, 0));
    circle(face_image, left_pupil, 3, Scalar(0, 255, 0));
    //circle(face_image, center, 3, Scalar(255, 0, 0));

    std::string left_eye_region_width = std::to_string(left_eye_region.x + (left_eye_region.width/2));
    std::string left_eye_region_height = std::to_string(left_eye_region.y + (left_eye_region.height/2));

    std::string right_eye_region_width = std::to_string(right_eye_region.x + (right_eye_region.width/2));
    std::string right_eye_region_height = std::to_string(right_eye_region.y + (right_eye_region.height/2));

    std::string xleft_pupil_string = std::to_string(left_pupil.x);
    std::string yleft_pupil_string = std::to_string(left_pupil.y);

    std::string xright_pupil_string = std::to_string(right_pupil.x);
    std::string yright_pupil_string = std::to_string(right_pupil.y);

    String text1 = "Pupil(L,R): ([" + xleft_pupil_string + "," + yleft_pupil_string + "],[" + xright_pupil_string + "," + yright_pupil_string + "])";
    String text2 = "Center(L,R): ([" + left_eye_region_width + ", " + left_eye_region_height +"]," + "[" + right_eye_region_width + ", " + right_eye_region_height +"])";

    if (doCalibration && record) {
        cout << xleft_pupil_string << "," << yleft_pupil_string << ";"
             << xright_pupil_string << "," << yright_pupil_string << ";"
             << left_eye_region_width << "," << left_eye_region_height << ";"
             << right_eye_region_width << "," << right_eye_region_height << ";";
    }

    //add data
    putText (color_image, text1 + " " + text2, cvPoint(20,700), FONT_HERSHEY_SIMPLEX, double(1), Scalar(255,0,0));
}

void display_googley_eyes(Mat color_image, Rect face, Point left_pupil, Point right_pupil, Rect left_eye_region, Rect right_eye_region) {
    Mat face_image = color_image(face);

    // draw eye regions
    Point left_eye = Point(left_eye_region.x + left_eye_region.width / 2, left_eye_region.y + left_eye_region.height / 2);
    Point right_eye = Point(right_eye_region.x + right_eye_region.width / 2, right_eye_region.y + right_eye_region.height / 2);

    circle(face_image, left_eye, 50, Scalar(255, 255, 255), -1);
    circle(face_image, right_eye, 50, Scalar(255, 255, 255), -1);

    // draw pupils
    circle(face_image, right_pupil, 25, Scalar(0, 0, 0), -1);
    circle(face_image, left_pupil, 25, Scalar(0, 0, 0), -1);

}

Point closestPoint(vector<Point> shapes, Point guess){
    int best_dist = -1;
    Point best_point;

    for (Point s: shapes) {
        int dist = pow(s.x - guess.x, 2) + pow(s.y - guess.y, 2);
        if (best_dist == -1 or dist < best_dist) {
            best_dist = dist;
            best_point = s;
        }
    }
    return best_point;
}

void display_shapes_on_screen(Mat background, vector<Point> shapes, Point guess, unsigned char showGuess) {
    background.setTo(cv::Scalar(255,255,255));

    Point best_point = closestPoint(shapes, guess);

    for (Point s : shapes) {
        if (s == best_point) {
            circle(background, s, 20, Scalar(0,255,0), -1);
            circle(background, s, 20, Scalar(0,0,0), 2);
        } else {
            circle(background, s, 20, Scalar(0,0,255), -1);
            circle(background, s, 20, Scalar(0,0,0), 2);
        }
    }
    if(showGuess==1) {
        circle(background, guess, 5, Scalar(0, 0, 255), -1);
    }else if(showGuess==2){
        circle(background, guess, 5, Scalar(0, 0, 0), -1);
    }


    //display calibration points
    if(EyeSettings.eyeTopMax) {
        circle(background, Point(background.cols / 2, 0), 5, Scalar(0, 255,0), -1);
    }else{
        circle(background, Point(background.cols / 2, 0), 5, Scalar(0, 0, 255), -1);
    }
    if(EyeSettings.eyeRightMax) {
        circle(background, Point(background.cols, background.rows / 2), 5, Scalar(0, 255,0), -1);
    }else{
        circle(background, Point(background.cols, background.rows / 2), 5, Scalar(0, 0,255), -1);
    }
    if(EyeSettings.eyeBottomMax) {
        circle(background, Point(background.cols / 2, background.rows), 5, Scalar(0, 255,0), -1);
    }else{
        circle(background, Point(background.cols / 2, background.rows), 5, Scalar(0, 0,255), -1);
    }
    if(EyeSettings.eyeLeftMax) {
        circle(background, Point(0, background.rows / 2), 5, Scalar(0, 255,0), -1);
    }else{
        circle(background, Point(0, background.rows / 2), 5, Scalar(0, 0,255), -1);
    }
}

void ListenForCalibrate(int wait_key, Mat frame) {
    //left calibration 97
    //right calibration 100
    //bottom calibration 115
    //top calibration 119
    switch (wait_key) {
        case 97:
            EyeSettings.eyeLeftMax = abs(EyeSettings.OffsetFromEyeCenter.x);
            #if DEBUG
            imwrite("test/calib-left.png", frame);
            #endif
            break;
        case 100:
            EyeSettings.eyeRightMax = abs(EyeSettings.OffsetFromEyeCenter.x);
            #if DEBUG
            imwrite("test/calib-right.png", frame);
            #endif
            break;
        case 115:
            EyeSettings.eyeBottomMax = abs(EyeSettings.OffsetFromEyeCenter.y);
            #if DEBUG
            imwrite("test/calib-bot.png", frame);
            #endif
            break;
        case 119:
            EyeSettings.eyeTopMax = abs(EyeSettings.OffsetFromEyeCenter.y);
            #if DEBUG
            imwrite("test/calib-top.png", frame);
            #endif
            break;
    }
}

void cluster_image(Mat shapes_image, vector<Point> region_centers, Point &point_dst) {
    // clear original image
    shapes_image.setTo(cv::Scalar(255,255,255));
    //choose random region
    static int index = 0;
//    int region = rand() % region_centers.size();
    Point point = region_centers[index];
    circle(shapes_image, point, 20, Scalar(0,255,0), -1);
    point_dst = point;
    index++;
    if (index == region_centers.size())  {
        index = 0;
    }
}

vector<Point> find_regions_centers(Mat shapes_image, int x_regions, int y_regions) {
    vector<Point> regions_centers;
    int region_width = shapes_image.cols / x_regions;
    int region_height = shapes_image.rows / y_regions;
    int start_center_x = region_width/2;
    int start_center_y = region_height/2;
    int curr_x = 0, curr_y = 0;

    for (int x = 0; x < x_regions; x++) {
        curr_x = start_center_x + ((x) * region_width);

        for(int y = 0; y < y_regions; y++) {
            curr_y = start_center_y + ((y) * region_height);
            Point center(curr_x, curr_y);
            regions_centers.push_back(center);
            //cout << "center: " << center << endl;
        }
    }
    return regions_centers;
}


int main(int argc, char* argv[]) {
    bool doImport = false;
    bool doExport = false;
    bool doCalibrate = true;
    bool doTest = false;
    bool doTrain = false;
    bool doGoogle = false ;
    bool showCam = false;
    bool hasFile = false;
    int shapes_x = -1;
    int shapes_y = -1;
    fstream file;

    for(int i = 1; i < argc; i++) {
        if (string("-").compare(string(argv[i]).substr(0,1)) == 0) {
            if (string("--import").compare(argv[i]) == 0 || string("-i").compare(argv[i]) == 0) {
                doImport = true;
            } else if (string("--export").compare(argv[i]) == 0 || string("-e").compare(argv[i]) == 0) {
                doExport = true;
            } else if (string("--calibrate").compare(argv[i]) == 0 || string("-c").compare(argv[i]) == 0) {
                doCalibrate = true;
            } else if (string("--train").compare(argv[i]) == 0 || string("-T").compare(argv[i]) == 0) {
                doTrain = true;
            } else if (string("--test").compare(argv[i]) == 0 || string("-t").compare(argv[i]) == 0) {
                doTest = true;
            } else if (string("--show-cam").compare(argv[i]) == 0 || string("-w").compare(argv[i]) == 0) {
                doTrain = true;
            } else if (string("--file-name").compare(argv[i]) == 0 || string("-f").compare(argv[i]) == 0 || string("-F").compare(argv[i]) == 0) {
                if (i+1 < argc) {
                    struct stat buffer;
                    if (stat (string(argv[i+1]).c_str(), &buffer) != 0 || string("-F").compare(argv[i]) == 0                                           ) {
                        file.open(argv[i + 1]);
                        if (!file) {
                            cerr << "Failed to open <" << argv[i] << ">!";
                            exit(1);
                        } else {
                            hasFile = true;
                            i++;
                        }
                    } else {
                        cerr << "ERROR: File <" << argv[i +1] << "> already exists!";
                        exit(1);
                    }
                } else {
                    cerr << "ERROR: please enter a file name!";
                    exit(1);
                }
            } else {
                cerr << "ERROR: No argument <" << argv[i] << "> exists!";
                exit(1);
            }
        } else if (i+2 == argc){
            shapes_x = atoi(argv[i]);
            shapes_y = atoi(argv[i+1]);
            break;
        } else {
            cerr << "ERROR: Incorrect number of arguments!\n" <<
                    "Syntax main [--import|-i] [--export|-e] [--calibrate|-c] [--test| -T] [--train|-t] " <<
                    "[--show-cam|-w] [--filename|-f CALIBRATION_FILE] [SHAPES_X SHAPES_Y]";
            exit(1);
        }
    }
    if (showCam + doTrain + doTest > 1) {
        cerr << "You cannot show the camera or train or test at the same time! (Mutually exclusive)";
        exit(1);
    }

    if ((doImport || doExport) && !hasFile) {
        cerr << "You must define a file! -f <FILENAME>";
    }

    string line;
    if(doImport) {
            while(getline(file, line)) {
                vector<string> esArgs = split(line, ';');
                if (esArgs.size() != 7) {
                    cerr << "ERROR: Malformed file imported!";
                    exit(1);
                }

                vector<string> cpArgs = split(esArgs[0], ',');
                EyeSettings.CenterPointOfEyes = Point(atoi(cpArgs[0].c_str()), atoi(cpArgs[1].c_str()));
                vector<string> ocArgs = split(esArgs[1], ',');
                EyeSettings.OffsetFromEyeCenter = Point(atoi(ocArgs[0].c_str()), atoi(ocArgs[1].c_str()));
                EyeSettings.eyeLeftMax = atoi(esArgs[2].c_str());
                EyeSettings.eyeRightMax = atoi(esArgs[3].c_str());
                EyeSettings.eyeTopMax = atoi(esArgs[4].c_str());
                EyeSettings.eyeBottomMax = atoi(esArgs[5].c_str());
                EyeSettings.count = atoi(esArgs[6].c_str());
            }
        }

    const int height = 800;
    const int width = 1440;

    //define font
    CvFont font;
    double hScale=1.0;
    double vScale=1.0;
    int    lineWidth=6;
    cvInitFont(&font,CV_FONT_HERSHEY_SIMPLEX|CV_FONT_ITALIC, hScale,vScale,0,lineWidth);

    CascadeClassifier face_cascade;
    face_cascade.load("haar_data/haarcascade_frontalface_alt.xml");

    VideoCapture cap(0);
    if (!cap.isOpened()) {
        return -1;
    }

    namedWindow("window");
    Mat frame, shape_screen;
    shape_screen = Mat(height,width, CV_8UC3);
    cap >> frame;

    vector<Point> region_centers = find_regions_centers(shape_screen, shapes_x, shapes_y);
    //random_shuffle(region_centers.begin(), region_centers.end());

    int count = 0;
    int record = 0;
    int currentShape=-1;
    while (1) {
        Mat gray_image;
        vector<Rect> faces;
        cvtColor(frame, gray_image, COLOR_BGRA2GRAY);

        face_cascade.detectMultiScale(gray_image, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE|CV_HAAR_FIND_BIGGEST_OBJECT);

        Point left_pupil, right_pupil;
        Rect left_eye, right_eye;
        if (faces.size() > 0) {
            find_eyes(frame, faces[0], left_pupil, right_pupil, left_eye, right_eye);
            display_eyes(frame, faces[0], left_pupil, right_pupil, left_eye, right_eye);
        }

        // if 'q' is tapped, exit
        int wait_key = waitKey(8);
        if (wait_key == 113) {
            break;
        }

        if (wait_key == 103) {
            doGoogle = !doGoogle;
        }

        EyeSettings.CenterPointOfEyes.x = ((right_eye.x + right_eye.width/2) + (left_eye.x + left_eye.width/2))/2;
        EyeSettings.CenterPointOfEyes.y = ((right_eye.y + right_eye.height/2) + (left_eye.y + left_eye.height/2))/2;

        EyeSettings.OffsetFromEyeCenter.x = EyeSettings.CenterPointOfEyes.x - (right_pupil.x + left_pupil.x)/2;
        EyeSettings.OffsetFromEyeCenter.y = EyeSettings.CenterPointOfEyes.y - (right_pupil.y + left_pupil.y)/2;

        ListenForCalibrate(wait_key, frame);

        //space for test
        if(wait_key == 32)
        {
            doCalibrate = false;
            if(doExport) {
                file << to_string(EyeSettings.CenterPointOfEyes.x) << "," <<
                                to_string(EyeSettings.CenterPointOfEyes.y) << ";";
                file << to_string(EyeSettings.OffsetFromEyeCenter.x) << "," <<
                                to_string(EyeSettings.OffsetFromEyeCenter.y) << ";";
                file << to_string(EyeSettings.eyeLeftMax) << ";";
                file << to_string(EyeSettings.eyeRightMax) << ";";
                file << to_string(EyeSettings.eyeTopMax) << ";";
                file << to_string(EyeSettings.eyeBottomMax) << ";";
                file << to_string(EyeSettings.count) << ";";
                file.close();
            }
        }

        if (!doCalibrate) {
            double pupilOffsetfromLeft = EyeSettings.OffsetFromEyeCenter.x+EyeSettings.eyeLeftMax;
            double pupilOffsetfromBottom = EyeSettings.OffsetFromEyeCenter.y+EyeSettings.eyeBottomMax;

            double percentageWidth = pupilOffsetfromLeft / (double)(EyeSettings.eyeLeftMax + EyeSettings.eyeRightMax);
            if(percentageWidth < 0){
                percentageWidth = 0;
            }else if(percentageWidth > 1){
                percentageWidth = 1;
            }
            double percentageHeight = pupilOffsetfromBottom / (double)(EyeSettings.eyeTopMax + EyeSettings.eyeBottomMax);
            if(percentageHeight < 0){
                percentageHeight = 0;
            }else if(percentageHeight > 1){
                percentageHeight = 1;
            }

            #if DEBUG
            cout << "xmax: " << (EyeSettings.eyeLeftMax + EyeSettings.eyeRightMax) << " cur: " << pupilOffsetfromLeft << " = "<< percentageWidth << " , "
                 << "ymax: " << (EyeSettings.eyeTopMax + EyeSettings.eyeBottomMax) << " cur: " << pupilOffsetfromBottom << " = "<< percentageHeight << endl;
            //draw expected position on screen from pupils
            circle(frame, Point(
                           (frame.cols*(percentageWidth)),
                           (frame.rows*(1-percentageHeight))),
                   5, Scalar(255, 255, 0), -1);

            Point pupilCenter = Point((right_pupil.x + left_pupil.x)/2, (right_pupil.y + left_pupil.y)/2);
            //draw pupil position
            circle(frame, Point(
                           pupilCenter.x + faces[0].x,
                           pupilCenter.y + faces[0].y),
                   3, Scalar(255, 0, 0), -1);
            //draw pupil bounding box from config
            rectangle(frame,
                      Rect(
                              EyeSettings.CenterPointOfEyes.x - EyeSettings.eyeRightMax + faces[0].x,
                              EyeSettings.CenterPointOfEyes.y - EyeSettings.eyeBottomMax + faces[0].y,
                              (EyeSettings.eyeLeftMax + EyeSettings.eyeRightMax),
                              (EyeSettings.eyeTopMax + EyeSettings.eyeBottomMax)
                      ),Scalar(255,255,0), 1);
            //draw eye center
            Point drawEyeCenter = Point(EyeSettings.CenterPointOfEyes.x + faces[0].x,
                                        EyeSettings.CenterPointOfEyes.y + faces[0].y);
            circle(frame, drawEyeCenter, 3, Scalar(0, 0, 255));

            //imwrite(("test/test"+std::to_string(EyeSettings.count)+".png"), shape_screen);
            //imwrite(("test/testcolor"+std::to_string(EyeSettings.count)+".png"), frame);
            EyeSettings.count++;
            imshow("window", frame);
            #else
            if (doGoogle) {
                display_googley_eyes(frame, faces[0], left_pupil, right_pupil, left_eye, right_eye);
                imshow("window", frame);
            } else {
                display_shapes_on_screen(shape_screen, region_centers,
                                         Point(frame.cols * percentageWidth, frame.rows * (1 - percentageHeight)),
                                         (faces.size() > 0 ? 2 : 1));
                imshow("window", shape_screen);
            }

            #endif

            #if CLUSTERING
            //looking at new point, start recording data 'z'
            if (wait_key == 122 && currentShape<(int)region_centers.size()) {
                record=1;
                currentShape++;
                cout << "Record data for grid area " << currentShape << endl;
            }
            if(record < 20 && record > 0){
                //actual sphere looking at, sphere it thinks we're looking at, exact screen point thinks looking at
                cout << region_centers[currentShape] << ","
                << closestPoint(region_centers, Point(frame.cols*percentageWidth, frame.rows*(1-percentageHeight))) << ","
                <<  Point(frame.cols*percentageWidth, frame.rows*(1-percentageHeight)) << endl;
                circle(shape_screen, region_centers[currentShape], 4, Scalar(0,0,0), -1);
                imshow("window", shape_screen);

                record++;
            }
            #endif
        }

        if(doCalibrate && DEBUG) {
            imshow("window", frame);
        }
        if(doCalibrate && !DEBUG){
            display_shapes_on_screen(shape_screen, region_centers, Point(), 0);
            imshow("window", shape_screen);
        }

        cap >> frame;
    }

    return 0;
}