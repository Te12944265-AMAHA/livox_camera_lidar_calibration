#include <iostream>
#include <sstream>
#include <time.h>
#include <stdio.h>
#include <fstream>
#include <ros/ros.h>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

string camera_in_path, camera_folder_path, result_path;
int row_number, col_number, width, height;
int isFisheye;

void getParameters() {
    cout << "Get the parameters from the launch file" << endl;

    if (!ros::param::get("camera_in_path", camera_in_path)) {
        cout << "Cannot get the value of camera_in_path" << endl;
        exit(1);
    }
    if (!ros::param::get("camera_folder_path", camera_folder_path)) {
        cout << "Cannot get the value of camera_folder_path" << endl;
        exit(1);
    }
    if (!ros::param::get("result_path", result_path)) {
        cout << "Cannot get the value of result_path" << endl;
        exit(1);
    }
    if (!ros::param::get("row_number", row_number)) {
        cout << "Cannot get the value of row_number" << endl;
        exit(1);
    }
    if (!ros::param::get("col_number", col_number)) {
        cout << "Cannot get the value of col_number" << endl;
        exit(1);
    }
    if (!ros::param::get("width", width)) {
        cout << "Cannot get the value of width" << endl;
        exit(1);
    }
    if (!ros::param::get("height", height)) {
        cout << "Cannot get the value of height" << endl;
        exit(1);
    }

	if (!ros::param::get("fisheye", isFisheye)) {
        cout << "Cannot get the value of fisheye" << endl;
        exit(1);
    }
}


int main(int argc, char **argv) {
	ros::init(argc, argv, "cameraCalib");
	getParameters();

	ifstream fin(camera_in_path);  
	ofstream fout(result_path);   
	int image_count = 0;  
	Size image_size;     
	Size board_size = Size(row_number, col_number); 
	vector<Point2f> image_points_buf;        
	vector<vector<Point2f>> image_points_seq; 
	string filename; 
	vector<string> filenames; // filenames of images whose corners can be found

	// try finding checkerboard in all input images, if found append to filenames, else discard
	while (getline(fin, filename) && filename.size() > 1) {
		filename = camera_folder_path + filename;
		cout << filename << endl;
		Mat imageInput = imread(filename);
		if (imageInput.empty()) {  // use the file name to search the photo
			cout << "**" << filename << "** doesn't exist!\n";
        	continue;
    	}

		if (image_count == 0) {
			image_size.width = imageInput.cols;
			image_size.height = imageInput.rows;
		}

		// extract corners
		if (0 == findChessboardCorners(imageInput, board_size, image_points_buf)) {
			cout << "**" << filename << "** cannot find checkerboard corners!\n";
			continue;
		}
		else {
			image_count++;
			filenames.push_back(filename);
			Mat view_gray;
			cvtColor(imageInput, view_gray, cv::COLOR_RGB2GRAY);

			// refine the corners to subpixel accuracy
			cornerSubPix(view_gray, image_points_buf, Size(5, 5), Size(-1, -1), TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));

			image_points_seq.push_back(image_points_buf); 

			// display corners
			Mat gray_RGB;
			cvtColor(view_gray, gray_RGB, cv::COLOR_GRAY2RGB);
			drawChessboardCorners(gray_RGB, board_size, image_points_buf, true);
			cv::resize(gray_RGB, gray_RGB, cv::Size(), 0.3, 0.3);
			imshow("Camera Calibration", gray_RGB); 

			waitKey(500);    
		}
	}

	if (image_count == 0){
		cout << "Not enough data!" << endl;
		exit(1);
	}
	
	//-------------camera intrinsics calibration------------------

	Size square_size = Size(width, height);  
	vector<vector<Point3f>> object_points; 

	Mat cameraMatrix = Mat(3, 3, CV_32FC1, Scalar::all(0));
	vector<int> point_counts; 
	Mat distCoeffs = Mat(1, 5, CV_32FC1, Scalar::all(0));       // k1,k2,p1,p2,k3
	Mat distCoeffs_fisheye = Mat(1, 4, CV_32FC1, Scalar::all(0));  // θ_distorted = θ*(1+k1*θ^2+k2*θ^4+k3*θ^6+k4*θ^8) 
	vector<Mat> tvecsMat;
	vector<Mat> rvecsMat;

	int i, j, t;
	for (t = 0; t<image_count; t++) {
		vector<Point3f> tempPointSet;
		for (i = 0; i<board_size.height; i++) {
			for (j = 0; j<board_size.width; j++) {
				Point3f realPoint;
				// in the checkerboard frame, x axis is parallel with board height, y is parallel with board width
				// upper left corner is the origin
				realPoint.x = i * square_size.height;
				realPoint.y = j * square_size.width;
				realPoint.z = 0;
				tempPointSet.push_back(realPoint);
			}
		}
		object_points.push_back(tempPointSet);
	}

	for (i = 0; i<image_count; i++) {
		point_counts.push_back(board_size.width * board_size.height);
	}

	if (!isFisheye)
		calibrateCamera(object_points, image_points_seq, image_size, cameraMatrix, distCoeffs, rvecsMat, tvecsMat, 0);
	else
		fisheye::calibrate(object_points, image_points_seq, image_size, cameraMatrix, distCoeffs_fisheye, rvecsMat, tvecsMat, fisheye::CALIB_RECOMPUTE_EXTRINSIC);


	// -------------------evaluation------------------------------

	double total_err = 0.0; 
	double err = 0.0; 
	vector<Point2f> image_points2; 
	fout << "Average error: \n";
	ifstream fin2(camera_in_path); 
	for (i = 0; i < filenames.size(); i++){
		string fname = filenames[i];
		Mat imageInput = imread(fname);
		const char delim = '/';
		vector<string> tok_out;
		stringstream ss(fname); 
		string temp_str;
		while(getline(ss, temp_str, delim)){ //use comma as delim for cutting string
			tok_out.push_back(temp_str);
		}
		fname = tok_out.back();
		cout << "image: " << fname << endl;
		Mat view_gray;
		Mat undistorted_gray, img_buf2;
		cvtColor(imageInput, view_gray, cv::COLOR_RGB2GRAY);

		vector<Point3f> tempPointSet = object_points[i];

		if (!isFisheye)
			projectPoints(tempPointSet, rvecsMat[i], tvecsMat[i], cameraMatrix, distCoeffs, image_points2);
		else
			fisheye::projectPoints(tempPointSet, image_points2, rvecsMat[i], tvecsMat[i], cameraMatrix, distCoeffs_fisheye);

		vector<Point2f> tempImagePoint = image_points_seq[i];
		Mat tempImagePointMat = Mat(1, tempImagePoint.size(), CV_32FC2);
		Mat image_points2Mat = Mat(1, image_points2.size(), CV_32FC2);

		for (unsigned int j = 0; j < tempImagePoint.size(); j++) {
			image_points2Mat.at<Vec2f>(0, j) = Vec2f(image_points2[j].x, image_points2[j].y);
			tempImagePointMat.at<Vec2f>(0, j) = Vec2f(tempImagePoint[j].x, tempImagePoint[j].y);
		}
		err = norm(image_points2Mat, tempImagePointMat, NORM_L2);
		total_err += err /= point_counts[i];
		fout << "The error of " << fname << " is " << err << " pixel" << endl;

		if (!isFisheye)
			undistort(view_gray, undistorted_gray, cameraMatrix, distCoeffs);
		else
			fisheye::undistortImage(view_gray, undistorted_gray, cameraMatrix, distCoeffs_fisheye);
		cv::resize(view_gray, view_gray, cv::Size(), 0.3, 0.3);
		cv::resize(undistorted_gray, undistorted_gray, cv::Size(), 0.3, 0.3);
		hconcat(view_gray, undistorted_gray, img_buf2);
		imshow("Distorted vs Undistorted Images", img_buf2);
		waitKey(1000); 
	}
	fout << "Overall average error is: " << total_err / image_count << " pixel" << endl << endl;
	cout << "Overall average error is: " << total_err / image_count << " pixel" << endl << endl;


	//-----------------------save data------------------------------------------- 
	Mat rotation_matrix = Mat(3, 3, CV_32FC1, Scalar::all(0));
	fout << "Intrinsic: " << endl;
	fout << cameraMatrix << endl << endl;
	cout << "Intrinsic: " << endl;
	cout << cameraMatrix << endl << endl;
	if (!isFisheye) {
		fout << "Regular distortion parameters: " << endl;
		fout << distCoeffs << endl << endl << endl;
		cout << "Regular distortion parameters: " << endl;
		cout << distCoeffs << endl << endl << endl;
	} else {
		fout << "Fisheye distortion parameters: " << endl;
		fout << distCoeffs_fisheye << endl << endl << endl;
		cout << "Fisheye distortion parameters: " << endl;
		cout << distCoeffs_fisheye << endl << endl << endl;
	}
	cout << "Results saved." << endl;

	fin.close();
	fout.close();
	return 0;
}