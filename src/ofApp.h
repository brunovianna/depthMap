#pragma once

#include "ofMain.h"
#include "ofxTensorFlow2.h"

class ofApp : public ofBaseApp {
public:
    void setup() override;
    void update() override;
    void draw() override;
    void keyPressed(int key) override;

    // Optional function to build/update the 3D point cloud from depth data
    void updatePointCloud(const std::vector<float>& depthData, float minVal, float maxVal);

    ofVideoGrabber cam;
    ofPixels resizedPixels;
    ofImage depthImg;

    ofxTF2::Model model;

    // 3D Point Cloud & Camera
    ofEasyCam easyCam;
    ofVboMesh pointCloud;
    bool bShowPointCloud = true;
    float depthScale = 400.0f;
    float zOffset = 200.0f;

    // MiDaS v2.1 small standard resolution
    const int modelWidth = 256;
    const int modelHeight = 256;
};