#pragma once

#include "ofMain.h"
#include "ofxTensorFlow2.h"

class ofApp : public ofBaseApp {
public:
    void setup() override;
    void update() override;
    void draw() override;

    ofVideoGrabber cam;
    ofPixels resizedPixels;
    ofImage depthImg;

    ofxTF2::Model model;
    
    // MiDaS v2.1 small standard resolution
    const int modelWidth = 256;
    const int modelHeight = 256;
};