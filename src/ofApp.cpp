#include "ofApp.h"

void ofApp::setup() {
    ofSetWindowTitle("ofxTensorFlow2 3D Depth Point Cloud");
    ofSetVerticalSync(true);

    ofxTF2::setGPUMaxMemory(ofxTF2::GPU_PERCENT_70);

    cam.setup(640, 480);

    resizedPixels.allocate(modelWidth, modelHeight, OF_PIXELS_RGB);
    depthImg.allocate(modelWidth, modelHeight, OF_IMAGE_GRAYSCALE);

    // Configure point cloud mesh
    pointCloud.setMode(OF_PRIMITIVE_POINTS);
    pointCloud.enableColors();
    glPointSize(3.0f); // Make points clearly visible

    // Position the 3D camera slightly back looking at the origin
    easyCam.setDistance(450);

    if (!model.load("model")) {
        ofLogError() << "Failed to load TensorFlow SavedModel!";
        std::exit(EXIT_FAILURE);
    }

    model.setup({"serving_default_x"}, {"StatefulPartitionedCall"});
}

void ofApp::update() {
    cam.update();

    if (cam.isFrameNew()) {
        // 1. Resize webcam feed to 256x256
        cam.getPixels().resizeTo(resizedPixels);

        // 2. Convert to float32 tensor normalized to [0.0, 1.0]
        cppflow::tensor input = ofxTF2::pixelsToTensor(resizedPixels);
        input = cppflow::cast(input, TF_UINT8, TF_FLOAT);
        input = cppflow::div(input, cppflow::tensor({255.0f}));
        input = cppflow::expand_dims(input, 0);

        // 3. Run inference
        cppflow::tensor output = model.runModel(input);
        std::vector<float> depthData = output.get_data<float>();

        // 4. Find min/max inverse depth values
        auto [minIt, maxIt] = std::minmax_element(depthData.begin(), depthData.end());
        float minVal = *minIt;
        float maxVal = *maxIt;
        float range = (maxVal - minVal > 1e-6f) ? (maxVal - minVal) : 1.0f;

        // Update 2D grayscale depth image
        ofPixels& outPixels = depthImg.getPixels();
        for (size_t i = 0; i < depthData.size(); ++i) {
            float normalized = (depthData[i] - minVal) / range;
            outPixels[i] = static_cast<unsigned char>(ofClamp(normalized * 255.0f, 0.0f, 255.0f));
        }
        depthImg.update();

        // 5. Optionally build the 3D point cloud
        if (bShowPointCloud) {
            updatePointCloud(depthData, minVal, maxVal);
        }
    }
}

void ofApp::updatePointCloud(const std::vector<float>& depthData, float minVal, float maxVal) {
    pointCloud.clear();

    float range = (maxVal - minVal > 1e-6f) ? (maxVal - minVal) : 1.0f;

    // Virtual pinhole focal length (approx. 60-degree horizontal FOV)
    float focalLength = modelWidth * 0.9f;
    float cx = modelWidth * 0.5f;
    float cy = modelHeight * 0.5f;

    for (int y = 0; y < modelHeight; ++y) {
        for (int x = 0; x < modelWidth; ++x) {
            int idx = y * modelWidth + x;

            // Normalize inverse depth [0 = far, 1 = close]
            float normInvDepth = ofClamp((depthData[idx] - minVal) / range, 0.0f, 1.0f);

            // Convert to forward Z distance (closer objects have smaller distance from camera)
            float z = zOffset + (1.0f - normInvDepth) * depthScale;

            // Pinhole back-projection into 3D world space (flip Y so up is positive in OF)
            float worldX = (x - cx) * z / focalLength;
            float worldY = -(y - cy) * z / focalLength;
            float worldZ = -z; // Push scene away from camera along -Z

            // Sample matching RGB color from the resized webcam frame
            ofColor color = resizedPixels.getColor(x, y);

            pointCloud.addVertex(glm::vec3(worldX, worldY, worldZ));
            pointCloud.addColor(color);
        }
    }
}

void ofApp::draw() {
    ofBackground(15);

    if (bShowPointCloud) {
        ofEnableDepthTest();
        easyCam.begin();

        // Center the point cloud around the origin for smooth orbiting
        ofPushMatrix();
        ofTranslate(0, 0, zOffset + depthScale * 0.5f);
        pointCloud.draw();
        ofPopMatrix();

        easyCam.end();
        ofDisableDepthTest();

        // Draw small 2D picture-in-picture previews in the bottom-left corner
        cam.draw(15, ofGetHeight() - 135, 160, 120);
        depthImg.draw(185, ofGetHeight() - 135, 160, 120);
    } else {
        // Standard 2D side-by-side view
        cam.draw(0, 0, 640, 480);
        depthImg.draw(640, 0, 640, 480);
    }

    // HUD instructions
    std::string hud = "FPS: " + ofToString(ofGetFrameRate(), 1) +
                      " | [SPACE] Toggle 3D/2D | [UP/DOWN] Depth Scale: " + ofToString(depthScale, 0) +
                      " | [R] Reset Cam | Mouse: Drag to orbit, Scroll to zoom";
    ofDrawBitmapStringHighlight(hud, 15, 25);
}

void ofApp::keyPressed(int key) {
    if (key == ' ') {
        bShowPointCloud = !bShowPointCloud;
    } else if (key == 'r' || key == 'R') {
        easyCam.reset();
        easyCam.setDistance(450);
    } else if (key == OF_KEY_UP) {
        depthScale += 25.0f;
    } else if (key == OF_KEY_DOWN) {
        depthScale = std::max(25.0f, depthScale - 25.0f);
    }
}