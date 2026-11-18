#include "ofApp.h"

void ofApp::setup() {
    ofSetWindowTitle("ofxTensorFlow2 Depth Extractor");
    ofSetVerticalSync(true);

    // Optional: Prevent TF from allocating 100% of GPU VRAM on startup
    ofxTF2::setGPUMaxMemory(ofxTF2::GPU_PERCENT_70);

    // Initialize webcam
    cam.setup(640, 480);

    // Allocate buffers
    resizedPixels.allocate(modelWidth, modelHeight, OF_PIXELS_RGB);
    depthImg.allocate(modelWidth, modelHeight, OF_IMAGE_GRAYSCALE);

    // Load SavedModel from bin/data/model
    if (!model.load("model")) {
        ofLogError() << "Failed to load TensorFlow SavedModel!";
        std::exit(EXIT_FAILURE);
    }

    // Optional: Explicitly define input/output signature names if your model uses custom ones
    model.setup({"serving_default_x"}, {"StatefulPartitionedCall"});
}

void ofApp::update() {
    cam.update();

    if (cam.isFrameNew()) {
        // 1. Resize webcam feed to model's expected input dimensions (256x256)
        cam.getPixels().resizeTo(resizedPixels);

        // 2. Convert ofPixels (uint8) to float32 cppflow::tensor and normalize to [0.0, 1.0]
        cppflow::tensor input = ofxTF2::pixelsToTensor(resizedPixels);
        input = cppflow::cast(input, TF_UINT8, TF_FLOAT);
        input = cppflow::div(input, cppflow::tensor({255.0f}));

        // Expand dims from [256, 256, 3] to batch shape [1, 256, 256, 3]
        input = cppflow::expand_dims(input, 0);

        // 3. Run inference
        cppflow::tensor output = model.runModel(input);

        // 4. Post-process: Extract float vector from output tensor
        std::vector<float> depthData = output.get_data<float>();

        // Find min and max inverse-depth values for dynamic contrast normalization
        auto [minIt, maxIt] = std::minmax_element(depthData.begin(), depthData.end());
        float minVal = *minIt;
        float maxVal = *maxIt;
        float range = (maxVal - minVal > 1e-6f) ? (maxVal - minVal) : 1.0f;

        // Map normalized depth values to 8-bit grayscale pixels (0 = far, 255 = close)
        ofPixels& outPixels = depthImg.getPixels();
        for (size_t i = 0; i < depthData.size(); ++i) {
            float normalized = (depthData[i] - minVal) / range;
            outPixels[i] = static_cast<unsigned char>(ofClamp(normalized * 255.0f, 0.0f, 255.0f));
        }

        // Upload updated pixels to GPU texture
        depthImg.update();
    }
}

void ofApp::draw() {
    ofBackground(20);

    // Draw original webcam feed on the left
    cam.draw(0, 0, 640, 480);

    // Draw upscaled depth map on the right
    depthImg.draw(640, 0, 640, 480);

    ofDrawBitmapStringHighlight("FPS: " + ofToString(ofGetFrameRate(), 1), 15, 25);
}