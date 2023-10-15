package com.example.myarm64rknn;

public class YoloDetector {
    static {
        System.loadLibrary("myarm64rknn");
    }

    private String imageData = "";
    private String modelPath = "";
    private String labelPath = "";

    // YD类的构建方法，用于传入图片数据、模型路径、标签路径
    public YoloDetector(String imageData, String modelPath, String labelPath) {
        this.imageData = imageData;
        this.modelPath = modelPath;
        this.labelPath = labelPath;
    }

    public void startDetect() {
        startDetect(imageData, modelPath, labelPath);
    }

    private native void startDetect(String modelDataSource, String modelPath, String labelPath);

    private native void prepare();

}
