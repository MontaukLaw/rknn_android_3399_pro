package com.example.myarm64rknn;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.os.Environment;
import android.widget.TextView;

import com.example.myarm64rknn.databinding.ActivityMainBinding;

import java.io.File;

public class MainActivity extends AppCompatActivity {

    // Used to load the 'myarm64rknn' library on application startup.
    static {
        System.loadLibrary("myarm64rknn");
    }

    private ActivityMainBinding binding;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        // 将文件路径转成String格式
        String imageData = (new File(Environment.getExternalStorageDirectory(), "test_rgb_2000.rgb")).getAbsolutePath();
        String modelPath = (new File(Environment.getExternalStorageDirectory(), "yolov5s_relu_out_opt.rknn")).getAbsolutePath();
        String labelPath = (new File(Environment.getExternalStorageDirectory(), "coco_80_labels_list.txt")).getAbsolutePath();

        // 创建YD类的实例
        YoloDetector yoloDetector = new YoloDetector(imageData, modelPath, labelPath);
        yoloDetector.startDetect();

        // Example of a call to a native method
        TextView tv = binding.sampleText;
        tv.setText(stringFromJNI());
    }

    /**
     * A native method that is implemented by the 'myarm64rknn' native library,
     * which is packaged with this application.
     */
    public native String stringFromJNI();
}