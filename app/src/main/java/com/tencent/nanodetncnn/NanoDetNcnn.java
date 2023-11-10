package com.tencent.nanodetncnn;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.view.Surface;
import android.widget.ImageView;

import java.util.ArrayList;

public class NanoDetNcnn
{
    public native boolean loadModel(AssetManager mgr, int modelid, int cpugpu);
    public native String predict(ImageView imageView, Bitmap bitmap);

    static {
        System.loadLibrary("nanodetncnn");
    }
}
