// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

package com.tencent.nanodetncnn;

import android.app.Activity;
import android.content.res.Resources;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceHolder;
import android.view.View;
import android.view.WindowManager;
import android.widget.AdapterView;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Spinner;

import java.util.ArrayList;

public class MainActivity extends Activity
{
    private NanoDetNcnn nanodetncnn = new NanoDetNcnn();

    private Spinner spinnerModel;
    private Spinner spinnerCPUGPU;
    private int current_model = 0;
    private int current_cpugpu = 0;

    private ImageView imageView;
    private Button predictButton;

    private int currentImageIndex = 0; // 현재 사용 중인 이미지 인덱스
    private int[] imageResources = {R.drawable.test2, R.drawable.test3, R.drawable.test4}; // 이미지 리소스 배열

    private String bboxdata;

    @Override
    public void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        imageView = (ImageView) findViewById(R.id.image_view);

        // 예측버튼
        predictButton = (Button) findViewById(R.id.predictbutton);
        predictButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {

                // 현재 이미지 리소스 가져오기
                int resourceId = imageResources[currentImageIndex];
                Resources resources = getResources();
                Bitmap bitmap = BitmapFactory.decodeResource(resources, resourceId);

                // 이미지 인식하기
                bboxdata = nanodetncnn.predict(imageView, bitmap);
                if(bboxdata != null) {
                    Log.e("Main", String.valueOf(bboxdata));
                }
                // 다음 이미지를 사용하기 위해 인덱스 업데이트
                currentImageIndex = (currentImageIndex + 1) % imageResources.length;
            }
        });

        spinnerModel = (Spinner) findViewById(R.id.spinnerModel);
        spinnerModel.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> arg0, View arg1, int position, long id)
            {
                if (position != current_model)
                {
                    current_model = position;

                    Log.e("Main","model : "+current_model);

                    reload();
                }
            }
            @Override
            public void onNothingSelected(AdapterView<?> arg0)
            {
            }
        });

        spinnerCPUGPU = (Spinner) findViewById(R.id.spinnerCPUGPU);
        spinnerCPUGPU.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> arg0, View arg1, int position, long id)
            {
                if (position != current_cpugpu)
                {
                    current_cpugpu = position;
                    reload();
                }
            }
            @Override
            public void onNothingSelected(AdapterView<?> arg0)
            {
            }
        });

        reload();
    }

    // 이미지를 설정하는 메서드
    public void setImageBitmap(Bitmap bitmap) {
        runOnUiThread(() -> imageView.setImageBitmap(bitmap));
    }

    private void reload()
    {
        boolean ret_init = nanodetncnn.loadModel(getAssets(), 0, current_cpugpu);
        if (!ret_init)
        {
            Log.e("MainActivity", "nanodetncnn loadModel failed");
        }
    }

    @Override
    public void onResume()
    {
        super.onResume();
    }

    @Override
    public void onPause()
    {
        super.onPause();
    }
}
