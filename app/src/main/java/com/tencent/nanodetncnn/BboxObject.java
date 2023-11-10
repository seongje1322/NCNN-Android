package com.tencent.nanodetncnn;

// 자바로 bboxes 데이터 반환
public class BboxObject {
    public int label;
    public float prob;
    public float x;
    public float y;
    public float width;
    public float height;

    public BboxObject(int label,float prob, float x, float y, float width, float height) {
        this.label = label;
        this.prob = prob;
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;
    }
}