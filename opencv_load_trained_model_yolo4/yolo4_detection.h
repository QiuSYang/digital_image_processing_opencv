/*
# ʹ��OpenCV dnnʵ�� yolo4Ŀ����
*/
#ifndef _YOLO4_DETECTION_H_
#define _YOLO4_DETECTION_H_

#include <iostream>
#include <string.h>
#include <vector>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

using namespace std;
using namespace cv;
using namespace dnn;

class Yolo4Detection 
{
public:
    // ���캯��
    Yolo4Detection();
    // ��������
    ~Yolo4Detection();
    // ��ʼ������
    void Initialize(int width, int height);
    // ��ȡ����ģ��
    void ReadModel();
    // �����복�����
    bool Detecting(Mat frame);
    // ��ȡ�������������
    vector<String> GetOutputsNames();
    // ��������д���ʹ��NMSѡ������ʵĿ�
    void PostProcess();
    // �������
    void Drawer();
    // ��������������Ϣ
    void DrawBoxes(int classId, float conf, int left, int top, int right, int bottom);
    // ��ȡMat����
    Mat GetFrame();
    // ��ȡͼ����
    int GetResWidth();
    // ��ȡͼ��߶�
    int GetResHeight();

private:
    int m_width;			// ͼ����
    int m_height;			// ͼ��߶�
    // ���紦�����
    Net m_model;			// ����ģ��
    Mat m_frame;			// ÿһ֡
    Mat m_blob;				// ��ÿһ֡����һ��4D��blob������������
    vector<Mat> m_outs;		// �������
    vector<float> m_confs;	// ���Ŷ�
    vector<Rect> m_boxes;	// �������Ͻ����ꡢ����
    vector<int> m_classIds;	// ���id
    vector<int> m_perfIndx;	// �Ǽ�����ֵ�����߽����±�
    //��ⳬ����
    int m_inpWidth;			// ��������ͼ����
    int m_inpHeight;		// ��������ͼ��߶�
    float m_confThro;		// ���Ŷ���ֵ
    float m_NMSThro;		// NMS�Ǽ���������ֵ
    vector<string> m_classes; // �������
};

#endif // !_YOLO4_DETECTION_H_

