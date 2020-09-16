#include "yolo4_detection.h"

Yolo4Detection::Yolo4Detection()
{
    // 图像属性
    m_width = 0;
    m_height = 0; 
    m_inpWidth = 416;
    m_inpHeight = 416;

    // 检测阈值属性
    m_confThro = 0.25f; 
    m_NMSThro = 0.4f;

    // 构建检测网络
    ReadModel();
}

Yolo4Detection::~Yolo4Detection()
{
    // vector变量释放
    // m_outs.clear();
    vector<Mat>().swap(m_outs);
    // m_boxes.clear();
    vector<Rect>().swap(m_boxes);
    // m_confs.clear();
    vector<float>().swap(m_confs);
    // m_classIds.clear();
    vector<int>().swap(m_classIds);
    // m_perfIndx.clear();
    vector<int>().swap(m_perfIndx);
    // m_classes.clear();
    vector<string>().swap(m_classes);
}

void Yolo4Detection::Initialize(int width, int height)
{
    m_width = width;
    m_height = height;
}

void Yolo4Detection::ReadModel()
{
    string classFile = "./config/coco.names";
    string modelConfig = "./config/yolov4.cfg";
    string modelWeights = "./config/yolov4.weights";

    // 加载类别名称
    ifstream ifs(classFile.c_str());
    string line;
    while (getline(ifs, line)) {
        m_classes.push_back(line);
    }

    m_model = readNetFromDarknet(modelConfig, modelWeights);
    // m_model = dnn_DetectionModel(modelConfig, modelWeights);  // opencv4.4新增加加载深度学习模型的接口
    m_model.setPreferableBackend(DNN_BACKEND_OPENCV);
    m_model.setPreferableTarget(DNN_TARGET_CPU);
}

bool Yolo4Detection::Detecting(Mat frame)
{
    m_frame = frame.clone();

    //创建4D的blob用于网络输入
    blobFromImage(m_frame, m_blob, 1 / 255.0, Size(m_inpWidth, m_inpHeight), Scalar(0, 0, 0), true, false);

    //设置网络输入
    m_model.setInput(m_blob);

    //前向预测得到网络输出，forward需要知道输出层，这里用了一个函数找到输出层
    m_model.forward(m_outs, GetOutputsNames());

    //使用非极大抑制NMS删除置信度较低的边界框
    PostProcess();

    //画检测框
    Drawer();

    return true;
}

vector<String> Yolo4Detection::GetOutputsNames()
{
    static vector<String> names;
    if (names.empty())
    {
        //得到输出层索引号
        vector<int> outLayers = m_model.getUnconnectedOutLayers();

        //得到网络中所有层名称
        vector<String> layersNames = m_model.getLayerNames();

        //获取输出层名称
        names.resize(outLayers.size());
        for (int i = 0; i < outLayers.size(); ++i)
            names[i] = layersNames[outLayers[i] - 1];
    }

    return names;
}

void Yolo4Detection::PostProcess()
{
    for (int num = 0; num < m_outs.size(); num++)
    {
        Point Position;
        double confidence;

        //得到每个输出的数据
        float* data = (float*)m_outs[num].data;
        for (int j = 0; j < m_outs[num].rows; j++, data += m_outs[num].cols)
        {
            //得到该输出的所有类别的
            Mat scores = m_outs[num].row(j).colRange(5, m_outs[num].cols);

            //获取最大置信度对应的值和位置
            minMaxLoc(scores, 0, &confidence, 0, &Position);

            //对置信度大于阈值的边界框进行相关计算和保存
            if (confidence > m_confThro)
            {
                //data[0],data[1],data[2],data[3]都是相对于原图像的比例
                int centerX = (int)(data[0] * m_width);
                int centerY = (int)(data[1] * m_height);
                int width = (int)(data[2] * m_width);
                int height = (int)(data[3] * m_height);
                int left = centerX - width / 2;
                int top = centerY - height / 2;
                //保存信息
                m_classIds.push_back(Position.x);
                m_confs.push_back((float)confidence);
                m_boxes.push_back(Rect(left, top, width, height));
            }
        }
    }
    //非极大值抑制，以消除具有较低置信度的冗余重叠框
    NMSBoxes(m_boxes, m_confs, m_confThro, m_NMSThro, m_perfIndx);
}

void Yolo4Detection::Drawer()
{
    //获取所有最佳检测框信息
    for (int i = 0; i < m_perfIndx.size(); i++)
    {
        int idx = m_perfIndx[i];
        Rect box = m_boxes[idx];
        DrawBoxes(m_classIds[idx], m_confs[idx], box.x, box.y,
            box.x + box.width, box.y + box.height);
    }
}

void Yolo4Detection::DrawBoxes(int classId, float conf, int left, int top, int right, int bottom)
{
    //画检测框
    rectangle(m_frame, Point(left, top), Point(right, bottom), Scalar(255, 178, 50), 3);

    //该检测框对应的类别和置信度
    string label = format("%.2f", conf);
    if (!m_classes.empty())
    {
        CV_Assert(classId < (int)m_classes.size());
        label = m_classes[classId] + ":" + label;
    }

    //将标签显示在检测框顶部
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = max(top, labelSize.height);
    rectangle(m_frame, Point(left, top - round(1.5*labelSize.height)), 
        Point(left + round(1.5*labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
    putText(m_frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 0), 1);
}

Mat Yolo4Detection::GetFrame()
{
    return m_frame;
}

int Yolo4Detection::GetResWidth()
{
    return m_width;
}

int Yolo4Detection::GetResHeight()
{
    return m_height;
}
