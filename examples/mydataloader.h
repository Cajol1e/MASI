#ifndef MY_DATALOADER_H
#define MY_DATALOADER_H
#include <stdio.h>
#include "mat.h"
#include "stdlib.h"
#include "errno.h"

std::string getFilename(const std::string& path)
{
    auto last_slash_pos = path.find_last_of("/\\"); // 查找最后一个斜杠或反斜杠的位置
    if (last_slash_pos != std::string::npos)
    {
        return path.substr(last_slash_pos + 1); // 返回从最后一个斜杠或反斜杠之后的部分
    }
    else
    {
        return path; // 如果没有找到斜杠或反斜杠，则整个字符串就是文件名
    }
}

class MyDataLoader
{
public:
    MyDataLoader();
    MyDataLoader(const char* path);
    MyDataLoader(const char* path, const char* input_shape);
    virtual int loadData(ncnn::Mat& in, const ncnn::Option& opt);
    virtual int loadData(ncnn::Mat& in, const ncnn::Option& opt, int& target, ncnn::Mat& output, int& predicted);

public:
    FILE* file;
    std::string file_name;
    std::string input_shape = "";
};

MyDataLoader::MyDataLoader()
{
}

MyDataLoader::MyDataLoader(const char* path)
{
    file = fopen(path, "r");
    if (file == NULL)
    {
        file_name = "";
        printf("file empty, use random data!\n");
    }
    else {
        file_name = getFilename(std::string(path));
    }
}
MyDataLoader::MyDataLoader(const char* path,const char* _input_shape)
{
    input_shape = std::string(_input_shape);

    file = fopen(path, "r");
    if (file == NULL)
    {
        file_name = "";
        printf("file empty, use random data!\n");
    }
    else
    {
        file_name = getFilename(std::string(path));
    }
}

int MyDataLoader::loadData(ncnn::Mat& in, const ncnn::Option& opt)
{
    int w, h, c, d;
    sscanf(input_shape.c_str(), "[%d,%d,%d,%d]", &d, &c, &h, &w);

    if (in.empty()) {
        in = ncnn::Mat(w, h, d, c);
    }
    Randomize(in);
    return 0;
}

int MyDataLoader::loadData(ncnn::Mat& in, const ncnn::Option& opt, int& target, ncnn::Mat& output, int& predicted)
{
    int w, h, c, d;
    sscanf(input_shape.c_str(), "[%d,%d,%d,%d]", &d, &c, &h, &w);

    if (in.empty())
        in = ncnn::Mat(w, h, d, c);

    float temp; 

    for (int c = 0; c < 3; c++)
    {
        float* ptr = in.channel(c);

        for (int i = 0; i < 32 * 32; i++)
        {

            if (fscanf(file, "%f", &temp) == EOF)
            {
                return -1;
            }
            else {
                *ptr = temp;
                ptr++;
            }
        }
    }
    

    fscanf(file, "%f", &temp);
    target = (int)temp;

    if (output.empty())
    {
        output = ncnn::Mat(10);
    }

    float* ptr = output.channel(0);

    for (int i = 0; i < 10; i++)
    {
        fscanf(file, "%f", &temp);
        *ptr = temp;
        ptr++;
    }

    fscanf(file, "%f", &temp);
    predicted = (int)temp;

    return 0;
}

#endif