#include "lifnode.h"
#include <stdio.h>

namespace ncnn {

LIFNode::LIFNode()
{
    one_blob_only = true;
    support_inplace = true;
    v_data = Mat();
}

int LIFNode::load_param(const ParamDict& pd)
{
    tau = pd.get(0, 2.0f);
    decay_input = pd.get(1, true);
    v_threshold = pd.get(2, 1.0f);
    v_reset = pd.get(3, 0.0f);
    v_init = pd.get(4, 0.0f);
    w = pd.get(5, 0);
    h = pd.get(6, 0);
    c = pd.get(7, 0);
    return 0;
}

int LIFNode::load_model(const ModelBin& mb)
{
    size_t elemsize = 4;
    if (v_data.empty()) {
        v_data.create(w, h, c, elemsize);
        fill(v_init);
    }
    
    return 0;
}

int LIFNode::load_model(const ModelBin& mb, const Option& opt)
{
    size_t elemsize = 4;

    if (v_data.empty())
    {
        if (std::string(opt.mode).find("old") != std::string::npos)
        {
            v_data.create(w, h, c, elemsize, opt.weight_allocator);
        }
        else {
            v_data.create(w, h, c, elemsize, opt.persistence_weight_allocator);
        }

        fill(v_init);
    }
    else {
        if (opt.weight_allocator || opt.persistence_weight_allocator)
        {
            //Ìø¹ý
            size_t cstep = alignSize((size_t)w * h * elemsize, 16) / elemsize;
            size_t totalsize = alignSize(c * cstep * elemsize, 4);
            if (std::string(opt.mode).find("old") != std::string::npos)
            {
                opt.weight_allocator->fastMalloc(totalsize + 4);
            }
            else
            {
                opt.persistence_weight_allocator->fastMalloc(totalsize + 4);
            }
        }
    }
    return 0;
}

int LIFNode::single_step_forward_hard_reset_decay_input(Mat& x, const Option& opt) const
{
    int size = x.w * x.h;

    //print_float_mat_lif(v_data, "/home/root/flexnn/data/v_data.txt");

#pragma omp parallel for num_threads(opt.num_threads)
    for (int i = 0; i < x.c; i++)
    {   
        float* x_ptr = x.channel(i);
        float* v_ptr = v_data.channel(i);
        for (int j = 0; j < size; j++)
        {   

            v_ptr[j] = v_ptr[j] + (x_ptr[j] - (v_ptr[j] - v_reset)) / tau;
            x_ptr[j] = v_ptr[j] >= v_threshold ? 1.0 : 0.0;
            v_ptr[j] = x_ptr[j] * v_reset + v_ptr[j] * (1.0 - x_ptr[j]);
        }
    }
    return 0;
}

int LIFNode::single_step_forward_hard_reset_no_decay_input(Mat& x, const Option& opt) const
{
    int size = x.w * x.h;

#pragma omp parallel for num_threads(opt.num_threads)
    for (int i = 0; i < x.c; i++)
    {
        float* x_ptr = x.channel(i);
        float* v_ptr = v_data.channel(i);
        for (int j = 0; j < size; j++)
        {
            v_ptr[j] = v_ptr[j] + (v_ptr[j] - v_reset) / tau + x_ptr[j];
            x_ptr[j] = v_ptr[j] >= v_threshold ? 1.0 : 0.0;
            v_ptr[j] = x_ptr[j] * v_reset + v_ptr[j] * (1.0 - x_ptr[j]);
        }
    }
    return 0;
}

int LIFNode::single_step_forward_soft_reset_decay_input(Mat& x, const Option& opt) const
{
    int size = x.w * x.h;

#pragma omp parallel for num_threads(opt.num_threads)
    for (int i = 0; i < x.c; i++)
    {
        float* x_ptr = x.channel(i);
        float* v_ptr = v_data.channel(i);
        for (int j = 0; j < size; j++)
        {   
            v_ptr[j] = v_ptr[j] + (x_ptr[j] - v_ptr[j]) / tau;
            x_ptr[j] = v_ptr[j] >= v_threshold ? 1.0 : 0.0;
            v_ptr[j] = v_ptr[j] - x_ptr[j] * v_threshold;
        }
    }
    return 0;
}

int LIFNode::single_step_forward_soft_reset_no_decay_input(Mat& x, const Option& opt) const
{
    int size = x.w * x.h;

#pragma omp parallel for num_threads(opt.num_threads)
    for (int i = 0; i < x.c; i++)
    {
        float* x_ptr = x.channel(i);
        float* v_ptr = v_data.channel(i);
        for (int j = 0; j < size; j++)
        {
            v_ptr[j] = v_ptr[j] * (1.0 - 1.0 / tau) + x_ptr[j];
            x_ptr[j] = v_ptr[j] >= v_threshold ? 1.0 : 0.0;
            v_ptr[j] = v_ptr[j] - x_ptr[j] * v_threshold;
        }
    }
    return 0;
}

int LIFNode::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{

    int res;
    if (v_reset < 0)
    {
        if (decay_input)
        {
            res = single_step_forward_soft_reset_decay_input(bottom_top_blob, opt);
        }
        else
        {
            res = single_step_forward_soft_reset_no_decay_input(bottom_top_blob, opt);
        }
    }
    else
    {
        if (decay_input)
        {
            res = single_step_forward_hard_reset_decay_input(bottom_top_blob, opt);
        }
        else
        {
            res = single_step_forward_hard_reset_no_decay_input(bottom_top_blob, opt);
        }
    }
        

    if (!res)
    {
        return 0;
    }
    else
    {
        return -1;
    }
}

int LIFNode::forward(const flexnn::DummyMat& bottom_blob, flexnn::DummyMat& top_blob, const Option& opt) const
{
    //printf("%d %d %d\n",w,h,c);
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int c = bottom_blob.c;
    //printf("%d %d %d\n", w, h, c);
    int dim = bottom_blob.dims;
    size_t elementSize = bottom_blob.elemsize;
    if (dim == 2 && (c == 1 || c == -1))
    {
        top_blob.create(w, h, elementSize, opt.blob_allocator);
    }
    else {
        top_blob.create(w, h, c, elementSize, opt.blob_allocator);
    }
    
    if (top_blob.empty())
        return -100;

    return 0;
}

int LIFNode::reset() {
    if (v_data.empty())
        return -1;

    fill(v_init);
    //print_float_mat_lif(v_data, "/home/root/flexnn/data/v_data.txt");
    return 0;
}

int LIFNode::fill(float init_v)
{
    if (v_data.empty())
        return -1;

    int h, w, d, c;
    h = v_data.h;
    w = v_data.w;
    d = v_data.d;
    c = v_data.c;
    for (int i = 0; i < c; i++)
    {
        float* ptr = v_data.channel(i);
        for (int j = 0; j < d; j++)
        {
            for (int k = 0; k < h; k++)
            {
                for (int l = 0; l < w; l++)
                {
                    *ptr = init_v;
                    ptr++;
                }
            }
        }
    }
    return 0;
}

} // namespace ncnn