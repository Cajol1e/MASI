#ifndef LIFNODE_H
#define LIFNODE_H

#include "layer.h"
namespace ncnn {

class LIFNode : public Layer
{
public:
    LIFNode();
    virtual int load_param(const ParamDict& pd);
    virtual int load_model(const ModelBin& mb);
    virtual int load_model(const ModelBin& mb, const Option& opt);
    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
    virtual int forward(const flexnn::DummyMat& bottom_blob, flexnn::DummyMat& top_blob, const Option& opt) const;
    virtual int reset();

private:
    virtual int single_step_forward_hard_reset_decay_input(Mat& x, const Option& opt) const;
    virtual int single_step_forward_hard_reset_no_decay_input(Mat& x, const Option& opt) const;
    virtual int single_step_forward_soft_reset_decay_input(Mat& x, const Option& opt) const;
    virtual int single_step_forward_soft_reset_no_decay_input(Mat& x, const Option& opt) const;
    virtual int fill(float init_v);

public:
    float v_init;
    float tau;
    bool decay_input;
    float v_threshold;
    float v_reset;
    int w, h, c;
    mutable Mat v_data;
};
} // namespace ncnn

#endif
