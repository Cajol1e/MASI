#ifdef _MSC_VER
#define _CRT_SECURE_NO_DEPRECATE
#endif

#include <algorithm>
#include <map>
#include <set>
#include <vector>
#include <stack>

// ncnn public header
#include "datareader.h"
#include "modelbin.h"
#include "layer.h"
#include "layer_type.h"
#include "net.h"
#include "profiler.h"

// ncnn private header
#include "modelwriter.h"

// flexnn utils
#include "mydataloader.h"
#include "flexnn_utils.h"
#include "dummymat.h"

#include "benchmark_utils.h"

#include "flexnnschedule.h"
#include "flexnnslice.h"

static int g_warmup_loop_count = 0;
static int g_loop_count = 32;
static int g_cooling_down_duration = 0; // seconds
static bool g_enable_cooling_down = false;
static bool g_load_model_bin = true;
static int g_computing_powersave = 2;
static int g_loading_powersave = 3;
static int g_num_threads = 2;
static int batch_size = 1;
static int T = 4;
static const char* input_shape = "[1,16,32,32]";
static double seenn_threshold = -1;
static bool use_seenn = false;
static int log_num = 1000;

long get_memory_budget(std::vector<MemoryProfilerEvent> events,int layer_num)
{
    std::map<void*, std::pair<int, int> > event_record;
    std::vector<int> memory_placed(layer_num,0);
    int max_memory = -1;

    for (int i = 0; i < (int)events.size(); i++)
    {
        const flexnn::MemoryProfilerEvent& event = events[i];
        if (event.event_type == 0) {
            if (event_record.find(event.ptr) != event_record.end()) {
                event_record[event.ptr].second = event.layer_index;
            }
        }
        else {
            event_record[event.ptr] = {event.layer_index,event.layer_index};
        }
    }

    for (int i = 0; i < (int)events.size(); i++)
    {
        const flexnn::MemoryProfilerEvent& event = events[i];
        if (event.event_type == 0)
        {
            continue;
        }
        
        std::pair<int, int> record = event_record[event.ptr];
        for (int i = record.first; i <= record.second; i++) {
            memory_placed[i] += event.size;
            max_memory = std::max(max_memory, memory_placed[i]);
        }
    }
    return max_memory;
}
    
double get_latency(std::vector<LayerTimeProfile> events)
{
    return events[0].computing_duration+events[0].loading_duration;
}

static int transform_kernel_convolution_winograd63_arm(int layer_index, const ncnn::Net& net)
{
    std::vector<ncnn::Blob> blobs = net.blobs();
    std::vector<ncnn::Layer*> layers = net.layers();
    const size_t layer_count = layers.size();
    const size_t blob_count = blobs.size();

    if (layers[layer_index]->type != "Convolution")
    {
        fprintf(stderr, "Error: layer %d %s is not convolution\n", layer_index, layers[layer_index]->name.c_str());
        return -1;
    }

    int top_blob_index = layers[layer_index]->tops[0];
    int bottom_blob_index = layers[layer_index]->bottoms[0];

    ncnn::Convolution* convolution = (ncnn::Convolution*)layers[layer_index];
    int kernel_w = convolution->kernel_w;
    int kernel_h = convolution->kernel_h;
    int dilation_w = convolution->dilation_w;
    int dilation_h = convolution->dilation_h;
    int stride_w = convolution->stride_w;
    int stride_h = convolution->stride_h;

    //slice_convolution只对3x3s1的卷积进行了winograd分片
    if (!(kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1))
    {
        fprintf(stderr, "Error: layer %d %s is not conv3x3s1\n", layer_index, layers[layer_index]->name.c_str());
        return -1;
    }

    //使用ncnn::conv3x3s1_winograd63_transform_kernel方法获取winograd63算法对应的卷积权重参数
    ncnn::Mat weight_winograd63_data;
    ncnn::Option opt;
    //opt.num_threads = thread_num;
    ncnn::conv3x3s1_winograd63_transform_kernel(convolution->weight_data, weight_winograd63_data, 16, convolution->num_output, opt);

    convolution->weight_data_type = 3;
    convolution->weight_data = weight_winograd63_data; // replace with transformed one
    if (convolution->weight_data.empty())
    {
        fprintf(stderr, "Error: winograd63 layer %d %s weight data is empty\n", layer_index, layers[layer_index]->name.c_str());
        return -1;
    }
    convolution->weight_w = weight_winograd63_data.w;
    convolution->weight_h = weight_winograd63_data.h;
    convolution->weight_c = weight_winograd63_data.c;

    return 0;
}

static int transform_kernel_convolution_winograd43_arm(int layer_index, const ncnn::Net& net)
{
    std::vector<ncnn::Blob> blobs = net.blobs();
    std::vector<ncnn::Layer*> layers = net.layers();
    const size_t layer_count = layers.size();
    const size_t blob_count = blobs.size();

    if (layers[layer_index]->type != "Convolution")
    {
        fprintf(stderr, "Error: layer %d %s is not convolution\n", layer_index, layers[layer_index]->name.c_str());
        return -1;
    }

    int top_blob_index = layers[layer_index]->tops[0];
    int bottom_blob_index = layers[layer_index]->bottoms[0];

    ncnn::Convolution* convolution = (ncnn::Convolution*)layers[layer_index];
    int kernel_w = convolution->kernel_w;
    int kernel_h = convolution->kernel_h;
    int dilation_w = convolution->dilation_w;
    int dilation_h = convolution->dilation_h;
    int stride_w = convolution->stride_w;
    int stride_h = convolution->stride_h;

    //slice_convolution只对3x3s1的卷积进行了winograd分片
    if (!(kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1))
    {
        fprintf(stderr, "Error: layer %d %s is not conv3x3s1\n", layer_index, layers[layer_index]->name.c_str());
        return -1;
    }

    //使用ncnn::conv3x3s1_winograd43_transform_kernel方法获取winograd43算法对应的卷积权重参数
    ncnn::Mat weight_winograd43_data;
    ncnn::Option opt;
    //opt.num_threads = thread_num;
    ncnn::conv3x3s1_winograd43_transform_kernel(convolution->weight_data, weight_winograd43_data, 16, convolution->num_output, opt);

    convolution->weight_data_type = 4;
    convolution->weight_data = weight_winograd43_data; // replace with transformed one
    if (convolution->weight_data.empty())
    {
        fprintf(stderr, "Error: winograd43 layer %d %s weight data is empty\n", layer_index, layers[layer_index]->name.c_str());
        return -1;
    }

    convolution->weight_w = weight_winograd43_data.w;
    convolution->weight_h = weight_winograd43_data.h;
    convolution->weight_c = weight_winograd43_data.c;

    return 0;
}

static int transform_kernel_convolution_winograd23_arm(int layer_index, const ncnn::Net& net)
{
    std::vector<ncnn::Blob> blobs = net.blobs();
    std::vector<ncnn::Layer*> layers = net.layers();
    const size_t layer_count = layers.size();
    const size_t blob_count = blobs.size();

    if (layers[layer_index]->type != "Convolution")
    {
        fprintf(stderr, "Error: layer %d %s is not convolution\n", layer_index, layers[layer_index]->name.c_str());
        return -1;
    }

    int top_blob_index = layers[layer_index]->tops[0];
    int bottom_blob_index = layers[layer_index]->bottoms[0];

    ncnn::Convolution* convolution = (ncnn::Convolution*)layers[layer_index];
    int kernel_w = convolution->kernel_w;
    int kernel_h = convolution->kernel_h;
    int dilation_w = convolution->dilation_w;
    int dilation_h = convolution->dilation_h;
    int stride_w = convolution->stride_w;
    int stride_h = convolution->stride_h;

    //slice_convolution只对3x3s1的卷积进行了winograd分片
    if (!(kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1))
    {
        fprintf(stderr, "Error: layer %d %s is not conv3x3s1\n", layer_index, layers[layer_index]->name.c_str());
        return -1;
    }

    //使用ncnn::conv3x3s1_winograd43_transform_kernel方法获取winograd43算法对应的卷积权重参数
    ncnn::Mat weight_winograd23_data;
    ncnn::Option opt;
    //opt.num_threads = thread_num;
    ncnn::conv3x3s1_winograd23_transform_kernel(convolution->weight_data, weight_winograd23_data, 16, convolution->num_output, opt);

    convolution->weight_data_type = 5;
    convolution->weight_data = weight_winograd23_data; // replace with transformed one
    if (convolution->weight_data.empty())
    {
        fprintf(stderr, "Error: winograd43 layer %d %s weight data is empty\n", layer_index, layers[layer_index]->name.c_str());
        return -1;
    }
    convolution->weight_w = weight_winograd23_data.w;
    convolution->weight_h = weight_winograd23_data.h;
    convolution->weight_c = weight_winograd23_data.c;

    return 0;
}

static int transform_kernel_convolution_im2col_gemm(int layer_index, const ncnn::Net& net)
{
    std::vector<ncnn::Blob> blobs = net.blobs();
    std::vector<ncnn::Layer*> layers = net.layers();
    const size_t layer_count = layers.size();
    const size_t blob_count = blobs.size();

    if (layers[layer_index]->type != "Convolution")
    {
        fprintf(stderr, "Error: layer %d %s is not convolution\n", layer_index, layers[layer_index]->name.c_str());
        return -1;
    }

    int top_blob_index = layers[layer_index]->tops[0];
    int bottom_blob_index = layers[layer_index]->bottoms[0];

    ncnn::Convolution* convolution = (ncnn::Convolution*)layers[layer_index];

    const flexnn::DummyMat in = blobs[bottom_blob_index].dummy_shape;
    const flexnn::DummyMat out = blobs[top_blob_index].dummy_shape;

    ncnn::Mat weight_sgemm_data;
    ncnn::Option opt;

    //使用ncnn::convolution_im2col_gemm_transform_kernel方法获取im2col_gemm算法对应的卷积权重参数
    ncnn::convolution_im2col_gemm_transform_kernel(convolution->weight_data, weight_sgemm_data, 16, convolution->num_output, convolution->kernel_w, convolution->kernel_h, opt);

    convolution->weight_data_type = 2;
    convolution->weight_data = weight_sgemm_data; // replace with transformed one
    if (convolution->weight_data.empty() || convolution->weight_data.w == 0 || convolution->weight_data.h == 0 || convolution->weight_data.c == 0)
    {
        fprintf(stderr, "Error: gemm layer %d %s weight data is empty\n", layer_index, layers[layer_index]->name.c_str());
        return -1;
    }
    // convolution->weight_data_size = weight_sgemm_data.total() * weight_sgemm_data.elemsize; // update weight size
    convolution->weight_w = weight_sgemm_data.w;
    convolution->weight_h = weight_sgemm_data.h;
    convolution->weight_c = weight_sgemm_data.c;

    return 0;
}

void transform_conv(const ncnn::Net& net, char* mode)
{
    std::vector<ncnn::Blob> blobs = net.blobs();
    std::vector<ncnn::Layer*> layers = net.layers();
    const size_t layer_count = layers.size();
    const size_t blob_count = blobs.size();

    for (size_t i = 0; i < layer_count; i++)
    {
        if (layers[i]->type != "Convolution")
            continue;

        ncnn::Convolution* convolution = (ncnn::Convolution*)layers[i];
        fprintf(stderr, "deal with kernel of conv : Layer No.%d\n", i);

        int kernel_w = convolution->kernel_w;
        int kernel_h = convolution->kernel_h;
        int dilation_w = convolution->dilation_w;
        int dilation_h = convolution->dilation_h;
        int stride_w = convolution->stride_w;
        int stride_h = convolution->stride_h;

        int top_blob_index = layers[i]->tops[0];
        int bottom_blob_index = layers[i]->bottoms[0];

        std::string mode_str = std::string(mode);

        if (mode_str.find("winograd63") != std::string::npos)
        {
            transform_kernel_convolution_winograd63_arm(i,net);
        }

        if (mode_str.find("winograd43") != std::string::npos)
        {
            transform_kernel_convolution_winograd43_arm(i, net);
        }

        if (mode_str.find("winograd23") != std::string::npos)
        {
            transform_kernel_convolution_winograd23_arm(i, net);
        }

        if (mode_str.find("gemm") != std::string::npos) {
            transform_kernel_convolution_im2col_gemm(i, net);
        }
    }
}

void print_memory_budget(std::vector<MemoryProfilerEvent> events, std::vector<ncnn::Layer*> layers)
{
    std::map<void*, std::pair<int, int> > event_record;
    std::map<void*, int> size_record;
    std::map<std::string, std::pair<int, int> > mapper;
    int max_memory = -1;
    std::vector<int> memory_placed(layers.size(), 0);

    for (int i = 0; i < (int)events.size(); i++)
    {
        const flexnn::MemoryProfilerEvent& event = events[i];

        if (event.memory_type == 1)
            continue;

        if (event.memory_type == 2)
            continue;

        if (event.event_type == 0)
        {
            if (event_record.find(event.ptr) != event_record.end())
            {
                event_record[event.ptr].second = event.layer_index;
                for (int lid = event_record[event.ptr].first; lid <= event_record[event.ptr].second; lid++) {
                    memory_placed[lid] += size_record[event.ptr];
                }
                event_record.erase(event.ptr);
                size_record.erase(event.ptr);
            }
        }
        else
        {
            if (event.memory_type == 3)
            {
                memory_placed[event.layer_index] += event.size;
            }
            else {
                event_record[event.ptr] = {event.layer_index, event.layer_index};
                size_record[event.ptr] = event.size;
            }
        }
    }

    for (int i = 0; i < (int)memory_placed.size(); i++)
    {
        printf("%d %s:%ld\n", i, layers[i]->name.c_str(), memory_placed[i]);
    }

}

void profile_layer(char* param_path, char* bin_path) {
    std::string final_param_path = std::string(param_path);
    std::string final_bin_path = std::string(bin_path);
    flexnn::MemoryProfiler g_memory_profiler;
    flexnn::MemoryProfilerInterface g_weight_interface;
    flexnn::MemoryProfilerInterface g_blob_interface;
    flexnn::MemoryProfilerInterface g_intermediate_interface;
    flexnn::MemoryProfilerInterface g_persistence_weight_interface;
    //flexnn::UnlockedTimeProfiler g_unlocked_time_profiler;

    std::vector<MemoryProfilerEvent> g_memory_profiles;
    std::vector<LayerTimeProfile> g_time_profiles;

    g_weight_interface.set_attributes(0, 0);
    g_blob_interface.set_attributes(0, 1);
    g_intermediate_interface.set_attributes(0, 2);
    g_persistence_weight_interface.set_attributes(0, 3);

    g_memory_profiler.add(&g_weight_interface);
    g_memory_profiler.add(&g_blob_interface);
    g_memory_profiler.add(&g_intermediate_interface);
    g_memory_profiler.add(&g_persistence_weight_interface);

    ncnn::Net net;
    ncnn::Option opt;

    // common options
    opt.lightmode = true;
    opt.num_threads = g_num_threads;
    opt.use_local_pool_allocator = false;
    opt.use_local_threads = true;

    // conv impl
    opt.use_winograd_convolution = false;
    opt.use_sgemm_convolution = false;
    opt.use_winograd23_convolution = false;
    opt.use_winograd43_convolution = false;
    opt.use_winograd63_convolution = false;

    // int8, fp16, packing
    opt.use_int8_inference = false;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_int8_storage = false;
    opt.use_int8_arithmetic = false;
    opt.use_packing_layout = false;

    opt.use_ondemand_loading = true;
    opt.use_pretransform = false;
    opt.use_memory_profiler = true;

    opt.blob_allocator = &g_blob_interface;
    opt.weight_allocator = &g_weight_interface;
    opt.workspace_allocator = &g_intermediate_interface;
    opt.persistence_weight_allocator = &g_persistence_weight_interface;
    //opt.time_profiler = &g_unlocked_time_profiler;

    strcpy(opt.mode, "layer_memory_profile");

    // omp settings
    ncnn::set_omp_dynamic(0);
    ncnn::set_omp_num_threads(g_num_threads);
    ncnn::set_cpu_powersave(g_computing_powersave); // 1 使用小核 2 使用大核 3 倾向使用中核

    int w, h, c, d;
    sscanf(input_shape, "[%d,%d,%d,%d]", &d, &c, &h, &w);

    ncnn::Mat in(w, h, d, c, 4u, opt.blob_allocator);
    //ncnn::Mat in(w, h, d, c, 4u);
    in.fill(0.01f);
    ncnn::Mat out_temp;

    net.opt = opt;

    net.load_param(final_param_path.c_str());
    net.load_model(final_bin_path.c_str());

    const std::vector<const char*>& input_names = net.input_names();
    const std::vector<const char*>& output_names = net.output_names();

    ncnn::Extractor ex = net.create_extractor();
    ex.input(input_names[0], in);
    ex.extract(output_names[0], out_temp);

    g_memory_profiler.save(g_memory_profiles);
    //g_unlocked_time_profiler.save(g_time_profiles);

    print_memory_budget(g_memory_profiles, net.layers());

}

void test_kernel(char* mode, char* param_path,char* bin_path)
{
    std::string final_param_path = std::string(param_path);
    std::string final_bin_path = std::string(bin_path);

    if (std::string(mode).find("winograd") != std::string::npos || std::string(mode).find("gemm") != std::string::npos) {
        FlexnnSlice pre_net;

        pre_net.load_param(param_path);
        pre_net.load_model(bin_path);

        transform_conv(pre_net, mode);

        final_param_path = "/home/root/flexnn/FlexNN/models/ncnn/trans_conv.ncnn.param";
        final_bin_path = "/home/root/flexnn/FlexNN/models/ncnn/trans_conv.ncnn.bin";
        pre_net.save(final_param_path.c_str(), final_bin_path.c_str());
    }


    flexnn::MemoryProfiler g_memory_profiler;
    flexnn::MemoryProfilerInterface g_weight_interface;
    flexnn::MemoryProfilerInterface g_blob_interface;
    flexnn::MemoryProfilerInterface g_intermediate_interface;
    flexnn::MemoryProfilerInterface g_persistence_weight_interface;
    flexnn::UnlockedTimeProfiler g_unlocked_time_profiler;
    
    std::vector<MemoryProfilerEvent> g_memory_profiles;
    std::vector<LayerTimeProfile> g_time_profiles;

    g_weight_interface.set_attributes(0, 0);
    g_blob_interface.set_attributes(0, 1);
    g_intermediate_interface.set_attributes(0, 2);
    g_persistence_weight_interface.set_attributes(0, 3);

    g_memory_profiler.add(&g_weight_interface);
    g_memory_profiler.add(&g_blob_interface);
    g_memory_profiler.add(&g_intermediate_interface);
    g_memory_profiler.add(&g_persistence_weight_interface);

    ncnn::Net net;
    ncnn::Option opt;
    
    // common options
    opt.lightmode = true;
    opt.num_threads = g_num_threads;
    opt.use_local_pool_allocator = false;
    opt.use_local_threads = true;

    // conv impl
    opt.use_winograd_convolution = false;
    opt.use_sgemm_convolution = false;
    opt.use_winograd23_convolution = false;
    opt.use_winograd43_convolution = false;
    opt.use_winograd63_convolution = false;
    if (std::string(mode).find("winograd63") != std::string::npos) {
        opt.use_winograd63_convolution = true;
    }
    if (std::string(mode).find("winograd43") != std::string::npos)
    {
        opt.use_winograd43_convolution = true;
    }
    if (std::string(mode).find("winograd23") != std::string::npos)
    {
        opt.use_winograd23_convolution = true;
    }
    opt.use_winograd_convolution = opt.use_winograd23_convolution || opt.use_winograd43_convolution || opt.use_winograd63_convolution;
    opt.use_sgemm_convolution = std::string(mode).find("gemm") != std::string::npos;

    // int8, fp16, packing
    opt.use_int8_inference = false;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_int8_storage = false;
    opt.use_int8_arithmetic = false;
    opt.use_packing_layout = false;

    opt.use_ondemand_loading = true;
    opt.use_pretransform = true;
    opt.use_memory_profiler = true;

    opt.blob_allocator = &g_blob_interface;
    opt.weight_allocator = &g_weight_interface;
    opt.workspace_allocator = &g_intermediate_interface;
    opt.persistence_weight_allocator = &g_persistence_weight_interface;
    opt.time_profiler = &g_unlocked_time_profiler;

    strcpy(opt.mode,mode);

    // omp settings
    ncnn::set_omp_dynamic(0);
    ncnn::set_omp_num_threads(g_num_threads);
    ncnn::set_cpu_powersave(g_computing_powersave); // 1 使用小核 2 使用大核 3 倾向使用中核 

    int w, h, c, d;
    sscanf(input_shape, "[%d,%d,%d,%d]", &d, &c, &h, &w);

    ncnn::Mat in(w, h, d, c, 4u, opt.blob_allocator);
    in.fill(0.01f);
    ncnn::Mat out_temp;

    net.opt = opt;


    net.load_param(final_param_path.c_str());
    net.load_model(final_bin_path.c_str());

    const std::vector<const char*>& input_names = net.input_names();
    const std::vector<const char*>& output_names = net.output_names();

    double time_sum = 0.0;
    long memory_budget = 0;

    for (int i = 0; i < 100; i++) {
        printf("%d\n", i);
        g_unlocked_time_profiler.clear();
        g_memory_profiler.clear();

        ncnn::Extractor ex = net.create_extractor();
        ex.input(input_names[0], in);
        ex.extract(output_names[0], out_temp);

        g_memory_profiler.save(g_memory_profiles);
        g_unlocked_time_profiler.save(g_time_profiles);

        time_sum += get_latency(g_time_profiles);
        memory_budget += get_memory_budget(g_memory_profiles,net.layers().size() - 1);
    }
    
    

    printf("%s spend time : %lf \n memory_budget : %ld", mode, time_sum / 100, memory_budget / 100);
}

int main(int argc, char** argv)
{   
    char mode[256];
    char param_path[256];
    char bin_path[256];

    strcpy(mode, argv[1]);
    strcpy(param_path, argv[2]);
    strcpy(bin_path, argv[3]);

    if (std::string(mode).find("layer_memory_profile") != std::string::npos) {
        profile_layer(param_path, bin_path);
    }
    else {
        test_kernel(mode, param_path, bin_path);
    }
    

	return 0;
}