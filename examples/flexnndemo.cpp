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
#include "powerlogger.h"

static char ncnnparam[256];
static char ncnnbin[256];
static char flexnnparam[256];
static char flexnnbin[256];
static char resultpath[256];
static char datapath[256];

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
static char input_shape[256];
static int w, h, c, d;
static double seenn_threshold = -1;
static int jump = 0;
static bool use_seenn = false;
static int log_num = 1000;
static long long wait_time = 300000000;

// for infer
static flexnn::PlannedAllocator g_planned_allocator;
static flexnn::PlannedAllocatorInterface g_planned_weight_allocator;
static flexnn::PlannedAllocatorInterface g_planned_blob_allocator;
static flexnn::PlannedAllocatorInterface g_planned_intermediate_allocator;
static flexnn::PlannedAllocatorInterface g_planned_persistence_weight_allocator;
static flexnn::LockedTimeProfiler g_locked_time_profiler;

// interfaces
static std::vector<std::vector<int> > g_malloc_offsets;
static std::vector<int> g_persistent_offsets;
static std::vector<int> g_layer_dependencies;
static std::vector<MemoryProfilerEvent> g_memory_profiles;
static std::vector<LayerTimeProfile> g_time_profiles;

// for profiler
static flexnn::MemoryProfiler g_memory_profiler;
static flexnn::MemoryProfilerInterface g_weight_interface;
static flexnn::MemoryProfilerInterface g_blob_interface;
static flexnn::MemoryProfilerInterface g_intermediate_interface;
static flexnn::MemoryProfilerInterface g_persistence_weight_interface;
static flexnn::UnlockedTimeProfiler g_unlocked_time_profiler;

static std::vector<double> starts, ends;
static MyDataLoader loader; //实验室 nx data
static PowerLogger plog = PowerLogger();
static bool is_plog = false;

int startsWith(std::string s, std::string sub)
{
    return s.find(sub) == 0 ? 1 : 0;
}

auto split = [](const std::string& s, char delim) -> std::vector<std::string> {
    printf("try to split config information \n");
    std::vector<std::string> vecRes;
    vecRes.reserve(s.size() / 2); // 预留足够的空间，减少realloc

    std::string strCur;
    for (char c : s)
    {
        if (c == delim)
        {
            if (!strCur.empty())
            { // 忽略空字符串
                vecRes.push_back(move(strCur));
            }
            strCur.clear();
        }
        else
        {
            strCur += c;
        }
    }

    if (!strCur.empty())
    {
        vecRes.push_back(move(strCur)); // 如果 cur 不是空，添加到 vecRes
    }

    printf("split complete! \n");
    return vecRes;
};

//模型切分 conv_mem卷积空间限制 fc_mem全连接空间限制
int run_slice(int conv_mem, int fc_mem)
{
    if (is_plog)
        plog.record_event("run_slice start.");

    //记录开始时间
    double start = flexnn::get_current_time();

    //默认fp32
    int max_conv_size = conv_mem / 4;
    int max_fc_size = fc_mem / 4;

    //新网络 Net->ModelWriter->slicer
    FlexnnSlice slicer;

    slicer.storage_type = 0; //以32位浮点数存储

    //轻量级方法仅解析网络拓扑结构，不执行内存分配和运行时环境初始化
    slicer.load_param_dummy(ncnnparam);

    //尝试读取模型参数文件，若失败则使用DataReaderFromEmpty将参数赋值为0
    if (strcmp(ncnnbin, "null") == 0)
    {
        DataReaderFromEmpty dr;
        slicer.load_model(dr);
        slicer.gen_random_weight = true;
    }
    else
    {
        int ret = slicer.load_model(ncnnbin);
        if (ret)
        {
            DataReaderFromEmpty dr;
            slicer.load_model(dr);
            slicer.gen_random_weight = true;
        }
    }

    // resolve all shapes at first
    //模型形状推理
    slicer.shape_inference();

    //模型切分
    slicer.slice_innerproduct(max_fc_size);

    if (std::string(mode).find("old") != std::string::npos)
    {
        slicer.slice_convolution(max_conv_size);
    }
    else
    {
        slicer.slice_convolution_new(max_conv_size, mode);
    }

    //切分后模型形状推理
    slicer.topological_sort_new();
    slicer.shape_inference();

    slicer.transform_kernel_convolution_new(max_conv_size, g_num_threads);
    
    slicer.save(flexnnparam, flexnnbin);

    double end = flexnn::get_current_time();
    double time = end - start;

    printf("slicing spend time : %lf \n", time);

    return 0;
}

// 收集每层内存分配及推理时间信息
int run_profile()
{
    if (is_plog)
        plog.record_event("run_profile start.");

    printf("start run profile\n");

    double start = flexnn::get_current_time();

    // benchmark configs
    ncnn::Option opt;
    set_benchmark_config(opt, "flexnn_profile", g_num_threads);

    g_memory_profiler.clear();
    g_unlocked_time_profiler.clear();

    // 三个已经注册到g_memory_profiler的allocator负责监控不同类型的内存的分配情况 
    // g_unlocked_time_profiler负责记录每层分配开始和结束时间
    opt.blob_allocator = &g_blob_interface;
    opt.weight_allocator = &g_weight_interface;
    opt.workspace_allocator = &g_intermediate_interface;
    opt.persistence_weight_allocator = &g_persistence_weight_interface;
    opt.time_profiler = &g_unlocked_time_profiler;

    // omp settings
    ncnn::set_omp_dynamic(0);
    ncnn::set_omp_num_threads(g_num_threads);
    ncnn::set_cpu_powersave(g_computing_powersave); // 1 使用小核 2 使用大核 3 倾向使用中核 

    // 正向传播一次进行profile
    ncnn::Mat in = cstr2mat(input_shape,opt);
    in.fill(0.01f);

    ncnn::Mat out_py, out;
    out_py = ncnn::Mat(10, 4u, opt.persistence_weight_allocator);
    out = ncnn::Mat(10, 4u, opt.persistence_weight_allocator);


    ncnn::Mat out_temp;

    ncnn::Net net;

    net.opt = opt;

    net.load_param(flexnnparam);

    if (g_load_model_bin)
    {
        net.load_model(flexnnbin);
    }
    else
    {
        DataReaderFromEmpty dr; // load from empty
        net.load_model(dr);
    }

    const std::vector<const char*>& input_names = net.input_names();
    const std::vector<const char*>& output_names = net.output_names();

    ncnn::Extractor ex = net.create_extractor();
    ex.input(input_names[0], in);
    ex.extract(output_names[0], out_temp);
    //print_one_dim_mat(out_temp);


    {
        flexnn::MemoryProfilerInterface* blob_interface = (flexnn::MemoryProfilerInterface*)opt.blob_allocator;
        flexnn::MemoryProfilerInterface* workspace_interface = (flexnn::MemoryProfilerInterface*)opt.workspace_allocator;
        flexnn::MemoryProfilerInterface* weight_interface = (flexnn::MemoryProfilerInterface*)opt.weight_allocator;
        flexnn::MemoryProfilerInterface* persistence_weight_interface = (flexnn::MemoryProfilerInterface*)opt.persistence_weight_allocator;

        weight_interface->set_attributes(net.layers().size() - 1, "");
        blob_interface->set_attributes(net.layers().size() - 1, "");
        workspace_interface->set_attributes(net.layers().size() - 1, "");
        persistence_weight_interface->set_attributes(net.layers().size() - 1, "");
    }

    in.release();
    out_temp.release();
    out_py.release();
    out.release();
    ex.clear();
    net.clear();

    g_memory_profiler.save(g_memory_profiles);
    g_unlocked_time_profiler.save(g_time_profiles);

    double end = flexnn::get_current_time();
    double time = end - start;

    printf("profile spend time : %lf \n", time);

    return 0;
}

//进行preload-aware memory planning
int run_schedule(int memory_budget)
{
    if (is_plog)
        plog.record_event("run_schedule start.");

    printf("start run schedule,budget:%d\n", memory_budget);
    double start = flexnn::get_current_time();

    //通过FlexnnSchedule类进行planning
    FlexnnSchedule scheduler;
    
    scheduler.set_memory_profiles(g_memory_profiles);
    scheduler.set_time_profiles(g_time_profiles);

    if (std::string(mode).find("old") != std::string::npos)
    {
        scheduler.schedule_naive(memory_budget, resultpath);
    }
    else {
        int res = scheduler.schedule_naive_new(memory_budget, resultpath, mode,jump);
        if (res < 0)
        {
            return -1;
        }
    }
    
    scheduler.get_malloc_plan(g_malloc_offsets, g_persistent_offsets);
    scheduler.get_layer_dependencies(g_layer_dependencies);
    double end = flexnn::get_current_time();
    double time = end - start;

    printf("schedule spend time : %lf \n", time);

    return 0;
}

int run_infer_ncnn_random_data(int memory_budget, int loop, std::string mode)
{
    if (is_plog)
        plog.record_event("run_infer prepare.");

    printf("start run_infer\n");

    double inf_start = flexnn::get_current_time();

    // benchmark configs
    ncnn::Option opt;
    set_benchmark_config(opt, mode.c_str(),g_num_threads);

    // omp settings
    ncnn::set_omp_dynamic(0);
    ncnn::set_omp_num_threads(g_num_threads);
    ncnn::set_cpu_powersave(g_computing_powersave);

    printf("prepare Mat structure\n");
    int w, h, c, d;
    sscanf(input_shape, "[%d,%d,%d,%d]", &d, &c, &h, &w);

    ncnn::Mat true_in = ncnn::Mat(w, h, d, c, 4ull);
    ncnn::Mat py_out = ncnn::Mat(10);
    ncnn::Mat out_c = ncnn::Mat(10);

    ncnn::Mat out_tmp;

    ncnn::Net net;

    net.opt = opt;

    double load_start = flexnn::get_current_time();

    printf("load_param\n");
    net.load_param(ncnnparam);

    printf("load_model_lif_first\n");
    net.load_model_lif_first(ncnnbin);

    double load_end = flexnn::get_current_time();

    std::vector<ncnn::LIFNode*> lifnodes;

    std::vector<ncnn::Layer*> net_layers = net.layers();

    for (int i = 0; i < net_layers.size(); i++)
    {
        if (net_layers[i]->type == "LIFNode")
        {
            lifnodes.push_back((ncnn::LIFNode*)net_layers[i]);
        }
    }

    const std::vector<const char*>& input_names = net.input_names();
    const std::vector<const char*>& output_names = net.output_names();

    cooling_down(g_enable_cooling_down, g_cooling_down_duration); // cooling down if enabled

    printf("reset lifnode\n");
    for (int k = 0; k < lifnodes.size(); k++)
    {
        lifnodes[k]->reset();
    }

    double time_min = DBL_MAX;
    double time_max = -DBL_MAX;
    double time_avg = 0;
    double t = 0;
    int count = 0;
    int counter[100] = {0};

    if (is_plog) {
        plog.record_event("run_infer start.");
    }
        
    
    for (int l = 0; l < loop; l++)
    {
        printf("loop %d start\n",l);
        if (is_plog)
            plog.record_event("No." + std::to_string(l) + " start.");

        double start = flexnn::get_current_time();
        starts.push_back(start);

        out_c.fill(0.0);

        printf("load data\n");
        loader.loadData(true_in, opt);

        double start_each = flexnn::get_current_time();
        if (count % log_num == 0 )
        {
            printf("%d / %d.\n", count, l);
        }

        for (int sub_t = 0; sub_t < T; sub_t++)
        {
            ncnn::Extractor ex = net.create_extractor();
            ex.input(input_names[0], true_in);

            ex.extract(output_names[0], out_tmp);

            //print_one_dim_mat(out_tmp);

            for (int c = 0; c < 1; c++)
            {
                float* ptr_out = out_c.channel(c);
                float* ptr_tmp = out_tmp.channel(c);

                int size = out_tmp.w * out_tmp.h;

                for (int s = 0; s < size; s++)
                {
                    *ptr_out = *ptr_out + *ptr_tmp;

                    ptr_out++;
                    ptr_tmp++;
                }
            }

            //seenn
            if (use_seenn)
            {
                std::vector<float> temp(10);
                int w = out_tmp.w;
                float* ptr = out_tmp;

                float sum = 0.f;
                float max = -FLT_MAX;
                int max_idx = -1;
                for (int i = 0; i < w; i++)
                {
                    if (max < ptr[i]) {
                        max = ptr[i];
                        max_idx = i;
                    }
                }

                counter[max_idx]++;

                if (float(counter[max_idx]) / float(T) >= seenn_threshold)
                {
                    t += (sub_t + 1);
                    for (int c = 0; c < 100; c++)
                    {
                        counter[c] = 0;
                    }
                    break;
                }

            }
        }
        if (!use_seenn)
        {
            t += T;
        }

        printf("lifnode reset\n");
        for (int k = 0; k < lifnodes.size(); k++)
        {
            lifnodes[k]->reset();
        }

        count++;

        out_c.fill(0.0);

        double end = flexnn::get_current_time();

        ends.push_back(end);

        double time = end - start;

        time_min = std::min(time_min, time);
        time_max = std::max(time_max, time);
        time_avg += time;

        if (count % log_num == 0)
            printf("infer No.%d --->current time : %lf;\n time_min : %lf ;\n time_max : %lf ;\n average t : %lf \n", count, time, time_min, time_max, (double)t / (double)count);
        
        if (is_plog)
            plog.record_event("this inference end.");
    }

    printf("Mat struct release\n");
    py_out.release();
    true_in.release();
    out_c.release();
    out_tmp.release();
    net.clear();

    time_avg /= loop;

    double inf_end = flexnn::get_current_time();
    fprintf(stderr, "time_avg : %lf;\ninference start: %.3f s, ends: %.3f s\n", time_avg, inf_start, inf_end);
    return 0;
}

int run_infer_flexnn(int memory_budget, int loop)
{
    printf("start run_infer\n");

    double inf_start = flexnn::get_current_time();

    // benchmark configs
    ncnn::Option opt;
    set_benchmark_config(opt, "flexnn_parallel", g_num_threads);

    // omp settings
    ncnn::set_omp_dynamic(0);
    ncnn::set_omp_num_threads(g_num_threads);
    ncnn::set_cpu_powersave(g_computing_powersave);

    opt.weight_allocator = &g_planned_weight_allocator;
    opt.blob_allocator = &g_planned_blob_allocator;
    opt.workspace_allocator = &g_planned_intermediate_allocator;
    opt.persistence_weight_allocator = &g_planned_persistence_weight_allocator;

    g_planned_allocator.init_buffer(memory_budget); //设置内存限制
    g_planned_allocator.set_malloc_plan(g_malloc_offsets, g_persistent_offsets); //设置内存分配计划和持久性内存分配计划
    opt.layer_dependencies = &g_layer_dependencies; //层间依赖


    g_planned_allocator.clear();

    ncnn::Mat true_in = cstr2mat(input_shape, opt);
    ncnn::Mat py_out, out_c;

    py_out = ncnn::Mat(10, 4u, opt.persistence_weight_allocator);
    out_c = ncnn::Mat(10, 4u, opt.persistence_weight_allocator);

   
    ncnn::Mat out_tmp;
    int target, py_predicted;

    ncnn::Net net;

    net.opt = opt;

    double load_start = flexnn::get_current_time();

    net.load_param(flexnnparam);


    // load persistent weights if have
    g_planned_allocator.set_load_mode(0);
    ncnn::Option opt2 = opt;
    opt2.use_parallel_preloading = false;
    opt2.use_ondemand_loading = false;

    net.opt = opt2;
    if (g_load_model_bin)
    {
        if (net.load_model(flexnnbin))
        {
            // fall back to empty
            DataReaderFromEmpty dr; // load from empty
            net.load_model(dr);
        }
    }
    else
    {
        DataReaderFromEmpty dr; // load from empty
        net.load_model(dr);
    }


    // reset opt
    net.opt = opt;
    // reset mode

    g_planned_allocator.set_load_mode(1);
    g_planned_allocator.clear();

    if (std::string(mode).find("old") != std::string::npos)
    {
        if (true_in.empty())
        {
            true_in = cstr2mat(input_shape, opt);
        }
        else
        {
            size_t cstep = alignSize((size_t)32 * 32 * 4, 16) / 4;
            size_t totalsize = alignSize(cstep * 3 * 4, 4);
            opt.blob_allocator->fastMalloc(totalsize + 4);
        }
    }

    if (py_out.empty())
    {
        py_out = ncnn::Mat(10, 4u, opt.persistence_weight_allocator);
    }
    else
    {
        size_t cstep = alignSize((size_t)10 * 4, 16) / 4;
        size_t totalsize = alignSize(cstep * 1 * 4, 4);
        opt.persistence_weight_allocator->fastMalloc(totalsize + 4);
    }
    if (out_c.empty())
    {
        out_c = ncnn::Mat(10, 4u, opt.persistence_weight_allocator);
    }
    else
    {
        size_t cstep = alignSize((size_t)10 * 4, 16) / 4;
        size_t totalsize = alignSize(cstep * 1 * 4, 4);
        opt.persistence_weight_allocator->fastMalloc(totalsize + 4);
    }


    if (g_load_model_bin)
    {
        if (net.load_model(flexnnbin))
        {
            // fall back to empty
            DataReaderFromEmpty dr; // load from empty
            net.load_model(dr);
        }
    }
    else
    {
        DataReaderFromEmpty dr; // load from empty
        net.load_model(dr);
    }

    if (opt.use_parallel_preloading && opt.use_local_threads)
    {
        net.initialize_local_threads(g_computing_powersave, g_loading_powersave);
    }

    double load_end = flexnn::get_current_time();

    std::vector<ncnn::LIFNode*> lifnodes;

    std::vector<ncnn::Layer*> net_layers = net.layers();

    for (int i = 0; i < net_layers.size(); i++) {
        if (net_layers[i]->type == "LIFNode") {
            lifnodes.push_back((ncnn::LIFNode*)net_layers[i]);
        }
    }

    const std::vector<const char*>& input_names = net.input_names();
    const std::vector<const char*>& output_names = net.output_names();

    cooling_down(g_enable_cooling_down, g_cooling_down_duration); // cooling down if enabled

    for (int k = 0; k < lifnodes.size(); k++)
    {
        lifnodes[k]->reset();
    }

    double time_min = DBL_MAX;
    double time_max = -DBL_MAX;
    double time_avg = 0;
    double t = 0;

    for (int l = 0; l < loop; l++) {

        double start = flexnn::get_current_time();
        starts.push_back(start);

        out_c.fill(0.0);

        int count = 0, top_one_correct_with_py = 0, top_one_correct_with_target = 0;
        while (loader.loadData(true_in, opt, target, py_out, py_predicted) != -1)
        {
            double start_each = flexnn::get_current_time();
            if (count % log_num == 0 && count != 0) {
                printf("%d / 10000.\n", count);
            }
            
            for (int sub_t = 0; sub_t < T; sub_t++)
            {
                /*if (i == 0) {
                    print_one_dim_mat(out_c);
                }*/
                g_planned_allocator.clear();

                if (std::string(mode).find("old") != std::string::npos)
                {
                    if (true_in.empty())
                    {
                        true_in = cstr2mat(input_shape, opt);
                    }
                    else
                    {
                        size_t cstep = alignSize((size_t)32 * 32 * 4, 16) / 4;
                        size_t totalsize = alignSize(cstep * 3 * 4, 4);
                        opt.blob_allocator->fastMalloc(totalsize + 4);
                    }
                    
                }
                

                if (py_out.empty())
                {
                    py_out = ncnn::Mat(10, 4u, opt.persistence_weight_allocator);
                }
                else
                {
                    size_t cstep = alignSize((size_t)10 * 4, 16) / 4;
                    size_t totalsize = alignSize(cstep * 1 * 4, 4);
                    opt.persistence_weight_allocator->fastMalloc(totalsize + 4);
                }
                if (out_c.empty())
                {
                    out_c = ncnn::Mat(10, 4u, opt.persistence_weight_allocator);
                }
                else
                {
                    size_t cstep = alignSize((size_t)10 * 4, 16) / 4;
                    size_t totalsize = alignSize(cstep * 1 * 4, 4);
                    opt.persistence_weight_allocator->fastMalloc(totalsize + 4);
                }

                ncnn::Extractor ex = net.create_extractor();
                ex.input(input_names[0], true_in);

                ex.extract(output_names[0], out_tmp);

                //print_one_dim_mat(out_tmp);

                for (int c = 0; c < 1; c++) {
                    float* ptr_out = out_c.channel(c);
                    float* ptr_tmp = out_tmp.channel(c);

                    int size = out_tmp.w * out_tmp.h;

                    for (int s = 0; s < size; s++) {
                        *ptr_out = *ptr_out + *ptr_tmp;

                        ptr_out++;
                        ptr_tmp++;
                    }
                }

                //seenn
                if (use_seenn)
                {
                    std::vector<float> temp(10);
                    int w = out_c.w;
                    float* ptr = out_c;

                    float sum = 0.f;
                    float max = -FLT_MAX;
                    for (int i = 0; i < w; i++)
                    {
                        max = std::max(max, ptr[i]);
                    }
                    for (int i = 0; i < w; i++)
                    {
                        temp[i] = static_cast<float>(exp(ptr[i] - max));
                        sum += temp[i];
                    }

                    for (int i = 0; i < w; i++)
                    {
                        temp[i] /= sum;
                    }

                    std::sort(temp.begin(), temp.end(), [](float a, float b) { return a > b; });

                    if (temp[0] >= seenn_threshold) {
                        t += (sub_t + 1);
                        break;
                    }
                }
            }
            if (!use_seenn)
            {
                t += T;
            }


            for (int c = 0; c < 1; c++)
            {
                float* ptr_out = out_c.channel(c);
                for (int s = 0; s < 10; s++)
                {
                    *ptr_out = *ptr_out / T; 

                    ptr_out++;
                }
            }

            for (int k = 0; k < lifnodes.size(); k++) {
                lifnodes[k]->reset();
            }

            int max_idx = 0;
            
            float max_score = -100;
            float* ptr = out_c.channel(0);

            for (int i = 0; i < 10;  i++) {
                if (*ptr > max_score) {
                    max_score = *ptr;
                    max_idx = i;
                }

                ptr++;
            }

            double end_each = flexnn::get_current_time();
            double time = end_each - start_each;


            if (max_idx == py_predicted) {
                if (count % log_num == 0 && count != 0) {
                    printf("same to py.\n");
                }
                    
                top_one_correct_with_py++;
            }

            if (max_idx == target)
            {   
                if (count % log_num == 0 && count != 0) {
                    printf("same to target.\n");
                }
                top_one_correct_with_target++;
            }

            count++;

            out_c.fill(0.0);
        }

        printf("infer ---> top_one_correct_with_py : %.6f \n top_one_correct_with_target : %.6f \n", (double)top_one_correct_with_py / (double)count, (double)top_one_correct_with_target / (double)count);

        double end = flexnn::get_current_time();

        ends.push_back(end);

        double time = end - start;

        time_min = std::min(time_min, time);
        time_max = std::max(time_max, time);
        time_avg += time;
        t /= (double)count;

        printf("infer --->current time : %lf;\n time_min : %lf ;\n time_max : %lf ;\n average t : %lf \n",time, time_min, time_max, t);

        std::rewind(loader.file);
    }

    if (opt.use_parallel_preloading)
    {
        net.clear_local_threads();
    }

    //out.release();
    py_out.release();
    true_in.release();
    out_c.release();
    out_tmp.release();
    net.clear();
    g_planned_allocator.release_buffer();

    time_avg /= loop;

    double inf_end = flexnn::get_current_time();
    fprintf(stderr, "time_avg : %lf;\ninference start: %.3f s, ends: %.3f s\n", time_avg, inf_start, inf_end);
    return 0;
}

int run_infer_flexnn_random_data(int memory_budget, int loop)
{
    if (is_plog)
        plog.record_event("run_infer prepare.");

    printf("start run_infer\n");

    double inf_start = flexnn::get_current_time();

    // benchmark configs
    ncnn::Option opt;

    set_benchmark_config(opt, "flexnn_parallel", g_num_threads);

    // omp settings
    ncnn::set_omp_dynamic(0);
    ncnn::set_omp_num_threads(g_num_threads);
    ncnn::set_cpu_powersave(g_computing_powersave);

    opt.weight_allocator = &g_planned_weight_allocator;
    opt.blob_allocator = &g_planned_blob_allocator;
    opt.workspace_allocator = &g_planned_intermediate_allocator;
    opt.persistence_weight_allocator = &g_planned_persistence_weight_allocator;

    g_planned_allocator.init_buffer(memory_budget);                              //设置内存限制
    g_planned_allocator.set_malloc_plan(g_malloc_offsets, g_persistent_offsets); //设置内存分配计划和持久性内存分配计划
    opt.layer_dependencies = &g_layer_dependencies;                              //层间依赖

    g_planned_allocator.clear();

    printf("prepare Mat structure\n");
    ncnn::Mat true_in = cstr2mat(input_shape, opt);
    ncnn::Mat py_out, out_c;

    py_out = ncnn::Mat(10, 4u, opt.persistence_weight_allocator);
    out_c = ncnn::Mat(10, 4u, opt.persistence_weight_allocator);

    ncnn::Mat out_tmp;

    ncnn::Net net;

    net.opt = opt;

    double load_start = flexnn::get_current_time();

    printf("load_param\n");
    net.load_param(flexnnparam);

    // load persistent weights if have
    g_planned_allocator.set_load_mode(0);
    ncnn::Option opt2 = opt;
    opt2.use_parallel_preloading = false;
    opt2.use_ondemand_loading = false;

    net.opt = opt2;
    if (g_load_model_bin)
    {
        if (net.load_model(flexnnbin))
        {
            // fall back to empty
            DataReaderFromEmpty dr; // load from empty
            net.load_model(dr);
        }
    }
    else
    {
        DataReaderFromEmpty dr; // load from empty
        net.load_model(dr);
    }

    // reset opt
    net.opt = opt;
    // reset mode

    g_planned_allocator.set_load_mode(1);
    g_planned_allocator.clear();

    sscanf(input_shape, "[%d,%d,%d,%d]", &d, &c, &h, &w);

    if (std::string(mode).find("old") != std::string::npos)
    {
        if (true_in.empty())
        {
            true_in = cstr2mat(input_shape, opt);
        }
        else
        {
            size_t cstep = alignSize((size_t) w * h * 4, 16) / 4;
            size_t totalsize = alignSize(cstep * c * 4, 4);
            opt.persistence_weight_allocator->fastMalloc(totalsize + 4);
        }
    }

    if (py_out.empty())
    {
        py_out = ncnn::Mat(10, 4u, opt.persistence_weight_allocator);
    }
    else
    {
        size_t cstep = alignSize((size_t)10 * 4, 16) / 4;
        size_t totalsize = alignSize(cstep * 1 * 4, 4);
        opt.persistence_weight_allocator->fastMalloc(totalsize + 4);
    }
    if (out_c.empty())
    {
        out_c = ncnn::Mat(10, 4u, opt.persistence_weight_allocator);
    }
    else
    {
        size_t cstep = alignSize((size_t)10 * 4, 16) / 4;
        size_t totalsize = alignSize(cstep * 1 * 4, 4);
        opt.persistence_weight_allocator->fastMalloc(totalsize + 4);
    }

    if (g_load_model_bin)
    {
        if (net.load_model(flexnnbin))
        {
            // fall back to empty
            DataReaderFromEmpty dr; // load from empty
            net.load_model(dr);
        }
    }
    else
    {
        DataReaderFromEmpty dr; // load from empty
        net.load_model(dr);
    }

    if (opt.use_parallel_preloading && opt.use_local_threads)
    {
        net.initialize_local_threads(g_computing_powersave, g_loading_powersave);
    }

    double load_end = flexnn::get_current_time();

    std::vector<ncnn::LIFNode*> lifnodes;

    std::vector<ncnn::Layer*> net_layers = net.layers();

    for (int i = 0; i < net_layers.size(); i++)
    {
        if (net_layers[i]->type == "LIFNode")
        {
            lifnodes.push_back((ncnn::LIFNode*)net_layers[i]);
        }
    }

    const std::vector<const char*>& input_names = net.input_names();
    const std::vector<const char*>& output_names = net.output_names();
    printf("input_names size %d\n", input_names.size());
    printf("output_names size %d\n", output_names.size());

    cooling_down(g_enable_cooling_down, g_cooling_down_duration); // cooling down if enabled

    printf("reset lifnode\n");
    for (int k = 0; k < lifnodes.size(); k++)
    {
        lifnodes[k]->reset();
    }

    double time_min = DBL_MAX;
    double time_max = -DBL_MAX;
    double time_avg = 0;
    double t = 0;
    int count = 0;
    int counter[100] = {0};

    if (is_plog)
        plog.record_event("run_infer start.");
    for (int l = 0; l < loop; l++)
    {
        printf("loop %d start\n", l);
        if (is_plog)
            plog.record_event("No." + std::to_string(l) + " start.");
        double start = flexnn::get_current_time();
        starts.push_back(start);

        out_c.fill(0.0);

        printf("load data\n");
        loader.loadData(true_in, opt);

        double start_each = flexnn::get_current_time();
        if (count % log_num == 0 && count != 0)
        {
            printf("%d / %d.\n", count, l);
        }

        for (int sub_t = 0; sub_t < T; sub_t++)
        {
            g_planned_allocator.clear();

            if (std::string(mode).find("old") != std::string::npos)
            {
                if (true_in.empty())
                {
                    true_in = cstr2mat(input_shape, opt);
                }
                else
                {
                    size_t cstep = alignSize((size_t)w * h * 4, 16) / 4;
                    size_t totalsize = alignSize(cstep * c * 4, 4);
                    opt.persistence_weight_allocator->fastMalloc(totalsize + 4);
                }
            }

            if (py_out.empty())
            {
                py_out = ncnn::Mat(10, 4u, opt.persistence_weight_allocator);
            }
            else
            {
                size_t cstep = alignSize((size_t)10 * 4, 16) / 4;
                size_t totalsize = alignSize(cstep * 1 * 4, 4);
                opt.persistence_weight_allocator->fastMalloc(totalsize + 4);
            }
            if (out_c.empty())
            {
                out_c = ncnn::Mat(10, 4u, opt.persistence_weight_allocator);
            }
            else
            {
                size_t cstep = alignSize((size_t)10 * 4, 16) / 4;
                size_t totalsize = alignSize(cstep * 1 * 4, 4);
                opt.persistence_weight_allocator->fastMalloc(totalsize + 4);
            }

            ncnn::Extractor ex = net.create_extractor();
            ex.input(input_names[0], true_in);
            ex.extract(output_names[0], out_tmp);

            //print_one_dim_mat(out_tmp);

            for (int c = 0; c < 1; c++)
            {
                float* ptr_out = out_c.channel(c);
                float* ptr_tmp = out_tmp.channel(c);

                int size = out_tmp.w * out_tmp.h;

                for (int s = 0; s < size; s++)
                {
                    *ptr_out = *ptr_out + *ptr_tmp;

                    ptr_out++;
                    ptr_tmp++;
                }
            }

            //seenn
            if (use_seenn)
            {
                printf("seenn process start\n");
                std::vector<float> temp(10);
                int w = out_tmp.w;
                float* ptr = out_tmp;

                float sum = 0.f;
                float max = -FLT_MAX;
                int max_idx = -1;
                for (int i = 0; i < w; i++)
                {
                    if (max < ptr[i])
                    {
                        max = ptr[i];
                        max_idx = i;
                    }
                }

                counter[max_idx]++;

                if (float(counter[max_idx]) / float(T) >= seenn_threshold)
                {
                    t += (sub_t + 1);
                    
                    for (int c = 0; c < 100; c++)
                    {
                        counter[c] = 0;
                    }

                    break;
                }
            }
        }
        if (!use_seenn)
        {
            t += T;
        }

        printf("lifnode reset\n");
        for (int k = 0; k < lifnodes.size(); k++)
        {
            lifnodes[k]->reset();
        }

        count++;

        out_c.fill(0.0);

        double end = flexnn::get_current_time();

        ends.push_back(end);

        double time = end - start;

        time_min = std::min(time_min, time);
        time_max = std::max(time_max, time);
        time_avg += time;

        if (count % log_num == 0)
            printf("infer No.%d --->current time : %lf;\n time_min : %lf ;\n time_max : %lf ;\n average t : %lf \n", count, time, time_min, time_max, (double)t / (double)count);
    
        if (is_plog)
            plog.record_event("this inference end.");
    }

    if (opt.use_parallel_preloading)
    {
        net.clear_local_threads();
    }

    printf("Mat struct release\n");
    py_out.release();
    true_in.release();
    out_c.release();
    out_tmp.release();
    net.clear();
    g_planned_allocator.release_buffer();

    time_avg /= loop;

    double inf_end = flexnn::get_current_time();
    fprintf(stderr, "time_avg : %lf;\n inference start: %.3f s, ends: %.3f s\n", time_avg, inf_start, inf_end);
    return 0;
}

int main(int argc, char** argv)
{
    fprintf(stderr, "Usage: %s --<ncnn_param> <ncnn_bin> <flexnn_param> <input_shape> <flexnn_bin> <result_path> <data_path> <conv_sz> <fc_sz> <memory_budgets> <loop_num> <log_num> <mode> <timestep> <record> <seenn_threshold> <idle_duration>\n", argv[0]);

    const long long m = 1e6; 

    std::map<std::string,std::string> argMap;
    std::vector<std::string> args(argv+1,argv+argc);

    for (size_t i = 0; i < args.size(); i++) {
        if (startsWith(args[i], "--"))
        {
            std::string param = args[i].substr(2);
            size_t pos = param.find_first_of(" ");
            std::string key = param.substr(0, pos);
            std::string val = param.substr(pos + 1);
            if (!val.empty()) {
                argMap[key] = val;
            }
            else {
                argMap[key] = "";
            }
        }
    }

    strcpy(resultpath, argMap["result_path"].c_str());

    if (argMap["record"] != "")
    {
        is_plog = true;
        int log_level = std::atoi(argMap["record"].c_str());
        double sampletime = 1000;
        if (argMap.count("sampletime") != 0 && argMap["sampletime"] != "")
        {
            sampletime = std::atof(argMap["sampletime"].c_str());
            printf("get sampletime : %ld .\n", sampletime);
        }

        printf("will record memory and power consumption info by %ld ms.", sampletime);
        plog = PowerLogger(sampletime, log_level, std::string(resultpath));
        printf("info result will be saved to %s .", resultpath);

        plog.start();
    }

    strcpy(ncnnparam, argMap["ncnn_param"].c_str()); //初始网络结构
    strcpy(ncnnbin, argMap["ncnn_bin"].c_str());      //初始权重参数
    strcpy(flexnnparam, argMap["flexnn_param"].c_str()); //网络结构保存地址
    strcpy(flexnnbin, argMap["flexnn_bin"].c_str());     //权重参数保存地址
    strcpy(input_shape, argMap["input_shape"].c_str());

    if (argMap.count("data_path") == 0) {
        strcpy(datapath, "");
    } else {
        strcpy(datapath, argMap["data_path"].c_str());
    }

    loader = MyDataLoader(datapath, input_shape);

    int conv_sz, fc_sz;
    sscanf(argMap["conv_sz"].c_str(), "%d", &conv_sz);
    sscanf(argMap["fc_sz"].c_str(), "%d", &fc_sz);
    printf("get conv_sz : %d  and fc_sz : %d .\n", conv_sz, fc_sz);

    int num_configs = 0; //测试轮次
    int loops[10] = {0}; //前向传播次数
    int memory_budgets[10] = {0}; //各轮次内存限制
    int loop_num = 5;
    int idle_duration = 0; //冷却时间

    if (argMap["loop_num"] != "")
    {
        sscanf(argMap["loop_num"].c_str(), "%d", &loop_num);
        printf("get loop_num : %d .\n", loop_num);
    }

    if (argMap["log_num"] != "")
    {
        sscanf(argMap["log_num"].c_str(), "%d", &log_num);
        printf("get log_num : %d .\n", log_num);
    }

    std::vector<std::string> vecSplitString = split(std::string(argMap["memory_budgets"]), ',');
    num_configs = vecSplitString.size();

    printf("there is %d configs to run \n", num_configs);
    for (int i = 0; i < num_configs; i++)
    {
        
        loops[i] = loop_num;
        memory_budgets[i] = std::atoi(vecSplitString[i].c_str()) * m; 
        printf(" %d : %d run loop %d\n", i + 1, memory_budgets[i], loops[i]);
    }

    if (argMap["mode"] != "")
    {
        strcpy(mode, argMap["mode"].c_str()); 
    }
    else {
        strcpy(mode, "normal"); 
    }
    printf("mode is %s \n", mode);

    if (argMap["timestep"] != "")
    {
        T = std::atoi(argMap["timestep"].c_str());
        printf("get timestep : %d .\n", T);
    }

    if (argMap["waittime"] != "")
    {
        wait_time = std::atoi(argMap["waittime"].c_str());
        printf("get wait_time : %d .\n", wait_time);
    }

    if (argMap["jump"] != "")
    {
        jump = std::atoi(argMap["jump"].c_str());
        printf("get jump : %d .\n", jump);
    }

    

    if (argMap["seenn_threshold"] != "")
    {
        seenn_threshold = atof(argMap["seenn_threshold"].c_str());
        use_seenn = true;
        printf("get seenn_threshold : %lf .\n", seenn_threshold);
    }

    if (argMap["idle_duration"] != "")
    {
        idle_duration = atoi(argMap["idle_duration"].c_str());
        printf("get idle_duration : %d .\n", idle_duration);
    }

    // init
    //三个 Allocator，检测内存分配情况 weights、activations、intermediates
    g_weight_interface.set_attributes(0, 0);
    g_blob_interface.set_attributes(0, 1);
    g_intermediate_interface.set_attributes(0, 2);
    g_persistence_weight_interface.set_attributes(0, 3);
    

    //注册进总分析器，记录分配事件
    g_memory_profiler.add(&g_weight_interface);
    g_memory_profiler.add(&g_blob_interface);
    g_memory_profiler.add(&g_intermediate_interface);
    g_memory_profiler.add(&g_persistence_weight_interface);

    g_planned_weight_allocator.set_attributes(0);
    g_planned_blob_allocator.set_attributes(1);
    g_planned_intermediate_allocator.set_attributes(2);
    g_planned_persistence_weight_allocator.set_attributes(3);
    g_planned_allocator.add(&g_planned_weight_allocator);
    g_planned_allocator.add(&g_planned_blob_allocator);
    g_planned_allocator.add(&g_planned_intermediate_allocator);
    g_planned_allocator.add(&g_planned_persistence_weight_allocator);

    cooling_down(idle_duration > 0, idle_duration); // idle memory measure

    long long wt = 0;
    while (wt < wait_time)
    {
        printf("");
        wt++;
    }
    
    if (std::string(mode).find("normal") != std::string::npos) {
        run_slice(conv_sz * m, fc_sz * m); //模型切分——bottleneck-aware layer slicing

        for (int i = 0; i < num_configs; i++)
        {
            //preload-aware memory planning
            run_profile();
            int res = run_schedule(memory_budgets[i]);
            if (res < 0)
            {
                printf("run schedule wrong!\n");
                return 0;
            }

            //online executioon
            printf("Loop No.%d , current memory_budget is %d \n", i, memory_budgets[i]);

            if (loader.file == NULL)
            {
                run_infer_flexnn_random_data(memory_budgets[i], loops[i]);
            }
            else
            {
                run_infer_flexnn(memory_budgets[i], loops[i]);
            }

            long long wt = 0;
            while (wt < wait_time)
            {
                printf("");
                wt++;
            }
        }
    }
    else {
        for (int i = 0; i < num_configs; i++)
        {
            run_infer_ncnn_random_data(memory_budgets[i], loops[i], mode);

            long long wt = 0;
            while (wt < wait_time)
            {
                printf("");
                wt++;
            }
        }
    }
   

    if (is_plog) {
        plog.record_event("run_infer done, prepare to analyse and save.");

        long long wt = 0;
        while (wt < wait_time)
        {
            printf("");
            wt++;
        }


        plog.stop();
        std::vector<double> powers = plog.get_total_energy();

        printf("vdd_in : %lf ; vdd_cpu_gpu_cv : %lf ; vdd_soc : %lf ;\n", powers[0] / loop_num, powers[1] / loop_num, powers[2] / loop_num);

        plog.save();

        std::vector<long> max_infer_memory = plog.get_max_infer_memory_usage();

        for (int c = 0; c < num_configs; c++)
        {
            printf("max inference memory usage of config %d : %ld MB\n", c + 1, max_infer_memory[c]);
        }
        printf("max inference memory usage : %ld\n",plog.get_max_memory_usage());

        plog.clear();
    }
    

    return 0;
}