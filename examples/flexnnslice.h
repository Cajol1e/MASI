#ifndef FLEXNN_SLICE_H
#define FLEXNN_SLICE_H
#ifdef _MSC_VER
#define _CRT_SECURE_NO_DEPRECATE
#endif

#include <algorithm>
#include <map>
#include <set>
#include <vector>
#include <stack>
#include <queue>

// ncnn public header
#include "datareader.h"
#include "modelbin.h"
#include "layer.h"
#include "layer_type.h"

// ncnn private header
#include "modelwriter.h"

// flexnn utils
#include "flexnn_utils.h"
#include "dummymat.h"
#include "transformutils.h"

#include "layer/convolution.h"
#include <layer/memorydata.h>

class FlexnnSlice : public ModelWriter
{
public:
    FlexnnSlice();

    // main logic
    int slice_innerproduct(int max_data_size); // max_mem_size = max_data_size * element_size
    int slice_convolution_new(int max_data_size,char* mode);
    int slice_convolution(int max_data_size);  // max_mem_size = max_data_size * element_size
    int transform_kernel_convolution(int max_data_size);
    int slice_gemm();

    // layer operations
    int slice_innerproduct_outsz(int layer_index, int max_size); // max_size = max_outsz_per_slice

    int slice_convolution_outch(int layer_index, int max_size); // max_size = max_outsz_per_slice

    int transform_kernel_convolution_new(int max_data_size, int thread_num);

    int transform_kernel_convolution(int max_data_size, int thread_num);

    int slice_convolution_im2col_sgemm_inch(int layer_index);
    int slice_convolution_im2col_sgemm_outsz(int layer_index);
    int slice_convolution_im2col_sgemm_mixed(int layer_index);

    int slice_convolution_winograd_inch(int layer_index);
    int slice_convolution_winograd_outsz(int layer_index);
    int slice_convolution_winograd_mixed(int layer_index);

    int transform_kernel_convolution_winograd63(int layer_index);
    int transform_kernel_convolution_winograd63(int layer_index, int thread_num);
    int transform_kernel_convolution_winograd43(int layer_index);
    int transform_kernel_convolution_winograd43(int layer_index, int thread_num);
    int transform_kernel_convolution_winograd23(int layer_index);
    int transform_kernel_convolution_winograd23(int layer_index, int thread_num);

    int transform_kernel_convolution_winograd63_arm(int layer_index, int thread_num);

    int transform_kernel_convolution_winograd43_arm(int layer_index, int thread_num);

    int transform_kernel_convolution_im2col_gemm(int layer_index);
    int transform_kernel_convolution_im2col_gemm(int layer_index, int thread_num);

    int transform_kernel_convolution_winograd23_arm(int layer_index, int thread_num);

    int transform_kernel_convolution_3x3s2(int layer_index);

    // graph
    int topological_sort();
    int topological_sort_new();
    int shape_inference();

public:
    int get_size_convolution(int layer_index, const char* type, int nT = 1) const;
    int get_slice_outch_convolution(int layer_index, const char* type, int max_size, int nT = 1) const;
    // int get_size_convolution_winograd63(int layer_index, int nT = 1) const;
private:
    // first:layer_index second:convolution_algorithm --> 0:auto 1:winograd23 2:winograd43 3:winograd63 4:gemm
    std::map<int, int> slice_type_recorder;
};

int FlexnnSlice::get_slice_outch_convolution(int layer_index, const char* type, int max_size, int nT) const
{
    const ncnn::Layer* layer = layers[layer_index];
    if (layer->type != "Convolution")
    {
        fprintf(stderr, "Error: layer %d %s is not convolution\n", layer_index, layers[layer_index]->name.c_str());
        return -1;
    }

    int top_blob_index = layer->tops[0];
    int bottom_blob_index = layer->bottoms[0];

    const ncnn::Convolution* convolution = (const ncnn::Convolution*)layer;

    const flexnn::DummyMat in = blobs[bottom_blob_index].dummy_shape;
    const flexnn::DummyMat out = blobs[top_blob_index].dummy_shape;

    int kernel_size = convolution->kernel_w * convolution->kernel_h;
    int maxk = kernel_size;

    int outw = out.w;
    int outh = out.h;
    int inch = in.c;
    int outch = out.c;
    int outsz = out.w * out.h;

    int totalsize = 0;

    // pad to xn+2, winograd F(x,3)
    int x = (strcmp(type, "winograd63") == 0) ? 6 : ((strcmp(type, "winograd43") == 0) ? 4 : 2);
    int w_tiles = (outw + x - 1) / x;
    int h_tiles = (outh + x - 1) / x;
    int tiles = w_tiles * h_tiles;

    // how to search
    for (int ch = (outch + 1) / 2; ch >= 8; ch -= ch / 2)
    {
        // align to 8
        ch = (ch + 7) / 8 * 8;
        // const int M = outch;
        const int M = ch;
        const int N = tiles;
        const int K = inch;
        const int B = (x + 2) * (x + 2);

        // NCNN_LOGE("conv3x3s1_winograd63 %d %d %d", M, N, K);

        int TILE_M, TILE_N, TILE_K, nn_M, nn_N, nn_K;
        ncnn::conv3x3s1_winograd_get_optimal_tile_mnk(M, 0, K, B, TILE_M, TILE_N, TILE_K, nT);
        nn_M = (M + TILE_M - 1) / TILE_M;
        flexnn::DummyMat AT(TILE_K * TILE_M, B, (K + TILE_K - 1) / TILE_K, (M + TILE_M - 1) / TILE_M);

        ncnn::conv3x3s1_winograd_get_optimal_tile_mnk(M, N, K, B, TILE_M, TILE_N, TILE_K, nT);
        nn_M = (M + TILE_M - 1) / TILE_M;
        nn_N = (N + TILE_N - 1) / TILE_N;
        nn_K = (K + TILE_K - 1) / TILE_K;

        // NCNN_LOGE("TILE M/N/K = %d %d %d -> %d %d %d", M, N, K, TILE_M, TILE_N, TILE_K);

        flexnn::DummyMat BT(TILE_K * TILE_N, B, (K + TILE_K - 1) / TILE_K, (N + TILE_N - 1) / TILE_N);
        const int nn_NK = nn_N * nn_K;

        flexnn::DummyMat top_tileX(TILE_N * B * TILE_M, 1, nT);
        totalsize = in.total() + AT.total() + BT.total() + top_tileX.total() + out.total();
        if (totalsize < max_size)
            return ch;
    }
    return -1;
}

int FlexnnSlice::get_size_convolution(int layer_index, const char* type, int nT) const
{
    const ncnn::Layer* layer = layers[layer_index];
    if (layer->type != "Convolution")
    {
        fprintf(stderr, "Error: layer %d %s is not convolution\n", layer_index, layers[layer_index]->name.c_str());
        return -1;
    }

    int top_blob_index = layer->tops[0];
    int bottom_blob_index = layer->bottoms[0];

    const ncnn::Convolution* convolution = (const ncnn::Convolution*)layer;

    const flexnn::DummyMat in = blobs[bottom_blob_index].dummy_shape;
    const flexnn::DummyMat out = blobs[top_blob_index].dummy_shape;

    int kernel_size = convolution->kernel_w * convolution->kernel_h;
    int maxk = kernel_size;

    int outw = out.w;
    int outh = out.h;
    int inch = in.c;
    int outch = out.c;
    int outsz = out.w * out.h;

    int totalsize = 0;

    // padding size
    flexnn::DummyMat pad;
    convolution->make_padding(in, pad);

    if (in.total() != pad.total())
    {
        totalsize += pad.total();
    }

    if (strcmp(type, "im2col_gemm") == 0)
    {
        // todo
        return 0;

        const int M = outch;
        const int N = outsz;
        const int K = inch * maxk;

        int TILE_M, TILE_N, TILE_K;
        ncnn::convolution_im2col_gemm_get_minimal_tile_mnk(M, 0, K, TILE_M, TILE_N, TILE_K, nT);
        flexnn::DummyMat AT(TILE_K * TILE_M, (K + TILE_K - 1) / TILE_K, (M + TILE_M - 1) / TILE_M);

        ncnn::convolution_im2col_gemm_get_minimal_tile_mnk(M, N, K, TILE_M, TILE_N, TILE_K, nT);
        flexnn::DummyMat BT(TILE_K * TILE_N, (K + TILE_K - 1) / TILE_K, 1);

        flexnn::DummyMat topT_tileX;
        if (K > TILE_K)
            topT_tileX.create(TILE_N * TILE_M, 1, nT);

        totalsize += in.total() + AT.total() + BT.total() + topT_tileX.total() + out.total();
        // print size one by one
        // fprintf(stderr, "in: %ld, AT: %ld, BT: %ld, topT_tileX: %ld, out: %ld\n", in.total(), AT.total(), BT.total(), topT_tileX.total(), out.total());
        return totalsize;
    }
    else if (strcmp(type, "winograd63") == 0 || strcmp(type, "winograd43") == 0 || strcmp(type, "winograd23") == 0)
    {
        // pad to xn+2, winograd F(x,3)
        int x = (strcmp(type, "winograd63") == 0) ? 6 : ((strcmp(type, "winograd43") == 0) ? 4 : 2);
        int w_tiles = (outw + x - 1) / x;
        int h_tiles = (outh + x - 1) / x;
        int tiles = w_tiles * h_tiles;

        const int M = outch;
        const int N = tiles;
        const int K = inch;
        const int B = (x + 2) * (x + 2);

        // NCNN_LOGE("conv3x3s1_winograd63 %d %d %d", M, N, K);

        int TILE_M, TILE_N, TILE_K, nn_M, nn_N, nn_K;
        ncnn::conv3x3s1_winograd_get_optimal_tile_mnk(M, 0, K, B, TILE_M, TILE_N, TILE_K, nT);
        nn_M = (M + TILE_M - 1) / TILE_M;
        flexnn::DummyMat AT(TILE_K * TILE_M, B, (K + TILE_K - 1) / TILE_K, (M + TILE_M - 1) / TILE_M);

        ncnn::conv3x3s1_winograd_get_optimal_tile_mnk(M, N, K, B, TILE_M, TILE_N, TILE_K, nT);
        nn_M = (M + TILE_M - 1) / TILE_M;
        nn_N = (N + TILE_N - 1) / TILE_N;
        nn_K = (K + TILE_K - 1) / TILE_K;

        // NCNN_LOGE("TILE M/N/K = %d %d %d -> %d %d %d", M, N, K, TILE_M, TILE_N, TILE_K);

        flexnn::DummyMat BT(TILE_K * TILE_N, B, (K + TILE_K - 1) / TILE_K, (N + TILE_N - 1) / TILE_N);
        const int nn_NK = nn_N * nn_K;

        flexnn::DummyMat top_tileX(TILE_N * B * TILE_M, 1, nT);
        totalsize += in.total() + AT.total() + BT.total() + top_tileX.total() + out.total();
        // totalsize = INT32_MAX; // for microbench only. disable winograd
        return totalsize;
    }

    return -1;
}

// 根据新的网络结构对blob和layer进行排序
int FlexnnSlice::topological_sort_new()
{
    std::map<int, int> new_slice_type_recorder;

    // Kahn's algorithm, o(|V|+|E|), assume acyclic for simplicity
    fprintf(stderr, "topological_sort start\n");

    std::vector<ncnn::Layer*> layers_sorted;
    std::vector<ncnn::Blob> blobs_sorted;

    const size_t layer_count = layers.size();
    const size_t blob_count = blobs.size();

    // FILO for layers to visit: input degree is 0
    std::stack<size_t> layers_to_visit;
    // keep the order of these layers (starting layers, no input layer)
    std::queue<size_t> skip_layers;
    std::vector<size_t> layers_order;
    std::vector<bool> is_layer_visited;
    layers_order.resize(layer_count);
    int current_count = 0;
    is_layer_visited.resize(layer_count);

    // find the input layers to start, make sure Input is still layer 0 after sort!!!
    for (size_t i = 0; i < layer_count; i++)
    {
        if (layers[i]->type == "Input")
        {
            skip_layers.push(i);
        }

        /*if (layers[i]->type == "Input" || layers[i]->type == "MemoryData")
        {
            skip_layers.push(i);
        }*/
    }

    while (!skip_layers.empty() || !layers_to_visit.empty())
    {
        size_t layer_index = 0;
        // firstly visit the skip layers
        if (!skip_layers.empty())
        {
            layer_index = skip_layers.front();
            skip_layers.pop();
            layers_order[current_count++] = layer_index;
            is_layer_visited[layer_index] = true;
        }
        else
        {
            layer_index = layers_to_visit.top();
            layers_to_visit.pop();
            layers_order[current_count++] = layer_index;
            is_layer_visited[layer_index] = true;
        }
        std::set<size_t> local_pushed_layers;

        std::vector<int> sub_blobs = layers[layer_index]->tops;
        for (int i = sub_blobs.size() - 1; i >= 0; i--)
        {
            int top_blob_index = sub_blobs[i];
            int next_index = blobs[top_blob_index].consumer;

            if (next_index == -1)
                continue;
            if (next_index < -1 || next_index >= layer_count)
            {
                fprintf(stderr, "invalid consumer index %d.", next_index);
                return -1;
            }

            bool is_to_visit = true;

            std::vector<int> parent_blobs = layers[next_index]->bottoms;
            for (size_t j = 0; j < parent_blobs.size(); j++)
            {
                if (!is_layer_visited[blobs[parent_blobs[j]].producer])
                {   
                    int producer_id = blobs[parent_blobs[j]].producer;
                    if (layers[producer_id]->type == "MemoryData") {
                        layers_order[current_count++] = producer_id;
                        is_layer_visited[producer_id] = true;
                    }
                    else {
                        is_to_visit = false;
                        break;
                    }
                    /*is_to_visit = false;
                    break;*/
                }
            }
            // check local_pushed_layers to avoid layer1->blob1, blob2; blob1->layer2; blob2->layer2.
            if (is_to_visit && local_pushed_layers.count(next_index) == 0)
            {
                layers_to_visit.push(next_index);
                local_pushed_layers.insert(next_index);
            }
        }
    }

    // sort layers
    for (size_t i = 0; i < layer_count; i++)
    {
        size_t layer_index = layers_order[i];
        ncnn::Layer* layer = layers[layer_index];

        // insert layer
        layers_sorted.push_back(layer);

        if (slice_type_recorder.count(layer_index) != 0) {
            int algorithm_type = slice_type_recorder.find(layer_index)->second;
            new_slice_type_recorder.insert({i,algorithm_type});
        }

        // change blob producer & consumer info
        for (size_t j = 0; j < layer->tops.size(); j++)
        {
            int top_blob_index = layer->tops[j];
            ncnn::Blob& top_blob = blobs[top_blob_index];

            top_blob.producer = i;
        }
        for (size_t j = 0; j < layer->bottoms.size(); j++)
        {
            int bottom_blob_index = layer->bottoms[j];
            ncnn::Blob& bottom_blob = blobs[bottom_blob_index];

            bottom_blob.consumer = i;
        }
    }

    // sort blobs in their producers' order
    for (size_t i = 0; i < layer_count; i++)
    {
        size_t layer_index = layers_order[i];     // layer index of original vector
        ncnn::Layer* layer = layers[layer_index]; // layer pointer

        for (size_t j = 0; j < layer->tops.size(); j++)
        {
            int blob_index = layer->tops[j];
            ncnn::Blob& blob = blobs[blob_index];

            int blob_sorted_index = (int)blobs_sorted.size(); // get the new blob index (not pushed yet)
            if (blob.name != "in0" && blob.name != "out0")
            {
                blob.name = std::to_string(blob_sorted_index); // rename with index except in & out
            }
            blobs_sorted.push_back(blob);

            layer->tops[j] = blob_sorted_index; // update tops index

            // find consumer layer
            if (blob.consumer == -1)
                continue;
            if (blob.consumer < -1 || blob.consumer >= layer_count)
            {
                fprintf(stderr, "invalid consumer index %d.", blob.consumer);
                return -1;
            }
            std::vector<int>& bottoms = layers_sorted[blob.consumer]->bottoms; // reference
            auto it = std::find(bottoms.begin(), bottoms.end(), blob_index);
            if (it != bottoms.end())
            {
                *it = blob_sorted_index; // update bottoms index
                std::sort(bottoms.begin(),bottoms.end());
            }
            else
            {
                fprintf(stderr, "consumer layer's bottoms don't include blob %d.", blob_sorted_index);
                return -1;
            }
        }
    }

    // replace layers and blobs vectors
    layers = layers_sorted;
    blobs = blobs_sorted;
    slice_type_recorder = new_slice_type_recorder;

    return 0;
}

// 根据新的网络结构对blob和layer进行排序
int FlexnnSlice::topological_sort()
{
    // Kahn's algorithm, o(|V|+|E|), assume acyclic for simplicity
    fprintf(stderr, "topological_sort\n");

    std::vector<ncnn::Layer*> layers_sorted;
    std::vector<ncnn::Blob> blobs_sorted;

    const size_t layer_count = layers.size();
    const size_t blob_count = blobs.size();

    // FILO for layers to visit: input degree is 0
    std::stack<size_t> layers_to_visit;
    // keep the order of these layers (starting layers, no input layer)
    std::queue<size_t> skip_layers;
    std::vector<size_t> layers_order;
    std::vector<bool> is_layer_visited;
    layers_order.resize(layer_count);
    int current_count = 0;
    is_layer_visited.resize(layer_count);

    // find the input layers to start, make sure Input is still layer 0 after sort!!!
    for (size_t i = 0; i < layer_count; i++)
    {
        if (layers[i]->type == "Input" || layers[i]->bottoms.size() == 0)
        {
            skip_layers.push(i);
            break;
        }
    }

    while (!skip_layers.empty() || !layers_to_visit.empty())
    {
        size_t layer_index = 0;
        // firstly visit the skip layers
        if (!skip_layers.empty())
        {
            layer_index = skip_layers.front();
            skip_layers.pop();
            layers_order[current_count++] = layer_index;
            is_layer_visited[layer_index] = true;
        }
        else
        {
            layer_index = layers_to_visit.top();
            layers_to_visit.pop();
            layers_order[current_count++] = layer_index;
            is_layer_visited[layer_index] = true;
        }
        std::set<size_t> local_pushed_layers;

        std::vector<int> sub_blobs = layers[layer_index]->tops;
        for (int i = sub_blobs.size() - 1; i >= 0; i--)
        {
            int top_blob_index = sub_blobs[i];
            int next_index = blobs[top_blob_index].consumer;

            if (next_index == -1)
                continue; 
            if (next_index < -1 || next_index >= layer_count)
            {
                fprintf(stderr, "invalid consumer index %d.", next_index);
                return -1;
            }

            bool is_to_visit = true;

            std::vector<int> parent_blobs = layers[next_index]->bottoms;
            for (size_t j = 0; j < parent_blobs.size(); j++)
            {
                if (!is_layer_visited[blobs[parent_blobs[j]].producer])
                {
                    is_to_visit = false;
                    break;
                }
            }
            // check local_pushed_layers to avoid layer1->blob1, blob2; blob1->layer2; blob2->layer2.
            if (is_to_visit && local_pushed_layers.count(next_index) == 0)
            {
                layers_to_visit.push(next_index);
                local_pushed_layers.insert(next_index);
            }
        }
    }

    // sort layers
    for (size_t i = 0; i < layer_count; i++)
    {
        size_t layer_index = layers_order[i];
        ncnn::Layer* layer = layers[layer_index];

        // insert layer
        layers_sorted.push_back(layer);

        // change blob producer & consumer info
        for (size_t j = 0; j < layer->tops.size(); j++)
        {
            int top_blob_index = layer->tops[j];
            ncnn::Blob& top_blob = blobs[top_blob_index];

            top_blob.producer = i;
        }
        for (size_t j = 0; j < layer->bottoms.size(); j++)
        {
            int bottom_blob_index = layer->bottoms[j];
            ncnn::Blob& bottom_blob = blobs[bottom_blob_index];

            bottom_blob.consumer = i;
        }
    }

    // sort blobs in their producers' order
    for (size_t i = 0; i < layer_count; i++)
    {
        size_t layer_index = layers_order[i];     // layer index of original vector
        ncnn::Layer* layer = layers[layer_index]; // layer pointer

        for (size_t j = 0; j < layer->tops.size(); j++)
        {
            int blob_index = layer->tops[j];
            ncnn::Blob& blob = blobs[blob_index];

            int blob_sorted_index = (int)blobs_sorted.size(); // get the new blob index (not pushed yet)
            if (blob.name != "in0" && blob.name != "out0")
            {
                blob.name = std::to_string(blob_sorted_index); // rename with index except in & out
            }
            blobs_sorted.push_back(blob);

            layer->tops[j] = blob_sorted_index; // update tops index

            // find consumer layer
            if (blob.consumer == -1)
                continue;
            if (blob.consumer < -1 || blob.consumer >= layer_count)
            {
                fprintf(stderr, "invalid consumer index %d.", blob.consumer);
                return -1;
            }
            std::vector<int>& bottoms = layers_sorted[blob.consumer]->bottoms; // reference
            auto it = std::find(bottoms.begin(), bottoms.end(), blob_index);
            if (it != bottoms.end())
            {
                *it = blob_sorted_index; // update bottoms index
                std::sort(bottoms.begin(), bottoms.end());
            }
            else
            {
                fprintf(stderr, "consumer layer's bottoms don't include blob %d.", blob_sorted_index);
                return -1;
            }
        }
    }

    // replace layers and blobs vectors
    layers = layers_sorted;
    blobs = blobs_sorted;

    return 0;
}

//模型形状推理
int FlexnnSlice::shape_inference()
{   
    fprintf(stderr, "shape_inference preparing\n");

    if (has_custom_layer)
    {
        fprintf(stderr, "model has custom layer, shape_inference skipped\n");
        return -1;
    }

    const size_t layer_count = layers.size();
    const size_t blob_count = blobs.size();

    ncnn::Extractor ex = create_extractor();

    // doesn't need lightmode at all since we use dummymat
    // 启用loght_mode时会自动释放中间层空间
    // 在forward_layer函数中，当lightmode=true时会立即调用Mat::release()清理中间结果
    ex.set_light_mode(false);

    // prepare Input blobs
    for (size_t i = 0; i < layer_count; i++)
    {
        const ncnn::Layer* layer = layers[i];
        if (layer->type == "ncnnfused")
            continue;

        if (layer->type != "Input" && layer->type != "MemoryData")
            continue;
        
        int w, h, c;

        if (layer->type == "MemoryData") {
            ncnn::MemoryData* memo = (ncnn::MemoryData*)layer;

            w = memo->w;
            h = memo->h;
            c = memo->c;
        }
        else {
            ncnn::Input* input = (ncnn::Input*)layer;

            w = input->w;
            h = input->h;
            c = input->c;
        }

        int dims = 0;
        if (w == 0 && h == 0 && c == 0) dims = 0;
        if (w != 0 && h == 0 && c == 0) dims = 1;
        if (w != 0 && h != 0 && c == 0) dims = 2;
        if (w != 0 && h != 0 && c != 0) dims = 3;

        if (dims == 0)
        {
            fprintf(stderr, "Input layer %s without shape info, shape_inference skipped\n", layer->name.c_str());
            return -1;
        }

        //DummyMat只包含维度信息，不持有实际内存
        flexnn::DummyMat m;
        if (dims == 1) m.create(w);
        if (dims == 2) m.create(w, h);
        if (dims == 3) m.create(w, h, c);

        ex.input(layer->tops[0], m);
    }

    fprintf(stderr, "shape_inference starting\n");

    // resolve all layer output blob shape
    for (size_t i = 0; i < layer_count; i++)
    {
        const ncnn::Layer* layer = layers[i];
        if (layer->type == "ncnnfused")
            continue;

        //使用extract方法正向推理该层各个输出blob的形状
        for (size_t j = 0; j < layer->tops.size(); j++)
        {
            int top_blob_index = layer->tops[j];

            flexnn::DummyMat m;
            ex.extract(top_blob_index, m);

            if (m.dims == 0) {
                for (int i = 0 ; i < 100; i++) {
                    printf("WRONG SHAPE FOR LAYER:%s\n", layer->name.c_str());
                    return -1;
                }
            }

            blobs[top_blob_index].dummy_shape = m;
        }
    }

    // 为每个layer的dummy_shapes属性赋值
    for (size_t i = 0; i < layer_count; i++)
    {
        ncnn::Layer* layer = layers[i];
        if (layer->type == "ncnnfused")
            continue;

        layer->bottom_dummy_shapes.resize(layer->bottoms.size());
        for (size_t j = 0; j < layer->bottoms.size(); j++)
        {
            int bottom_blob_index = layer->bottoms[j];

            layer->bottom_dummy_shapes[j] = blobs[bottom_blob_index].dummy_shape;
        }

        layer->top_dummy_shapes.resize(layer->tops.size());
        for (size_t j = 0; j < layer->tops.size(); j++)
        {
            int top_blob_index = layer->tops[j];

            layer->top_dummy_shapes[j] = blobs[top_blob_index].dummy_shape;

        }
    }

    return 0;
}

//全连接层切分 weights slicing
int FlexnnSlice::slice_innerproduct(int max_data_size)
{
    const size_t layer_count = layers.size();

    fprintf(stderr, "slice_innerproduct starting\n");

    for (size_t i = 0; i < layer_count; i++)
    {   
        //只处理全连接层
        if (layers[i]->type != "InnerProduct")
            continue;

        fprintf(stderr, "try to slice innerproduct : layer No.%d\n",i);

        // decide slice size
        ncnn::InnerProduct* innerproduct = (ncnn::InnerProduct*)layers[i];
        int outsz = innerproduct->num_output;
        int insz = innerproduct->weight_data_size / outsz; //weight_data_size = insz * amount_of_neutron  (num_of_neutron = outsz)
        
        //按神经元个数分层 max_size = num_of_neutron
        // max_data_size = max_size * (1 + insz) + insz  activations = insz  weight = max_size * insz + max_size
        int max_size = (max_data_size - insz) / (1 + insz); 

        //第 i 层fc layer按最大输出（神经元个数）为max_size切分
        fprintf(stderr, "try to slice innerproduct by channel %d\n", max_size);

        int ret = slice_innerproduct_outsz(i, max_size);
        if (ret)
        {
            fprintf(stderr, "layer %ld %s slice innerproduct failed.", i, layers[i]->name.c_str());
            return -1;
        }
    }
    return 0;
}

//新卷积层切分
int FlexnnSlice::slice_convolution_new(int max_data_size,char* mode)
{
    fprintf(stderr, "slice_convolution starting\n");
    const size_t layer_count = layers.size();
    for (size_t i = 0; i < layer_count; i++)
    {
        if (layers[i]->type != "Convolution")
            continue;

        // decide slice size
        ncnn::Convolution* convolution = (ncnn::Convolution*)layers[i];

        fprintf(stderr, "try to slice convolution : layer No.%d\n",i);

        int kernel_w = convolution->kernel_w;
        int kernel_h = convolution->kernel_h;
        int dilation_w = convolution->dilation_w;
        int dilation_h = convolution->dilation_h;
        int stride_w = convolution->stride_w;
        int stride_h = convolution->stride_h;

        // 卷积是 one_blob_only 算子,单输入单输出
        int top_blob_index = layers[i]->tops[0];
        int bottom_blob_index = layers[i]->bottoms[0];

        const flexnn::DummyMat in = blobs[bottom_blob_index].dummy_shape;
        const flexnn::DummyMat out = blobs[top_blob_index].dummy_shape;

        // if use winograd: slice by outch
        //仅支持3x3s1且channel数够大时考虑Winograd算法
        if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1 && ( in.c >= 8 || out.c >= 8 ))
        {
            int require_size = -1;
            bool success = true;

            int winograd63_size = -1, winograd43_size = -1;
            if (in.c > 128 || out.c > 128)
            {
                //分析使用winograd43产生的内存需求
                winograd43_size = get_size_convolution(i, "winograd43");
                if (winograd43_size >= max_data_size)
                {
                    //要在max_data_size下使用winograd43算法则一个分片最大包含多少卷积核
                    int max_ch = get_slice_outch_convolution(i, "winograd43", max_data_size);
                    //channel小于8无法发挥winograd优势
                    if (max_ch < 8) continue;
                    // 8 <= max_ch <= 128 ?
                    if (max_ch >= 128)
                    {
                        int origin_layer_num = layers.size();
                        //按卷积核分片
                        fprintf(stderr, "layer %ld %s is trying to slice by winograd43.\n", i, layers[i]->name.c_str());
                        int ret = slice_convolution_outch(i, max_ch);
                        if (ret)
                        {
                            fprintf(stderr, "layer %ld %s slice convolution failed.\n", i, layers[i]->name.c_str());
                            return -1;
                        }
                        else
                        {
                            fprintf(stderr, "this conv will count by winograd43\n");

                            int num_slice = out.c / max_ch;
                            int remain_size = out.c % max_ch;
                            if (remain_size)
                            {
                                num_slice++;
                            }

                            for (int p = 0; p < num_slice; p++)
                            {
                                slice_type_recorder.insert({origin_layer_num + p, 2});
                            }
                        }
                    }
                }
                fprintf(stderr, "this conv will count by winograd43 wholly\n");
                slice_type_recorder.insert({i, 2});
                continue;
            }
            // 当计算内存需求适中时优先考虑winograd63，若使用winograd63产生的内存需求大于max_data_size则按卷积核(out.channel)分块
            // 分析使用winograd63产生的内存需求
            // winograd63
            winograd63_size = get_size_convolution(i, "winograd63");
            if (winograd63_size >= max_data_size)
            {
                //要在max_data_size下使用winograd63算法则一个分片最大包含多少卷积核
                int max_ch = get_slice_outch_convolution(i, "winograd63", max_data_size);
                if (max_ch < 8) continue;
                int origin_layer_num = layers.size();
                //按卷积核分片
                fprintf(stderr, "layer %ld %s is trying to slice by winograd63.\n", i, layers[i]->name.c_str());
                int ret = slice_convolution_outch(i, max_ch);
                if (ret)
                {
                    fprintf(stderr, "layer %ld %s slice convolution failed.\n", i, layers[i]->name.c_str());
                    return -1;
                }
                else
                {
                    fprintf(stderr, "this conv will count by winograd63\n");

                    int num_slice = out.c / max_ch;
                    int remain_size = out.c % max_ch;
                    if (remain_size)
                    {
                        num_slice++;
                    }

                    for (int p = 0; p < num_slice; p++)
                    {
                        slice_type_recorder.insert({origin_layer_num + p, 3});
                    }
                }
            }
            fprintf(stderr, "this conv will count by winograd63\n");
            slice_type_recorder.insert({i, 3});
            continue;
        }
        else {
            fprintf(stderr, "this conv will count by old mean\n");
            slice_type_recorder.insert({i, 0});
        }
    }

    return 0;
}

//卷积层切分
int FlexnnSlice::slice_convolution(int max_data_size)
{
    const size_t layer_count = layers.size();
    for (size_t i = 0; i < layer_count; i++)
    {
        if (layers[i]->type != "Convolution")
            continue;

        // decide slice size
        ncnn::Convolution* convolution = (ncnn::Convolution*)layers[i];

        int kernel_w = convolution->kernel_w;
        int kernel_h = convolution->kernel_h;
        int dilation_w = convolution->dilation_w;
        int dilation_h = convolution->dilation_h;
        int stride_w = convolution->stride_w;
        int stride_h = convolution->stride_h;

        // one_blob_only 单输入单输出
        int top_blob_index = layers[i]->tops[0];
        int bottom_blob_index = layers[i]->bottoms[0];

        const flexnn::DummyMat in = blobs[bottom_blob_index].dummy_shape;
        const flexnn::DummyMat out = blobs[top_blob_index].dummy_shape;
        
        // if use winograd: slice by outch
        //仅支持3x3s1且channel数够大时考虑Winograd算法
        if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1 && in.c >= 8 && out.c >= 8)
        {
            int winograd63_size = -1, winograd43_size = -1;
            // winograd43
            if (in.c > 128 || out.c > 128)
            {   
                //分析使用winograd43产生的内存需求
                winograd43_size = get_size_convolution(i, "winograd43");
                if (winograd43_size >= max_data_size)
                {   
                    //要在max_data_size下使用winograd43算法则一个分片最大包含多少卷积核
                    int max_ch = get_slice_outch_convolution(i, "winograd43", max_data_size);
                    //channel小于8无法发挥winograd优势
                    if (max_ch < 8) continue;
                    // 8 <= max_ch <= 128 ?
                    if (max_ch > 128)
                    {
                        int origin_layer_num = layers.size();
                        //按卷积核分片
                        int ret = slice_convolution_outch(i, max_ch);
                        if (ret)
                        {
                            fprintf(stderr, "layer %ld %s slice convolution failed.", i, layers[i]->name.c_str());
                            return -1;
                        }
                        else
                        {
                            fprintf(stderr, "this conv will count by winograd43\n");

                            int num_slice = out.c / max_ch;
                            int remain_size = out.c % max_ch;
                            if (remain_size)
                            {
                                num_slice++;
                            }

                            for (int p = 0; p < num_slice; p++)
                            {
                                slice_type_recorder.insert({origin_layer_num + p, 2});
                            }
                        }
                    }
                }
                continue;
            }
            // 当计算内存需求适中时优先考虑winograd63，若使用winograd63产生的内存需求大于max_data_size则按卷积核(out.channel)分块
            // 分析使用winograd63产生的内存需求
            // winograd63
            winograd63_size = get_size_convolution(i, "winograd63");
            if (winograd63_size >= max_data_size)
            {   
                //要在max_data_size下使用winograd63算法则一个分片最大包含多少卷积核
                int max_ch = get_slice_outch_convolution(i, "winograd63", max_data_size);
                if (max_ch < 8) continue;
                //按卷积核分片
                int origin_layer_num = layers.size();
                int ret = slice_convolution_outch(i, max_ch);
                if (ret)
                {
                    fprintf(stderr, "layer %ld %s slice convolution failed.", i, layers[i]->name.c_str());
                    return -1;
                }
                else
                {
                    fprintf(stderr, "this conv will count by winograd63\n");

                    int num_slice = out.c / max_ch;
                    int remain_size = out.c % max_ch;
                    if (remain_size)
                    {
                        num_slice++;
                    }

                    for (int p = 0; p < num_slice; p++)
                    {
                        slice_type_recorder.insert({origin_layer_num + p, 3});
                    }
                }
            }
            continue;
        }
    }

    // if 3x3s2?

    return 0;
}

int FlexnnSlice::slice_innerproduct_outsz(int layer_index, int max_size)
{
    const size_t layer_count = layers.size();
    const size_t blob_count = blobs.size();

    if (layers[layer_index]->type != "InnerProduct")
    {
        fprintf(stderr, "Error: layer %d %s is not innerproduct\n", layer_index, layers[layer_index]->name.c_str());
        return -1;
    }

    // one_blob_only 单输入单输出
    int top_blob_index = layers[layer_index]->tops[0];
    int bottom_blob_index = layers[layer_index]->bottoms[0];

    ncnn::InnerProduct* innerproduct = (ncnn::InnerProduct*)layers[layer_index];
    int outsz = innerproduct->num_output;

    // no need to slice
    if (outsz <= max_size)
    {
        return 0;
    }

    //  InnerProduct -> ...
    //  Split        -> ... -> Innerproducts -> Concat
    //  使用Split层拆分原有FC层

    // get number of slice
    int num_slice = outsz / max_size;
    int remain_size = outsz % max_size;
    if (remain_size)
    {
        num_slice++;
    }

    // new blobs
    std::vector<ncnn::Blob> slice_bottom_blobs, slice_top_blobs;
    slice_bottom_blobs.resize(num_slice);
    slice_top_blobs.resize(num_slice);
    for (size_t i = 0; i < num_slice; i++)
    {
        slice_bottom_blobs[i].producer = layer_index;     // split
        slice_bottom_blobs[i].consumer = layer_count + i; // ip
        slice_bottom_blobs[i].name = innerproduct->name + "_slice_" + std::to_string(i) + "_bottom";
        slice_top_blobs[i].producer = layer_count + i;         // ip
        slice_top_blobs[i].consumer = layer_count + num_slice; // concat
        slice_top_blobs[i].name = innerproduct->name + "_slice_" + std::to_string(i) + "_top";
    }
    blobs.insert(blobs.end(), slice_bottom_blobs.begin(), slice_bottom_blobs.end());
    blobs.insert(blobs.end(), slice_top_blobs.begin(), slice_top_blobs.end());

    // split
    ncnn::Split* split = (ncnn::Split*)ncnn::create_layer("Split");
    split->type = "Split";
    split->name = innerproduct->name + "_split";
    split->bottoms = innerproduct->bottoms;
    split->tops.resize(num_slice);
    for (size_t i = 0; i < num_slice; i++)
    {
        split->tops[i] = blob_count + i;
    }

    // innerproduct 创建子FC层
    std::vector<ncnn::InnerProduct*> innerproducts;
    innerproducts.resize(num_slice);
    for (size_t i = 0; i < innerproducts.size(); i++)
    {
        innerproducts[i] = (ncnn::InnerProduct*)ncnn::create_layer("InnerProduct");
        innerproducts[i]->type = "InnerProduct";
        innerproducts[i]->name = innerproduct->name + "_slice_" + std::to_string(i);
        // one_blob_only 单输入单输出
        innerproducts[i]->bottoms.resize(1, blob_count + i);
        innerproducts[i]->tops.resize(1, blob_count + num_slice + i);
    }

    // concat 收集FC子层输出
    ncnn::Concat* concat = (ncnn::Concat*)ncnn::create_layer("Concat");
    concat->type = "Concat";
    concat->name = innerproduct->name + "_concat";
    concat->tops = innerproduct->tops;
    concat->bottoms.resize(num_slice);
    for (size_t i = 0; i < num_slice; i++)
    {
        concat->bottoms[i] = blob_count + num_slice + i;
    }
    // gpt2
    if (blobs[bottom_blob_index].dummy_shape.dims == 1)
    {
        concat->axis = 0;
    }
    else
    {
        concat->axis = 1;
    }

    // 为FC子层分配参数及权重
    ncnn::ParamDict pd;
    for (size_t i = 0; i < innerproducts.size(); i++)
    {
        // params
        innerproducts[i]->load_param(pd);
        int size = max_size;
        if (i == innerproducts.size() - 1 && remain_size > 0)
        {
            size = remain_size;
        }

        innerproducts[i]->num_output = size;
        innerproducts[i]->bias_term = innerproduct->bias_term;
        innerproducts[i]->weight_data_size = innerproduct->weight_data_size / outsz * size;
        innerproducts[i]->int8_scale_term = innerproduct->int8_scale_term;
        innerproducts[i]->activation_type = innerproduct->activation_type;
        innerproducts[i]->activation_params = innerproduct->activation_params;

        // weights
        innerproducts[i]->weight_data = innerproduct->weight_data.range(innerproduct->weight_data_size / outsz * max_size * i, innerproduct->weight_data_size / outsz * size).clone();
        innerproducts[i]->bias_data = innerproduct->bias_data.range(max_size * i, size).clone();
    }

    // insert layers
    layers[layer_index] = split; //替换
    layers.insert(layers.end(), innerproducts.begin(), innerproducts.end());
    layers.push_back(concat);
    delete innerproduct;

    return 0;
}

int FlexnnSlice::slice_convolution_outch(int layer_index, int max_size)
{
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

    if (out.c <= max_size)
    {
        return 0;
    }

    //  Convolution -> ...
    //  Split       -> ... -> Convolutions -> Concat
    //  (replace)               (append at the end)

    // get number of slice
    int num_slice = out.c / max_size;
    int remain_size = out.c % max_size;
    if (remain_size)
    {
        num_slice++;
    }

    // new blobs
    std::vector<ncnn::Blob> slice_bottom_blobs, slice_top_blobs;
    slice_bottom_blobs.resize(num_slice);
    slice_top_blobs.resize(num_slice);
    for (size_t i = 0; i < num_slice; i++)
    {
        slice_bottom_blobs[i].producer = layer_index;     // split
        slice_bottom_blobs[i].consumer = layer_count + i; // Convolutions
        slice_bottom_blobs[i].name = convolution->name + "_slice_" + std::to_string(i) + "_bottom";
        slice_top_blobs[i].producer = layer_count + i;         // Convolutions
        slice_top_blobs[i].consumer = layer_count + num_slice; // concat
        slice_top_blobs[i].name = convolution->name + "_slice_" + std::to_string(i) + "_top";
    }
    blobs.insert(blobs.end(), slice_bottom_blobs.begin(), slice_bottom_blobs.end());
    blobs.insert(blobs.end(), slice_top_blobs.begin(), slice_top_blobs.end());

    // split
    ncnn::Split* split = (ncnn::Split*)ncnn::create_layer("Split");
    split->type = "Split";
    split->name = convolution->name + "_split";
    split->bottoms = convolution->bottoms;
    split->tops.resize(num_slice);
    for (size_t i = 0; i < num_slice; i++)
    {
        split->tops[i] = blob_count + i;
    }

    // convolution子层
    std::vector<ncnn::Convolution*> convolutions;
    convolutions.resize(num_slice);
    for (size_t i = 0; i < convolutions.size(); i++)
    {
        convolutions[i] = (ncnn::Convolution*)ncnn::create_layer("Convolution");
        convolutions[i]->type = "Convolution";
        convolutions[i]->name = convolution->name + "_slice_" + std::to_string(i);
        convolutions[i]->bottoms.resize(1, blob_count + i);
        convolutions[i]->tops.resize(1, blob_count + num_slice + i);
    }

    // concat
    ncnn::Concat* concat = (ncnn::Concat*)ncnn::create_layer("Concat");
    concat->type = "Concat";
    concat->name = convolution->name + "_concat";
    concat->axis = 0; // concat by channels
    concat->tops = convolution->tops;
    concat->bottoms.resize(num_slice);
    for (size_t i = 0; i < num_slice; i++)
    {
        concat->bottoms[i] = blob_count + num_slice + i;
    }

    // assign params and weights
    ncnn::ParamDict pd;
    for (size_t i = 0; i < convolutions.size(); i++)
    {
        // params
        convolutions[i]->load_param(pd);
        int size = max_size;
        if (i == convolutions.size() - 1 && remain_size > 0)
        {
            size = remain_size;
        }

        // shapes相关参数
        convolutions[i]->num_output = size;
        convolutions[i]->kernel_w = convolution->kernel_w;
        convolutions[i]->kernel_h = convolution->kernel_h;
        convolutions[i]->dilation_w = convolution->dilation_w;
        convolutions[i]->dilation_h = convolution->dilation_h;
        convolutions[i]->stride_w = convolution->stride_w;
        convolutions[i]->stride_h = convolution->stride_h;
        convolutions[i]->pad_bottom = convolution->pad_bottom;
        convolutions[i]->pad_left = convolution->pad_left;
        convolutions[i]->pad_right = convolution->pad_right;
        convolutions[i]->pad_top = convolution->pad_top;
        convolutions[i]->pad_value = convolution->pad_value;

        // others
        convolutions[i]->bias_term = convolution->bias_term;
        convolutions[i]->weight_data_size = convolution->weight_data_size / out.c * size;
        convolutions[i]->int8_scale_term = convolution->int8_scale_term;
        convolutions[i]->activation_type = convolution->activation_type;
        convolutions[i]->activation_params = convolution->activation_params;

        // weights
        convolutions[i]->weight_data = convolution->weight_data.range(convolution->weight_data_size / out.c * max_size * i, convolution->weight_data_size / out.c * size).clone();
        convolutions[i]->bias_data = convolution->bias_data.range(max_size * i, size).clone();
    }

    // insert layers
    layers[layer_index] = split;
    layers.insert(layers.end(), convolutions.begin(), convolutions.end());
    layers.push_back(concat);
    delete convolution;

    return 0;
}

int FlexnnSlice::transform_kernel_convolution_new(int max_data_size, int thread_num)
{
    fprintf(stderr, "transform_kernel_convolution start\n");

    const size_t layer_count = layers.size();
    const size_t blob_count = blobs.size();

    for (size_t i = 0; i < layer_count; i++)
    {
        if (layers[i]->type != "Convolution")
            continue;

        // decide slice size
        ncnn::Convolution* convolution = (ncnn::Convolution*)layers[i];
        fprintf(stderr, "deal with kernel of conv : Layer No.%d\n",i);

        int kernel_w = convolution->kernel_w;
        int kernel_h = convolution->kernel_h;
        int dilation_w = convolution->dilation_w;
        int dilation_h = convolution->dilation_h;
        int stride_w = convolution->stride_w;
        int stride_h = convolution->stride_h;

        int top_blob_index = layers[i]->tops[0];
        int bottom_blob_index = layers[i]->bottoms[0];

        const flexnn::DummyMat in = blobs[bottom_blob_index].dummy_shape;
        const flexnn::DummyMat out = blobs[top_blob_index].dummy_shape;

        // 对于3x3s1的卷积层，根据速度优先的顺序对已经经过分片的卷积算子选择实现算法：winograd63->winograd43->im2col_gemm
        if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {   
            int ret = 0;
            bool success = true;
            if (slice_type_recorder.count(i) != 0 && slice_type_recorder.find(i)->second != 0)
            {
                int conv_type = slice_type_recorder.find(i)->second;

                switch (conv_type)
                {
                case 3:
                    fprintf(stderr, "try to transform by winograd63\n");
                    ret = transform_kernel_convolution_winograd63_arm(i, thread_num);
                    fprintf(stderr, "successfully transform by winograd63\n");
                    break;
                case 2:
                    fprintf(stderr, "try to transform by winograd43\n");
                    ret = transform_kernel_convolution_winograd43_arm(i, thread_num);
                    fprintf(stderr, "successfully transform by winograd43\n");
                    break;
                case 1:
                    fprintf(stderr, "try to transform by winograd23\n");
                    ret = transform_kernel_convolution_winograd23_arm(i, thread_num);
                    fprintf(stderr, "successfully transform by winograd23\n");
                    break;
                }
                if (ret) {
                    fprintf(stderr, "layer %ld %s transform kernel convolution failed.\n", i, layers[i]->name.c_str());
                    return -1;
                } else {
                    continue;
                }
            } 
            else
            {
                fprintf(stderr, "try to transform by im2col_gemm\n");
                ret = transform_kernel_convolution_im2col_gemm(i, thread_num);
                fprintf(stderr, "successfully transform by im2col_gemm\n");
                break;
            }
            fprintf(stderr, "try to transform by old mean,first try winograd63\n");

            if (get_size_convolution(i, "winograd63") < max_data_size) //winograd63符合条件
                fprintf(stderr, "try to transform by winograd63\n");
                ret = transform_kernel_convolution_winograd63_arm(i, thread_num);
            if (ret)
            {
                fprintf(stderr, "transform by winograd63 fail\n");
                if (get_size_convolution(i, "winograd43") < max_data_size) //winograd43符合条件
                    fprintf(stderr, "try to transform by winograd43\n");
                    ret = transform_kernel_convolution_winograd43_arm(i, thread_num);
                if (ret)
                {
                    fprintf(stderr, "layer %ld %s transform kernel winograd43 failed, fallback to im2col gemm.\n", i, layers[i]->name.c_str());
                    if (get_size_convolution(i, "im2col_gemm") > max_data_size) //fallback to im2col_gemm
                    {
                        fprintf(stderr, "layer %ld %s im2col size %d > max_data_size %d\n", i, layers[i]->name.c_str(), get_size_convolution(i, "im2col_gemm"), max_data_size);
                        return -1;
                    }
                    ret = transform_kernel_convolution_im2col_gemm(i, thread_num);
                    if (ret)
                    {
                        fprintf(stderr, "layer %ld %s transform kernel convolution failed.\n", i, layers[i]->name.c_str());
                        return -1;
                    }
                    else {
                        fprintf(stderr, "successfully transform by im2col\n");
                    }
                } else {
                    fprintf(stderr, "successfully transform by winograd43\n");
                }
            } else {
                fprintf(stderr, "successfully transform by winograd63\n");
            }
            continue;
        }
        else
        {
            //break;
            //非3x3s1情况
            //获取cpu的L2缓存大小并判断是否需要对卷积层使用gemm
            int l2_cache_size_fp32 = ncnn::get_cpu_level2_cache_size() / sizeof(float);
            bool prefer_sgemm = in.c * out.c * kernel_w * kernel_h * dilation_w * dilation_h * stride_w * stride_h * 2 > l2_cache_size_fp32 || (in.c > 16 || out.c > 16 || (kernel_h == 1 && kernel_w == 1)); //WHY?

            if (prefer_sgemm)
            {
                int ret = -1;

                if (get_size_convolution(i, "im2col_gemm") < max_data_size) //使用im2col_gemm
                    ret = transform_kernel_convolution_im2col_gemm(i, thread_num);
                if (ret)
                {
                    fprintf(stderr, "layer %ld %s transform kernel convolution failed.\n", i, layers[i]->name.c_str());
                    return -1;
                }
                else
                {
                    fprintf(stderr, "successfully transform by im2col_gemm\n");
                }
                continue;
            }

            //一下情况不优化 （无法？）
            if ((kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
                || (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
                || (kernel_w == 4 && kernel_h == 4 && dilation_w == 1 && dilation_h == 1 && stride_w == 4 && stride_h == 4)
                || (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
                || (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
                || (kernel_w == 7 && kernel_h == 7 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
                || (kernel_w == 7 && kernel_h == 7 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2))
            {
                continue;
            }

            //3x3s2情况
            if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
            {
                int ret = transform_kernel_convolution_3x3s2(i); //尝试3x3s2特殊优化
                if (ret)
                {
                    fprintf(stderr, "layer %ld %s transform kernel 3x3s2 failed, fallback to im2col gemm.\n", i, layers[i]->name.c_str());
                    ret = -1;
                    if (get_size_convolution(i, "im2col_gemm") > max_data_size)
                    {
                        fprintf(stderr, "layer %ld %s im2col size %d > max_data_size %d\n", i, layers[i]->name.c_str(), get_size_convolution(i, "im2col_gemm"), max_data_size);
                        return -1;
                    }
                    else
                    {
                        ret = transform_kernel_convolution_im2col_gemm(i, thread_num); //fallback to im2col_gemm
                        if (ret)
                        {
                            fprintf(stderr, "layer %ld %s transform kernel convolution failed.\n", i, layers[i]->name.c_str());
                            return -1;
                        }
                    }
                }
                continue;
            }

            fprintf(stderr, "layer %ld %s no suitable transform.\n", i, layers[i]->name.c_str());
            return -1;
        }
    }
    return 0;
}

int FlexnnSlice::transform_kernel_convolution(int max_data_size, int thread_num)
{
    const size_t layer_count = layers.size();
    const size_t blob_count = blobs.size();

    for (size_t i = 0; i < layer_count; i++)
    {
        if (layers[i]->type != "Convolution")
            continue;

        // decide slice size
        ncnn::Convolution* convolution = (ncnn::Convolution*)layers[i];

        int kernel_w = convolution->kernel_w;
        int kernel_h = convolution->kernel_h;
        int dilation_w = convolution->dilation_w;
        int dilation_h = convolution->dilation_h;
        int stride_w = convolution->stride_w;
        int stride_h = convolution->stride_h;

        int top_blob_index = layers[i]->tops[0];
        int bottom_blob_index = layers[i]->bottoms[0];

        const flexnn::DummyMat in = blobs[bottom_blob_index].dummy_shape;
        const flexnn::DummyMat out = blobs[top_blob_index].dummy_shape;



        // 对于3x3s1的卷积层，根据速度优先的顺序对已经经过分片的卷积算子选择实现算法：winograd63->winograd43->im2col_gemm
        if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            int ret = -1;
            if (get_size_convolution(i, "winograd63") < max_data_size) //winograd63符合条件
                ret = transform_kernel_convolution_winograd63(i, thread_num);
            if (ret)
            {
                if (get_size_convolution(i, "winograd43") < max_data_size) //winograd43符合条件
                    ret = transform_kernel_convolution_winograd43(i, thread_num);
                if (ret)
                {
                    fprintf(stderr, "layer %ld %s transform kernel winograd43 failed, fallback to im2col gemm.\n", i, layers[i]->name.c_str());
                    if (get_size_convolution(i, "im2col_gemm") > max_data_size) //fallback to im2col_gemm
                    {
                        fprintf(stderr, "layer %ld %s im2col size %d > max_data_size %d\n", i, layers[i]->name.c_str(), get_size_convolution(i, "im2col_gemm"), max_data_size);
                        return -1;
                    }
                    ret = transform_kernel_convolution_im2col_gemm(i, thread_num);
                    if (ret)
                    {
                        fprintf(stderr, "layer %ld %s transform kernel convolution failed.\n", i, layers[i]->name.c_str());
                        return -1;
                    }
                }
            }
            continue;
        } else { 
            //非3x3s1情况
            //获取cpu的L2缓存大小并判断是否需要对卷积层使用gemm
            int l2_cache_size_fp32 = ncnn::get_cpu_level2_cache_size() / sizeof(float);
            bool prefer_sgemm = in.c * out.c * kernel_w * kernel_h * dilation_w * dilation_h * stride_w * stride_h * 2 > l2_cache_size_fp32 || (in.c > 16 || out.c > 16 || (kernel_h == 1 && kernel_w == 1)); //WHY?


            if (prefer_sgemm)
            {
                int ret = -1;
                
                if (get_size_convolution(i, "im2col_gemm") < max_data_size) //使用im2col_gemm
                    ret = transform_kernel_convolution_im2col_gemm(i, thread_num);
                if (ret)
                {
                    fprintf(stderr, "layer %ld %s transform kernel convolution failed.\n", i, layers[i]->name.c_str());
                    return -1;
                }
                continue;
            }

            //一下情况不优化 （无法？）
            if ((kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
                || (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
                || (kernel_w == 4 && kernel_h == 4 && dilation_w == 1 && dilation_h == 1 && stride_w == 4 && stride_h == 4)
                || (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
                || (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
                || (kernel_w == 7 && kernel_h == 7 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
                || (kernel_w == 7 && kernel_h == 7 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2))
            {
                continue;
            }

            //3x3s2情况
            if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
            {
                int ret = transform_kernel_convolution_3x3s2(i); //尝试3x3s2特殊优化
                if (ret)
                {
                    fprintf(stderr, "layer %ld %s transform kernel 3x3s2 failed, fallback to im2col gemm.\n", i, layers[i]->name.c_str());
                    ret = -1;
                    if (get_size_convolution(i, "im2col_gemm") > max_data_size) 
                    {
                        fprintf(stderr, "layer %ld %s im2col size %d > max_data_size %d\n", i, layers[i]->name.c_str(), get_size_convolution(i, "im2col_gemm"), max_data_size);
                        return -1;
                    } else {
                        ret = transform_kernel_convolution_im2col_gemm(i, thread_num); //fallback to im2col_gemm
                        if (ret)
                        {
                            fprintf(stderr, "layer %ld %s transform kernel convolution failed.\n", i, layers[i]->name.c_str());
                            return -1;
                        }
                    }
                }
                continue;
            }

            fprintf(stderr, "layer %ld %s no suitable transform.\n", i, layers[i]->name.c_str());
            return -1;
        }
    }
    return 0;
}


//指定层根据im2col_gemm算法预处理
int FlexnnSlice::transform_kernel_convolution_im2col_gemm(int layer_index, int thread_num)
{
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
    opt.num_threads = thread_num;

    //使用ncnn::convolution_im2col_gemm_transform_kernel方法获取im2col_gemm算法对应的卷积权重参数
    ncnn::convolution_im2col_gemm_transform_kernel(convolution->weight_data, weight_sgemm_data, in.c, convolution->num_output, convolution->kernel_w, convolution->kernel_h, opt);

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

//指定层根据winograd63算法预处理
int FlexnnSlice::transform_kernel_convolution_winograd63(int layer_index, int thread_num)
{
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

    const flexnn::DummyMat in = blobs[bottom_blob_index].dummy_shape;
    const flexnn::DummyMat out = blobs[top_blob_index].dummy_shape;

    //channel太少不适用winograd63
    if (in.c < 8 || out.c < 8)
    {
        return -1;
    }

    //channel太多不适用winograd63
    if (in.c > 128 || out.c > 128)
    {
        return -1;
    }

    //使用ncnn::conv3x3s1_winograd63_transform_kernel方法获取winograd63算法对应的卷积权重参数
    ncnn::Mat weight_winograd63_data;
    ncnn::Option opt;
    opt.num_threads = thread_num;
    ncnn::conv3x3s1_winograd63_transform_kernel(convolution->weight_data, weight_winograd63_data, in.c, convolution->num_output, opt);

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

//指定层根据winograd43算法预处理
int FlexnnSlice::transform_kernel_convolution_winograd43(int layer_index, int thread_num)
{
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
        return -1;
    }

    const flexnn::DummyMat in = blobs[bottom_blob_index].dummy_shape;
    const flexnn::DummyMat out = blobs[top_blob_index].dummy_shape;

    //channel太少不适用winograd63
    if (in.c < 8 || out.c < 8)
    {
        return -1;
    }

    //使用ncnn::conv3x3s1_winograd43_transform_kernel方法获取winograd43算法对应的卷积权重参数
    ncnn::Mat weight_winograd43_data;
    ncnn::Option opt;
    opt.num_threads = thread_num;
    ncnn::conv3x3s1_winograd43_transform_kernel(convolution->weight_data, weight_winograd43_data, in.c, convolution->num_output, opt);

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

//指定层根据winograd23算法预处理
int FlexnnSlice::transform_kernel_convolution_winograd23(int layer_index, int thread_num)
{
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
        return -1;
    }

    const flexnn::DummyMat in = blobs[bottom_blob_index].dummy_shape;
    const flexnn::DummyMat out = blobs[top_blob_index].dummy_shape;

    //channel太少不适用winograd23
    if (in.c < 8 || out.c < 8)
    {
        return -1;
    }

    //使用ncnn::conv3x3s1_winograd43_transform_kernel方法获取winograd43算法对应的卷积权重参数
    ncnn::Mat weight_winograd23_data;
    ncnn::Option opt;
    opt.num_threads = thread_num;
    ncnn::conv3x3s1_winograd23_transform_kernel(convolution->weight_data, weight_winograd23_data, in.c, convolution->num_output, opt);

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

//指定层根据winograd63算法预处理
int FlexnnSlice::transform_kernel_convolution_winograd63_arm(int layer_index, int thread_num)
{
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

    const flexnn::DummyMat in = blobs[bottom_blob_index].dummy_shape;
    const flexnn::DummyMat out = blobs[top_blob_index].dummy_shape;

    //channel太少不适用winograd63
    if (in.c < 8 && out.c < 8)
    {
        return -1;
    }

    //channel太多不适用winograd63
    if (in.c > 128 || out.c > 128)
    {
        return -1;
    }

    //使用ncnn::conv3x3s1_winograd63_transform_kernel方法获取winograd63算法对应的卷积权重参数
    ncnn::Mat weight_winograd63_data;
    ncnn::Option opt;
    //opt.num_threads = thread_num;
    ncnn::conv3x3s1_winograd63_transform_kernel(convolution->weight_data, weight_winograd63_data, in.c, convolution->num_output, opt);

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

//指定层根据winograd43算法预处理
int FlexnnSlice::transform_kernel_convolution_winograd43_arm(int layer_index, int thread_num)
{
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
        return -1;
    }

    const flexnn::DummyMat in = blobs[bottom_blob_index].dummy_shape;
    const flexnn::DummyMat out = blobs[top_blob_index].dummy_shape;

    //channel太少不适用winograd63
    if (in.c < 8 && out.c < 8)
    {
        return -1;
    }

    //使用ncnn::conv3x3s1_winograd43_transform_kernel方法获取winograd43算法对应的卷积权重参数
    ncnn::Mat weight_winograd43_data;
    ncnn::Option opt;
    //opt.num_threads = thread_num;
    ncnn::conv3x3s1_winograd43_transform_kernel(convolution->weight_data, weight_winograd43_data, in.c, convolution->num_output, opt);

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

//指定层根据winograd23算法预处理
int FlexnnSlice::transform_kernel_convolution_winograd23_arm(int layer_index, int thread_num)
{
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
        return -1;
    }

    const flexnn::DummyMat in = blobs[bottom_blob_index].dummy_shape;
    const flexnn::DummyMat out = blobs[top_blob_index].dummy_shape;

    //channel太少不适用winograd23
    if (in.c < 8 && out.c < 8)
    {
        return -1;
    }

    //使用ncnn::conv3x3s1_winograd43_transform_kernel方法获取winograd43算法对应的卷积权重参数
    ncnn::Mat weight_winograd23_data;
    ncnn::Option opt;
    //opt.num_threads = thread_num;
    ncnn::conv3x3s1_winograd23_transform_kernel(convolution->weight_data, weight_winograd23_data, in.c, convolution->num_output, opt);

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

int FlexnnSlice::transform_kernel_convolution_3x3s2(int layer_index)
{
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

    if (!(kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2))
    {
        fprintf(stderr, "Error: layer %d %s is not conv3x3s2\n", layer_index, layers[layer_index]->name.c_str());
        return -1;
    }

    const flexnn::DummyMat in = blobs[bottom_blob_index].dummy_shape;
    const flexnn::DummyMat out = blobs[top_blob_index].dummy_shape;

    ncnn::Mat weight_3x3s2_data;

    //使用ncnn::conv3x3s2_transform_kernel_neon方法获取3x3s2卷积优化后对应的卷积权重参数
    ncnn::conv3x3s2_transform_kernel_neon(convolution->weight_data, weight_3x3s2_data, in.c, out.c);

    convolution->weight_data_type = 6;
    convolution->weight_data = weight_3x3s2_data; // replace with transformed one
    if (convolution->weight_data.empty())
    {
        fprintf(stderr, "Error: 3x3s2 layer %d %s weight data is empty\n", layer_index, layers[layer_index]->name.c_str());
        return -1;
    }
    convolution->weight_w = weight_3x3s2_data.w;
    convolution->weight_h = weight_3x3s2_data.h;
    convolution->weight_c = weight_3x3s2_data.c;

    return 0;
}

FlexnnSlice::FlexnnSlice()
    : ModelWriter()
{
}

#endif // FLEXNN_SLICE_H