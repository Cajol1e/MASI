#ifndef FLEXNN_SCHEDULE_H
#define FLEXNN_SCHEDULE_H
#include "stdlib.h"
#include "stdio.h"
//#include "unistd.h"
#include "string.h"
#include "profiler.h"

#include <vector>
#include <queue>
#include <map>
#include <set>
#include <stack>
#include <string>
#include <iostream>
#include <list>
#include <algorithm>
#include <map>
#include <cmath>
#include <iterator>

#include "allocator.h"
#include "xyplane.h"

static char* path = "/home/tecl/results/";

class MemoryProfile
{
public:
    MemoryProfile()
        : start_layer_index(0), end_layer_index(0), size(0), memory_type(0), malloc_count(0), x(0), y(0), name("")
    {
    }

public:
    int start_layer_index; // start of lifetime
    int end_layer_index;   // end of lifetime
    int size;

    int memory_type;  // 0 for weight, 1 for blob, 2 for intermediate
    int malloc_count; // malloc count of this type
    std::string name;

    // schedule
    int x; // malloc time
    int y; // offset

public:
    int memory_index() const // get a uuid based on mem_t and m_cnt //x is not used?
    {
        return ((x & 0x3fff) << 18) | ((memory_type & 0x3) << 16) | (malloc_count & 0xffff); // 14 bits for x, 2 bits for mem_t, 16 bits for m_cnt
    }

    static int memory_index(int x, int mem_t, int m_cnt) // get a uuid based on mem_t and m_cnt
    {
        return ((x & 0x3fff) << 18) | ((mem_t & 0x3) << 30) | (m_cnt & 0xffff); // 14 bits for x, 2 bits for mem_t, 16 bits for m_cnt
    }

public:
    // compare
    static bool compare_layer_index(const MemoryProfile& a, const MemoryProfile& b)
    {
        return a.start_layer_index < b.start_layer_index;
    }
    static bool compare_size(const MemoryProfile& a, const MemoryProfile& b)
    {
        return a.size < b.size;
    }
};

using flexnn::LayerTimeProfile;
using flexnn::MemoryProfilerEvent;

// return >= offset, aligned to NCNN_MALLOC_ALIGN
static inline size_t alignOffsetBig(size_t offset)
{
    return ncnn::alignSize(offset, NCNN_MALLOC_ALIGN);
}

// return <= offset, aligned to NCNN_MALLOC_ALIGN
static inline size_t alignOffsetSmall(size_t offset)
{
    return ncnn::alignSize(offset - NCNN_MALLOC_ALIGN + 1, NCNN_MALLOC_ALIGN);
}

const std::string OUTPUT_PATH_PREFIX = "/home/root/flexnn/results/";

class FlexnnSchedule
{
public:
    FlexnnSchedule()
    {
        m_xy_plane = 0;
    }
    ~FlexnnSchedule()
    {
        if (m_xy_plane)
            delete m_xy_plane;
    }

    int init_xyplane(int x, int y)
    {
        if (m_xy_plane)
            delete m_xy_plane;
        m_xy_plane = new XYPlane(x, y, NCNN_MALLOC_ALIGN);
        return 0;
    }
    int read_profiles(const char* memory_profile_path, const char* time_profile_path);
    int read_memory_profile(const char* path);
    int read_time_profile(const char* path);
    int memory_events_to_profiles();

    // schedule functions: inputs -> memory_schedule
    int schedule_naive(const int memory_budget, char resultpath[256]);

    int schedule_naive_new(int memory_budget, char resultpath[256], char mode[256],int jump);

    // memory_schedule -> layer_dependency
    int resolve_layer_dependencies(const std::map<int, MemoryProfile>& memory_schedule, std::vector<int>& layer_dependencies);

    // predictor: layer_denpendencies -> latency
    double predict_latency(const std::vector<int>& layer_dependencies);

    // memory_schedule -> malloc_plan
    int generate_malloc_plan(const std::map<int, MemoryProfile>& memory_schedule, std::vector<std::vector<int> >& malloc_plan);

    // write to file
    int write_malloc_plan(const char* path) const;
    int write_layer_dependencies(const char* path) const;
    int write_memory_layout(const char* path) const;

    int generate_write_schedule(const char* malloc_plan_path, const char* layer_dependency_path, const char* memory_layout_path = 0);
    void print_predicted_latency();

    int get_layer_count() const;
    double get_total_loading_duration() const;
    double get_total_computing_duration() const;
    // int get_

    int get_malloc_plan(std::vector<std::vector<int> >& malloc_offsets, std::vector<int>& persistent_offsets)
    {
        generate_malloc_plan(m_memory_schedule, m_malloc_plan);
        malloc_offsets = m_malloc_plan;
        persistent_offsets = m_persistent_offsets;
        return 0;
    }
    int get_layer_dependencies(std::vector<int>& layer_dependencies)
    {
        resolve_layer_dependencies(m_memory_schedule, m_layer_dependencies);
        layer_dependencies = m_layer_dependencies;
        return 0;
    }
    int set_memory_profiles(const std::vector<MemoryProfilerEvent>& memory_profiler_events)
    {
        m_memory_profiler_events = memory_profiler_events;
        fprintf(stderr, "read %d memory events\n", (int)m_memory_profiler_events.size());
        return memory_events_to_profiles(); 
    }
    int set_time_profiles(const std::vector<LayerTimeProfile>& time_profiles)
    {
        m_time_profiles = time_profiles;
        fprintf(stderr, "read %d time profiles\n", (int)m_time_profiles.size());
        return 0;
    }

public:
    // outputs
    std::vector<int> m_layer_dependencies;
    std::vector<std::vector<int> > m_malloc_plan;

    // temp
    std::map<int, MemoryProfile> m_memory_schedule; // b.first=x=time, b.second=y=memory, sorted by x, for same x sorted by type and count (index)
    std::vector<int> m_persistent_offsets;
    std::vector<double> m_loading_begin;
    std::vector<double> m_loading_end;
    std::vector<double> m_computing_begin;
    std::vector<double> m_computing_end;

    // inputs
    std::vector<MemoryProfilerEvent> m_memory_profiler_events;
    std::vector<LayerTimeProfile> m_time_profiles;
    std::map<int, MemoryProfile> m_memory_profiles; // x=0, sorted by m_type and m_cnt
    int m_weight_count;
    int m_blob_count;
    int m_intermediate_count;
    int m_lifnode_count;

    // const
    int m_skip_layer_count = 1;

    XYPlane* m_xy_plane;
};

int FlexnnSchedule::get_layer_count() const
{
    int max_index = 0;
    for (auto profile : m_time_profiles)
    {
        if (profile.layer_index > max_index)
        {
            max_index = profile.layer_index;
        }
    }
    return max_index + 1;
}

int FlexnnSchedule::read_profiles(const char* memory_profile_path, const char* time_profile_path)
{
    read_memory_profile(memory_profile_path);
    read_time_profile(time_profile_path);

    return 0;
}

void FlexnnSchedule::print_predicted_latency()
{
    fprintf(stderr, "predicted latency: %f\n", predict_latency(m_layer_dependencies));
}

double FlexnnSchedule::get_total_loading_duration() const
{
    double total = 0;
    for (auto profile : m_time_profiles)
    {
        total += profile.loading_duration;
    }
    return total;
}

double FlexnnSchedule::get_total_computing_duration() const
{
    double total = 0;
    for (auto profile : m_time_profiles)
    {
        total += profile.computing_duration;
    }
    return total;
}

int FlexnnSchedule::read_memory_profile(const char* path)
{
    FILE* fp = fopen(path, "r");
    if (!fp)
    {
        fprintf(stderr, "fopen %s failed\n", path);
        return -1;
    }

    char line[256];

    while (fgets(line, sizeof(line), fp) != NULL)
    {
        if (line[0] == '#') // comment
            continue;
        if (strcmp(line, "layer_index,memory_type,event_type,ptr,size,time\n") == 0) // first line
            continue;

        flexnn::MemoryProfilerEvent event;
        int ret = sscanf(line, "%d,%d,%d,%p,%zu,%lf\n", &event.layer_index, &event.memory_type, &event.event_type, &event.ptr, &event.size, &event.time);
        if (ret != 6)
        {
            fprintf(stderr, "fscanf failed\n");
            break;
        }

        m_memory_profiler_events.push_back(event);
    }

    fprintf(stderr, "read %d memory events\n", (int)m_memory_profiler_events.size());

    return memory_events_to_profiles();
}

//将内存分配事件转为profiles
int FlexnnSchedule::memory_events_to_profiles()
{
    // events are ordered by malloc - free

    std::map<void*, int> memory_indices; //内存指针-profile_uuid对应表
    std::map<int, bool> profile_paired; //malloc-free对应表
    int counters[4] = {0, 0, 0, 0}; // malloc count of each type, 0 for weight, 1 for blob, 2 for intermediate
    int malloc_count = 0, free_count = 0;

    for (int i = 0; i < (int)m_memory_profiler_events.size(); i++)
    {
        const flexnn::MemoryProfilerEvent& event = m_memory_profiler_events[i];

        if (event.event_type == 1)
        {
            MemoryProfile profile;

            /*if (event.name.find("lif") != std::string::npos) {
                profile.start_layer_index = 0;
            }
            else {
                profile.start_layer_index = event.layer_index;
            }*/
            profile.start_layer_index = event.layer_index;
            
            profile.size = event.size;

            profile.memory_type = event.memory_type;
            profile.malloc_count = counters[event.memory_type]++;
            profile.name = event.name;
            memory_indices[event.ptr] = profile.memory_index();
            m_memory_profiles.insert({profile.memory_index(), profile});
            profile_paired.insert({profile.memory_index(), false});
            malloc_count++;
            
        }
        else if (event.event_type == 0)
        {
            if (memory_indices.find(event.ptr) == memory_indices.end())
            {
                fprintf(stderr, "free event with no pair malloc to pair %d %p\n", event.layer_index, event.ptr);
                continue;
            }
            if (profile_paired[memory_indices[event.ptr]])
            {
                fprintf(stderr, "free event with already paired malloc %d %p\n", event.layer_index, event.ptr);
                continue;
            }
            m_memory_profiles[memory_indices[event.ptr]].end_layer_index = event.layer_index;

            
            profile_paired[memory_indices[event.ptr]] = true;
            memory_indices.erase(event.ptr);
            free_count++;
        }
    }

    if (!memory_indices.empty())
    {
        fprintf(stderr, "memory free not detected:\n");
        for (auto it = memory_indices.begin(); it != memory_indices.end(); it++)
        {
            fprintf(stderr, "%p %d\n", it->first, it->second);
            fprintf(stderr, "%p allocated at layer %d\n", it->first, m_memory_profiles[it->second].start_layer_index);
        }
        return -1;
    }

    m_weight_count = counters[0];
    m_blob_count = counters[1];
    m_intermediate_count = counters[2];
    m_lifnode_count = counters[3];

    fprintf(stderr, "get %d memory profiles, malloc_count=%d, free_count=%d\n", (int)m_memory_profiles.size(), malloc_count, free_count);

    return 0;
}

int FlexnnSchedule::read_time_profile(const char* path)
{
    FILE* fp = fopen(path, "r");
    if (!fp)
    {
        fprintf(stderr, "fopen %s failed\n", path);
        return -1;
    }

    char line[256];

    while (fgets(line, sizeof(line), fp) != NULL)
    {
        if (line[0] == '#') // comment
            continue;
        if (strcmp(line, "layer_index,loading_begin,loading_end,loading_duration,computing_begin,computing_end,computing_duration\n") == 0) // first line
            continue;

        flexnn::LayerTimeProfile profile;
        int ret = sscanf(line, "%d,%lf,%lf,%lf,%lf,%lf,%lf\n", &profile.layer_index, &profile.loading_begin, &profile.loading_end, &profile.loading_duration, &profile.computing_begin, &profile.computing_end, &profile.computing_duration);
        if (ret != 7)
        {
            fprintf(stderr, "fscanf failed\n");
            break;
        }

        m_time_profiles.push_back(profile);
    }

    fprintf(stderr, "read %d time profiles\n", (int)m_time_profiles.size());

    // print total loading and computing
    fprintf(stderr, "total loading: %f, total computing: %f\n", get_total_loading_duration(), get_total_computing_duration());

    return 0;
}

int FlexnnSchedule::generate_malloc_plan(const std::map<int, MemoryProfile>& memory_schedule, std::vector<std::vector<int> >& malloc_plan)
{
    malloc_plan.resize(4);

    // schedule is sorted by x, type and count
    for (auto schedule : memory_schedule)
    {
        if (schedule.second.y < 0)
        {
            fprintf(stderr, "invalid y: %d at %d of lid %d\n", schedule.second.y, schedule.second.x, schedule.second.start_layer_index);
            fprintf(stderr, "%d,%d,%d,%d,%d,%d,%d\n", schedule.second.start_layer_index, schedule.second.end_layer_index, schedule.second.size, schedule.second.memory_type, schedule.second.malloc_count, schedule.second.x, schedule.second.y);
            return -1;
        }
        //printf("%x %d \n", schedule.second.y,schedule.second.size);
        malloc_plan[schedule.second.memory_type].push_back(schedule.second.y);
    }

    return 0;
}

int FlexnnSchedule::resolve_layer_dependencies(const std::map<int, MemoryProfile>& memory_schedule, std::vector<int>& layer_dependencies)
{
    std::vector<int> last_layer_before_loading(get_layer_count(), -1);
    layer_dependencies.resize(get_layer_count(), get_layer_count());
    for (int i = 0; i < m_skip_layer_count; i++)
    {
        layer_dependencies[i] = m_skip_layer_count + 1;
    }

    for (auto i = memory_schedule.begin(); i != memory_schedule.end(); i++)
    {
        if (i->second.memory_type == 0)
        {
            last_layer_before_loading[i->second.start_layer_index] = std::max(last_layer_before_loading[i->second.start_layer_index], i->second.x - 1);
        }
    }

    for (int i = 0; i < get_layer_count(); i++)
    {
        if (last_layer_before_loading[i] < m_skip_layer_count) // < or <=?
            continue;                                          // doesn't matter
        layer_dependencies[last_layer_before_loading[i] - 1] = std::min(layer_dependencies[last_layer_before_loading[i] - 1], i);
    }

    // dependency sequence should be monotonic non-decreasing
    for (int i = get_layer_count() - 1; i > 0; i--)
    {
        layer_dependencies[i - 1] = std::min(layer_dependencies[i], layer_dependencies[i - 1]);
    }

    // dependency is invalid if one layer depends on next
    for (int i = 0; i < get_layer_count() - 1; i++)
    {
        if (layer_dependencies[i] == i + 1)
        {
            fprintf(stderr, "layer %d depends on next layer %d\n", i, i + 1);
            return -1;
        }
    }

    return 0;
}

int FlexnnSchedule::write_malloc_plan(const char* path) const
{
    FILE* fp = fopen(path, "w");
    if (!fp)
    {
        fprintf(stderr, "fopen %s failed\n", path);
        return -1;
    }

    fprintf(fp, "# weight_count blob_count intermediate_count (persistent_count)\n");
    fprintf(fp, "%d %d %d", m_weight_count, m_blob_count, m_intermediate_count);
    if (!m_persistent_offsets.empty())
    {
        fprintf(fp, " %d", (int)m_persistent_offsets.size());
    }
    fprintf(fp, "\n");
    fprintf(fp, "# weight_offsets\n");
    for (int i = 0; i < (int)m_malloc_plan[0].size(); i++)
    {
        fprintf(fp, "%d\n", m_malloc_plan[0][i]);
    }
    fprintf(fp, "# blob_offsets\n");
    for (int i = 0; i < (int)m_malloc_plan[1].size(); i++)
    {
        fprintf(fp, "%d\n", m_malloc_plan[1][i]);
    }
    fprintf(fp, "# intermediate_offsets\n");
    for (int i = 0; i < (int)m_malloc_plan[2].size(); i++)
    {
        fprintf(fp, "%d\n", m_malloc_plan[2][i]);
    }
    fprintf(fp, "# persistent_offsets\n");
    for (int i = 0; i < (int)m_persistent_offsets.size(); i++)
    {
        fprintf(fp, "%d\n", m_persistent_offsets[i]);
    }

    fclose(fp);

    return 0;
}

int FlexnnSchedule::write_layer_dependencies(const char* path) const
{
    FILE* fp = fopen(path, "w");
    if (!fp)
    {
        fprintf(stderr, "fopen %s failed\n", path);
        return -1;
    }

    for (int i = 0; i < (int)m_layer_dependencies.size(); i++)
    {
        fprintf(fp, "%d\n", m_layer_dependencies[i]);
    }

    fclose(fp);

    return 0;
}

int FlexnnSchedule::write_memory_layout(const char* path) const
{
    FILE* fp = fopen(path, "w");
    if (!fp)
    {
        fprintf(stderr, "fopen %s failed\n", path);
        return -1;
    }

    for (auto schedule : m_memory_schedule)
    {
        auto profile = schedule.second;
        // fprintf(fp, "%d,%d,%d,%d,%d,%d,%d\n", schedule.second.start_layer_index, schedule.second.end_layer_index, schedule.second.size, schedule.second.memory_type, schedule.second.malloc_count, schedule.second.x, schedule.second.y);
        fprintf(fp, "%d,%d,%d,%d,%d,%d\n", profile.x, profile.end_layer_index, profile.y, profile.size, profile.start_layer_index, profile.memory_type);
    }

    // // write persistent offsets, use the same format
    // for (int i = 0; i < (int)m_persistent_offsets.size(); i++)
    // {
    //     fprintf(fp, "%d\n", m_persistent_offsets[i]);
    // }

    fclose(fp);

    return 0;
}

int FlexnnSchedule::generate_write_schedule(const char* malloc_plan_path, const char* layer_dependency_path, const char* memory_layout_path)
{
    if (!generate_malloc_plan(m_memory_schedule, m_malloc_plan))
        resolve_layer_dependencies(m_memory_schedule, m_layer_dependencies);

    if (!write_malloc_plan(malloc_plan_path))
        write_layer_dependencies(layer_dependency_path);

    if (memory_layout_path)
        write_memory_layout(memory_layout_path);

    return 0;
}

double FlexnnSchedule::predict_latency(const std::vector<int>& layer_dependencies)
{
    std::vector<double> loading_begin(get_layer_count(), .0f);
    std::vector<double> loading_end(get_layer_count(), .0f);
    std::vector<double> computing_begin(get_layer_count(), .0f);
    std::vector<double> computing_end(get_layer_count(), .0f);

    // layer 0: skip (Input)

    double tl = .0f, tc = .0f;
    loading_begin[m_skip_layer_count] = tl;
    tl += m_time_profiles[m_skip_layer_count].loading_duration;
    loading_end[m_skip_layer_count] = tl;

    for (int i = m_skip_layer_count; i < get_layer_count(); i++)
    {
        tc = std::max(loading_end[i], tc);
        computing_begin[i] = tc;
        tc += m_time_profiles[i].computing_duration;
        computing_end[i] = tc;

        // new loading tasks
        int start_index = layer_dependencies[i - 1];
        int end_index = layer_dependencies[i];
        for (int j = start_index; j < end_index; j++)
        {
            loading_begin[j] = tl;
            tl += m_time_profiles[j].loading_duration;
            loading_end[j] = tl;
        }
    }

    return tc;
}

//生成细粒度memory plan
int FlexnnSchedule::schedule_naive(int memory_budget, char resultpath[256])
{
    path = resultpath;

    // init xyplane
    init_xyplane(get_layer_count(), memory_budget);
    

    auto memory_profiles(m_memory_profiles);
    std::map<int, MemoryProfile> memory_schedule; // 用来存放已经生成的内存分配计划 (profile_uuid,planned_profile)

    // find min peak memory
    std::vector<int> layer_memory(get_layer_count(), 0); //每层内存需求
    std::vector<int> layer_weight_memory(get_layer_count(), 0); //每层权重的内存需求
    int total_weight_memory = 0; //总权重的内存体积(内存*层数)

    for (auto profile : memory_profiles)
    {
        if (profile.second.memory_type == 3)
            continue;

        for (int i = profile.second.start_layer_index; i <= profile.second.end_layer_index; i++)
        {
            layer_memory[i] += profile.second.size;
            if (profile.second.memory_type == 0)
            {
                layer_weight_memory[i] += profile.second.size;
                total_weight_memory += profile.second.size;
            }
        }
    }

    //找到内存峰值及对应层
    int peak_memory = 0, peak_index = -1; 
    for (int i = 0; i < layer_memory.size(); i++)
    {
        if (layer_memory[i] >= peak_memory)
        {
            peak_memory = layer_memory[i];
            peak_index = i;
        }
    }

    // 可分配的最大剩余内存，并利用它挑选权重作为持久化权重
    int max_memory_margin = memory_budget - peak_memory;
    auto persistent_weight_it = std::find_if(memory_profiles.begin(), memory_profiles.end(), [](const std::pair<int, MemoryProfile>& p) {
        return p.second.memory_type == 0;
    });

    auto lifnode_it = std::find_if(memory_profiles.begin(), memory_profiles.end(), [](const std::pair<int, MemoryProfile>& p) {
        return p.second.memory_type == 3;
    });

    // place higher score memory first, start from end of the buffe, don't exceed margin limit.
    // keep the order, important  <--WHY？？
    std::map<int, int> persistent_weights, lifnode_weights;
    int persistent_offset = alignOffsetSmall(memory_budget); //内存空间上限
    int persistent_min_offset = alignOffsetBig(memory_budget - max_memory_margin); //预留内存空间

    for (int i = 0; i < m_lifnode_count; i++)
    {
        int size = memory_profiles[lifnode_it->first].size;
        int next_offset = alignOffsetSmall(persistent_offset - size); //新内存上限
        if (next_offset < persistent_min_offset)
        {
            fprintf(stderr, " plan persistent lifnode weight size %d failed,lifnode number: %d.\n", size, i);
            return -1;
        }
        persistent_offset = next_offset;

        lifnode_weights.insert({lifnode_it->first, persistent_offset});

        // insert persistence memory
        m_xy_plane->insert_xrange_y(0, get_layer_count() - 1, persistent_offset, size, "persistent_weight_" + memory_profiles[lifnode_it->first].name);

        lifnode_it++;
    }


    // greedy
    // only do this when it's IO bound 
    // and 
    // there are enough memory!
    double compute_time = get_total_computing_duration();
    double load_time = get_total_loading_duration();
    fprintf(stderr, " compute time : %lf,load time : %lf.\n", compute_time, load_time);
    fprintf(stderr, " total_weight_memory : %d,layer_weight_memory[peak_index] : %d,max_memory_margin : %d.\n", total_weight_memory, layer_weight_memory[peak_index], max_memory_margin);
    if ((compute_time < 2 * load_time) && (0.7 * (total_weight_memory - layer_weight_memory[peak_index]) < max_memory_margin))
    {
        fprintf(stderr, " start to persistence the weights\n");
        for (int i = 0; i < m_weight_count; i++)
        {

            int size = memory_profiles[persistent_weight_it->first].size;
            int next_offset = alignOffsetSmall(persistent_offset - size); //新内存上限
            if (next_offset < persistent_min_offset)
                continue;
            persistent_offset = next_offset;
            persistent_weights.insert({persistent_weight_it->first, persistent_offset});
            m_xy_plane->insert_xrange_y(0, get_layer_count() - 1, persistent_offset, size, "persistent_weight_" + memory_profiles[persistent_weight_it->first].name);
            persistent_weight_it++;
        }
    }

    //持久化权重分配完毕，剩下空间供正常分配使用
    int dynamic_memory_budget = alignOffsetSmall(persistent_offset);

    // init xyplane
    //init_xyplane(get_layer_count(), dynamic_memory_budget);

    // greedy schedule
    auto weight_it = std::find_if(memory_profiles.begin(), memory_profiles.end(), [](const std::pair<int, MemoryProfile>& p) {
        return p.second.memory_type == 0;
    });
    auto blob_it = std::find_if(memory_profiles.begin(), memory_profiles.end(), [](const std::pair<int, MemoryProfile>& p) {
        return p.second.memory_type == 1;
    });
    auto intermediate_it = std::find_if(memory_profiles.begin(), memory_profiles.end(), [](const std::pair<int, MemoryProfile>& p) {
        return p.second.memory_type == 2;
    });

    auto lifnodes = std::find_if(memory_profiles.begin(), memory_profiles.end(), [](const std::pair<int, MemoryProfile>& p) {
        return p.second.memory_type == 3;
    });

    int lifnode_count = 0;

    while (lifnode_count < m_lifnode_count)
    {
        auto profile = lifnodes->second;

        if (lifnode_weights.find(lifnodes->first) != lifnode_weights.end())
        {
            // already scheduled as persistent weights
            // decide xy and add to memory schedule now
            profile.x = 0;
            profile.y = lifnode_weights[lifnodes->first];
            memory_schedule.insert({profile.memory_index(), profile});
        }

        lifnode_count++;
        lifnodes++;
    }


    int left = 0, right = dynamic_memory_budget;
    int layer_index = 0;
    // allocate blobs first, place blobs on two sides of the buffer
    // 按分配顺序遍历
    for (int i = 0; i < m_blob_count; i++)
    {
        MemoryProfile profile = blob_it->second;

        // new layer
        if (profile.start_layer_index > layer_index)
        {
            layer_index = profile.start_layer_index;
            // update left and right
            int next_left = 0, next_right = dynamic_memory_budget;

            //遍历已有计划，找到blob可用内存空间
            for (auto schedule : memory_schedule)
            {
                if (schedule.second.memory_type ==3)
                {
                    continue;
                }
                
                // find not freed blobs
                if (schedule.second.end_layer_index >= layer_index)
                {
                    if (schedule.second.start_layer_index % 2 == 0) //靠低位分配
                    {
                        next_left = std::max(next_left, schedule.second.y + schedule.second.size);
                    }
                    else //靠高位分配
                    {
                        next_right = std::min(next_right, schedule.second.y);
                    }
                }
            }
            left = next_left;
            right = next_right;
        }

        // 为profile的x y属性赋值代表从x层的y号地址开始分配内存空间和存活时间
        if (layer_index % 2 == 0)
        {
            profile.x = profile.start_layer_index;
            profile.y = alignOffsetBig(left);
            left = profile.y + profile.size;
        }
        else
        {
            profile.x = profile.start_layer_index;
            profile.y = alignOffsetSmall(right - profile.size);
            right = profile.y;
        }

        // 加入内存分配计划和分配工具对象m_xy_plane
        memory_schedule.insert({profile.memory_index(), profile});
        m_xy_plane->insert_xrange_y(profile.start_layer_index, profile.end_layer_index, profile.y, profile.size, "blob_" + profile.name); // insert to xyplane
        //m_xy_plane->insert_xrange_y(profile.start_layer_index, profile.end_layer_index, profile.y, profile.size);
        blob_it++;
    }

    // first allocate blobs, then weights and intermediates, try to preload weights.
    // place weights and intermediates in the middle of the buffer
    // first try: preload weights as much as possible, then schedule intermediates
    // fallback plan: don't preload at all, schedule weights and intermediates together

    // note that some weights are already scheduled as persistent weights and erased from memory_profiles
    int weight_count = 0;
    int intermediate_count = 0;
    int loading_x = 0; // preload可用层x
    bool is_success = true; // 是否分配成功
    int max_preload_count = 50; // 最多可以提前加载的层数

    // 对每一层分配所有可能的layer weights
    for (int i = 0; i < get_layer_count(); i++)
    {
        // fprintf(stderr, "schedule layer %d.\n", i);
        is_success = true;

        //备份，防止preload分配失败
        m_xy_plane->backup();
        int weight_count_backup = weight_count;
        int intermediate_count_backup = intermediate_count;
        auto weight_it_backup = weight_it;
        auto intermediate_it_backup = intermediate_it;

        // try preload weights
        while (weight_count < m_weight_count)
        {
            auto profile = weight_it->second;
            if (profile.start_layer_index > i)
                break;
            if (profile.memory_type != 0)
                break;

            if (persistent_weights.find(weight_it->first) != persistent_weights.end())
            {
                // already scheduled as persistent weights
                // decide xy and add to memory schedule now
                profile.x = loading_x;
                profile.y = persistent_weights[weight_it->first];
                memory_schedule.insert({profile.memory_index(), profile});
            }
            else {
                loading_x = std::max(loading_x, profile.start_layer_index - max_preload_count);

                std::pair<int, int> ret;
                if (profile.name.find("lif") != std::string::npos) {
                    ret = m_xy_plane->insert_xrange(0, get_layer_count() - 1, profile.size, "weight_" + profile.name);
                }
                else {
                    ret = m_xy_plane->insert_xrange(loading_x, profile.end_layer_index, profile.size, "weight_" + profile.name);
                }
                //auto ret = m_xy_plane->insert_xrange(loading_x, profile.end_layer_index, profile.size);
                if (ret.first < 0 || ret.second < 0)
                {
                    fprintf(stderr, "preload schedule weight size %d failed,start layer %d, ret={%d, %d}.\n", profile.size,profile.start_layer_index, ret.first, ret.second);
                    is_success = false;
                    break;
                }

                profile.x = loading_x;
                profile.y = ret.second; // aligned
                memory_schedule.insert({profile.memory_index(), profile});

                loading_x = profile.x; // next loading starts not before x
            }
            weight_count++;
            weight_it++;
        }

        if (!is_success)
        {
            std::string payout_path = std::string(path) + "xyplane_old_" + std::to_string(memory_budget) + ".payout";
            std::string budget_path = std::string(path) + "xyplane_old_" + std::to_string(memory_budget) + ".budget";
            m_xy_plane->save_payouts(payout_path.c_str(), i + 1);
            m_xy_plane->save_budgets(budget_path.c_str(), i + 1);
        }

        // schedule intermediates
        while (is_success && intermediate_count < m_intermediate_count)
        {
            auto profile = intermediate_it->second;
            if (profile.start_layer_index > i)
                break;
            if (profile.memory_type != 2)
                break;

            auto ret = m_xy_plane->insert_xrange(profile.start_layer_index, profile.end_layer_index, profile.size,  "intermediate_" +profile.name);
            //auto ret = m_xy_plane->insert_xrange(profile.start_layer_index, profile.end_layer_index, profile.size);
            if (ret.first < 0 || ret.second < 0)
            {
                fprintf(stderr, "preload schedule intermediate size %d failed,start layer %d, ret={%d, %d}.\n", profile.size, profile.start_layer_index, ret.first, ret.second);
                is_success = false;
                break;
            }
            profile.x = ret.first;
            profile.y = ret.second; // aligned
            memory_schedule.insert({profile.memory_index(), profile});

            intermediate_count++;
            intermediate_it++;
        }

        if (!is_success)
        {
            // re-schedule this layer
            fprintf(stderr, "re-schedule layer %d.\n", i);
            m_xy_plane->restore();
            is_success = true;
            weight_count = weight_count_backup;
            intermediate_count = intermediate_count_backup;
            weight_it = weight_it_backup;
            intermediate_it = intermediate_it_backup;

            while (weight_count < m_weight_count)
            {
                auto profile = weight_it->second;
                if (profile.start_layer_index > i)
                    break;
                if (profile.memory_type != 0)
                    break;

                if (persistent_weights.find(weight_it->first) != persistent_weights.end())
                {
                    // already scheduled as persistent weights
                    // dicide xy and add to memory schedule now
                    profile.x = profile.start_layer_index;
                    profile.y = persistent_weights[weight_it->first];
                    memory_schedule.insert({profile.memory_index(), profile});

                    weight_count++;
                    weight_it++;
                    continue;
                }
                else {

                    std::pair<int, int> ret;
                    if (profile.name.find("lif") != std::string::npos)
                    {
                        ret = m_xy_plane->insert_xrange(0, get_layer_count() - 1, profile.size, "weight_" + profile.name);
                    }
                    else
                    {
                        ret = m_xy_plane->insert_xrange(profile.start_layer_index, profile.end_layer_index, profile.size, "weight_" + profile.name);
                    }

                    //auto ret = m_xy_plane->insert_xrange(profile.start_layer_index, profile.end_layer_index, profile.size);
                    if (ret.first < 0 || ret.second < 0)
                    {
                        fprintf(stderr, "re-schedule weight size %d failed,start layer %d, ret={%d, %d}.\n", profile.size,profile.start_layer_index, ret.first, ret.second);
                        is_success = false;
                        break;
                    }
                    profile.x = profile.start_layer_index;
                    profile.y = ret.second; // aligned
                    memory_schedule.insert({profile.memory_index(), profile});

                    loading_x = profile.x; // next loading starts not before x
                    weight_count++;
                    weight_it++;
                }
            }

            if (!is_success)
            {
                std::string payout_path = std::string(path) + "xyplane_old_" + std::to_string(memory_budget) + ".payout";
                std::string budget_path = std::string(path) + "xyplane_old_" + std::to_string(memory_budget) + ".budget";
                m_xy_plane->save_payouts(payout_path.c_str(), i + 1);
                m_xy_plane->save_budgets(budget_path.c_str(), i + 1);
                break;
            }

            // schedule intermediates
            while (intermediate_count < m_intermediate_count)
            {
                auto profile = intermediate_it->second;
                if (profile.start_layer_index > i)
                    break;
                if (profile.memory_type != 2)
                    break;

                auto ret = m_xy_plane->insert_xrange(profile.start_layer_index, profile.end_layer_index, profile.size, "intermediate_" + profile.name);
                //auto ret = m_xy_plane->insert_xrange(profile.start_layer_index, profile.end_layer_index, profile.size);
                if (ret.first < 0 || ret.second < 0)
                {
                    fprintf(stderr, "re-schedule intermediate size %d failed,start layer %d, ret={%d, %d}.\n", profile.size,profile.start_layer_index, ret.first, ret.second);
                    is_success = false;
                    break;
                }
                profile.x = ret.first;
                profile.y = ret.second; // aligned
                memory_schedule.insert({profile.memory_index(), profile});

                intermediate_count++;
                intermediate_it++;
            }

            if (!is_success)
            {
                std::string payout_path = std::string(path) + "xyplane_old_" + std::to_string(memory_budget) + ".payout";
                std::string budget_path = std::string(path) + "xyplane_old_" + std::to_string(memory_budget) + ".budget";
                m_xy_plane->save_payouts(payout_path.c_str(), i + 1);
                m_xy_plane->save_budgets(budget_path.c_str(), i + 1);
                break;
            }
        }
    }

    std::string payout_path = std::string(path) + "xyplane_old_" + std::to_string(memory_budget) + ".payout";
    std::string budget_path = std::string(path) + "xyplane_old_" + std::to_string(memory_budget) + ".budget";
    m_xy_plane->save_payouts(payout_path.c_str(), get_layer_count());
    m_xy_plane->save_budgets(budget_path.c_str(), get_layer_count());

    if (!is_success)
    {
        fprintf(stderr, "schedule failed.\n");
        return -1;
    }

    // copy the schedule
    m_memory_schedule = memory_schedule;

    // offsets are persistent weights' offsets (values)
    m_persistent_offsets.clear();
    for (auto weight : persistent_weights)
    {
        m_persistent_offsets.push_back(weight.second);
    }

    return 0;
}

int FlexnnSchedule::schedule_naive_new(int memory_budget, char resultpath[256], char mode[256],int jump)
{
    path = resultpath;

    std::map<int,MemoryProfile> memory_profiles(m_memory_profiles);
    std::map<int, MemoryProfile> memory_schedule; 
    init_xyplane(get_layer_count(), memory_budget);

    // find min peak memory
    std::vector<int> layer_memory(get_layer_count(), 0);        
    std::vector<int> layer_weight_memory(get_layer_count(), 0); 
    int total_weight_memory = 0; 

    for (auto profile : memory_profiles)
    {
        if (profile.second.memory_type == 3)
            continue;

        for (int i = profile.second.start_layer_index; i <= profile.second.end_layer_index; i++)
        {
            layer_memory[i] += profile.second.size;
            if (profile.second.memory_type == 0)
            {
                layer_weight_memory[i] += profile.second.size;
                total_weight_memory += profile.second.size;
            }
        }
    }

    int peak_memory = 0, peak_index = -1;
    for (int i = 0; i < layer_memory.size(); i++)
    {
        if (layer_memory[i] >= peak_memory)
        {
            peak_memory = layer_memory[i];
            peak_index = i;
        }
    }

    int max_memory_margin = memory_budget - peak_memory;
    auto persistent_weight_it = std::find_if(memory_profiles.begin(), memory_profiles.end(), [](const std::pair<int, MemoryProfile>& p) {
        return p.second.memory_type == 0;
    });

    auto lifnode_it = std::find_if(memory_profiles.begin(), memory_profiles.end(), [](const std::pair<int, MemoryProfile>& p) {
        return p.second.memory_type == 3;
    });

    auto persistent_blob_it = std::find_if(memory_profiles.begin(), memory_profiles.end(), [](const std::pair<int, MemoryProfile>& p) {
        return p.second.memory_type == 1;
    });

    std::map<int, int> persistent_weights,lifnode_weights;
    int persistent_offset = alignOffsetSmall(memory_budget);                       
    int persistent_min_offset = alignOffsetBig(memory_budget - max_memory_margin); 

    
    for (int i = 0; i < m_lifnode_count; i++)
    {
        int size = memory_profiles[lifnode_it->first].size;
        int next_offset = alignOffsetSmall(persistent_offset - size); //新内存上限
        if (next_offset < persistent_min_offset)
        {
            fprintf(stderr, " plan persistent lifnode weight size %d failed,lifnode number: %d.\n", size, i);
            return -1;
        }
        persistent_offset = next_offset;
        
        lifnode_weights.insert({lifnode_it->first, persistent_offset});

        // insert persistence memory
        m_xy_plane->insert_xrange_y(0, get_layer_count() - 1, persistent_offset, size, "persistent_weight_" + memory_profiles[lifnode_it->first].name);

        lifnode_it++;
    }
    
    
    double compute_time = get_total_computing_duration();
    double load_time = get_total_loading_duration();
    fprintf(stderr, " compute time : %lf,load time : %lf.\n", compute_time, load_time);
    fprintf(stderr, " total_weight_memory : %d,layer_weight_memory[peak_index] : %d,max_memory_margin : %d.\n", total_weight_memory, layer_weight_memory[peak_index], max_memory_margin);
    if (std::string(mode).find("normal") != std::string::npos && std::string(mode).find("npos") == std::string::npos && (compute_time < 2 * load_time) && (0.7 * (total_weight_memory - layer_weight_memory[peak_index]) < max_memory_margin))
    {
        fprintf(stderr, " start to persistence the weights\n");
        for (int i = 0; i < m_weight_count; i++)
        {
            int size = memory_profiles[persistent_weight_it->first].size;
            int next_offset = alignOffsetSmall(persistent_offset - size); //新内存上限
            if (next_offset < persistent_min_offset)
                continue;
            persistent_offset = next_offset;
            persistent_weights.insert({persistent_weight_it->first, persistent_offset});
            // insert persistence memory
            m_xy_plane->insert_xrange_y(0, get_layer_count() - 1, persistent_offset, size, "persistent_weight_" + memory_profiles[persistent_weight_it->first].name);

            persistent_weight_it++;
        }
    }
    else if (std::string(mode).find("all_persistent") != std::string::npos)
    {
        fprintf(stderr, " start to persistence the weights\n");
        for (int i = 0; i < m_weight_count; i++)
        {
            int size = memory_profiles[persistent_weight_it->first].size;
            int next_offset = alignOffsetSmall(persistent_offset - size); //新内存上限
            if (next_offset < persistent_min_offset)
                continue;
            persistent_offset = next_offset;
            persistent_weights.insert({persistent_weight_it->first, persistent_offset});
            // insert persistence memory
            m_xy_plane->insert_xrange_y(0, get_layer_count() - 1, persistent_offset, size, "persistent_weight_" + memory_profiles[persistent_weight_it->first].name);

            persistent_weight_it++;
        }
    }

    int dynamic_memory_budget = alignOffsetSmall(persistent_offset);

    //init_xyplane(get_layer_count(), dynamic_memory_budget);

    auto weight_it = std::find_if(memory_profiles.begin(), memory_profiles.end(), [](const std::pair<int, MemoryProfile>& p) {
        return p.second.memory_type == 0;
    });
    auto blob_it = std::find_if(memory_profiles.begin(), memory_profiles.end(), [](const std::pair<int, MemoryProfile>& p) {
        return p.second.memory_type == 1;
    });
    auto intermediate_it = std::find_if(memory_profiles.begin(), memory_profiles.end(), [](const std::pair<int, MemoryProfile>& p) {
        return p.second.memory_type == 2;
    });

    auto lifnodes = std::find_if(memory_profiles.begin(), memory_profiles.end(), [](const std::pair<int, MemoryProfile>& p) {
        return p.second.memory_type == 3;
    });

    int lifnode_count = 0;

    while (lifnode_count < m_lifnode_count)
    {
        auto profile = lifnodes->second;

        if (lifnode_weights.find(lifnodes->first) != lifnode_weights.end())
        {
            // already scheduled as persistent weights
            // decide xy and add to memory schedule now
            profile.x = 0;
            profile.y = lifnode_weights[lifnodes->first];
            memory_schedule.insert({profile.memory_index(), profile});
        }
        

        lifnode_count++;
        lifnodes++;
    }
    

    int left = 0, right = dynamic_memory_budget;
    int layer_index = 0;
    
    for (int i = 0; i < m_blob_count; i++)
    {
        MemoryProfile profile = blob_it->second;

        if (profile.start_layer_index > layer_index)
        {
            layer_index = profile.start_layer_index;
            
            int next_left = 0, next_right = dynamic_memory_budget;

            for (auto schedule : memory_schedule)
            {
                if (schedule.second.memory_type == 3)
                {
                    continue;
                }
                if (schedule.second.end_layer_index >= layer_index)
                {
                    if (schedule.second.start_layer_index % 2 == 0) 
                    {
                        next_left = std::max(next_left, schedule.second.y + schedule.second.size);
                    }
                    else 
                    {
                        next_right = std::min(next_right, schedule.second.y);
                    }
                }
            }
            left = next_left;
            right = next_right;
        }

        
        if (layer_index % 2 == 0)
        {
            profile.x = profile.start_layer_index;
            profile.y = alignOffsetBig(left);
            left = profile.y + profile.size;
        }
        else
        {
            profile.x = profile.start_layer_index;
            profile.y = alignOffsetSmall(right - profile.size);
            right = profile.y;
        }

        
        memory_schedule.insert({profile.memory_index(), profile});
        m_xy_plane->insert_xrange_y(profile.start_layer_index, profile.end_layer_index, profile.y, profile.size, "blob_" + profile.name); // insert to xyplane

        blob_it++;
    }

    // first allocate blobs, then weights and intermediates, try to preload weights.
    // place weights and intermediates in the middle of the buffer
    // first try: preload weights as much as possible, then schedule intermediates
    // fallback plan: don't preload at all, schedule weights and intermediates together

    // note that some weights are already scheduled as persistent weights and erased from memory_profiles
    int weight_count = 0;
    int intermediate_count = 0;
    int loading_x = 0;         
    bool is_success = true;     
    int max_preload_count = 50; 

    int pre_loc = -1;
    std::vector<MemoryProfile> task_queue;

    //备份，防止preload分配失败
    m_xy_plane->backup();
    int weight_count_backup = weight_count;
    auto weight_it_backup = weight_it;

    //分配intermediates
    while (intermediate_count < m_intermediate_count)
    {
        auto profile_intermediate = intermediate_it->second;

        auto ret = m_xy_plane->insert_xrange(profile_intermediate.start_layer_index, profile_intermediate.end_layer_index, profile_intermediate.size, "intermediate_" + profile_intermediate.name);
        if (ret.first < 0 || ret.second < 0)
        {
            fprintf(stderr, "preload schedule intermediate size %d failed,start layer %d, ret={%d, %d}.\n", profile_intermediate.size, profile_intermediate.start_layer_index, ret.first, ret.second);
            fprintf(stderr, "schedule failed.\n");
            return -1;
        }
        profile_intermediate.x = ret.first;
        profile_intermediate.y = ret.second; // aligned
        memory_schedule.insert({profile_intermediate.memory_index(), profile_intermediate});

        intermediate_count++;
        intermediate_it++;
    }

    m_xy_plane->backup();

    for (int i = 0; i < get_layer_count(); i++)
    {
        
        // try preload weights
        while (weight_count < m_weight_count)
        {
            if (jump == 1)
            {
                is_success = false;
                break;
            }
            auto profile = weight_it->second;
            if (profile.start_layer_index > i)
            {
                break;
            }
            if (profile.memory_type != 0)
            {
                break;
            }

            if (persistent_weights.find(weight_it->first) != persistent_weights.end())
            {
                // already scheduled as persistent weights
                // decide xy and add to memory schedule now
                profile.x = loading_x;
                profile.y = persistent_weights[weight_it->first];
                memory_schedule.insert({profile.memory_index(), profile});
            }
            else
            {
                loading_x = std::max(loading_x, profile.start_layer_index - max_preload_count);
                auto ret = m_xy_plane->insert_xrange(loading_x, profile.end_layer_index, profile.size, "weight_" + profile.name);
                
                if (ret.first < 0 || ret.second < 0)
                {
                    // insert_xtange
                    m_xy_plane->restore();
                    std::sort(task_queue.begin(), task_queue.end(), [](MemoryProfile a, MemoryProfile b) { return a.end_layer_index >= b.end_layer_index; });

                    // insert_xtange
                    for (std::vector<MemoryProfile>::iterator task_it = task_queue.begin(); task_it != task_queue.end(); task_it++)
                    {
                        MemoryProfile p = *task_it;
                        
                        auto ret = m_xy_plane->insert_xrange(pre_loc, p.end_layer_index, p.size, "weight_" + p.name);

                        if (ret.first < 0 || ret.second < 0)
                        {
                            fprintf(stderr, "preload scheduled weight size %d failed,start layer %d, ret={%d, %d}.\n", p.size, p.start_layer_index, ret.first, ret.second);
                            is_success = false;
                            break;
                        }

                        p.x = ret.first;
                        p.y = ret.second; // aligned
                        memory_schedule.insert({p.memory_index(), p});
                    }
                    task_queue.clear();
                    m_xy_plane->backup();

                    fprintf(stderr, "preload schedule weight size %d failed,start layer %d, ret={%d, %d}.ready to malloc with no preloading\n", profile.size, profile.start_layer_index, ret.first, ret.second);
                    is_success = false;
                    break;
                }
                else if (pre_loc == -1 || pre_loc == ret.first)
                {
                    pre_loc = ret.first;
                    task_queue.push_back(profile);
                }
                else if (pre_loc != ret.first)
                {
                    m_xy_plane->restore();

                    std::sort(task_queue.begin(), task_queue.end(), [](MemoryProfile a, MemoryProfile b) { return a.end_layer_index >= b.end_layer_index; });

                    // insert_xtange
                    for (std::vector<MemoryProfile>::iterator task_it = task_queue.begin(); task_it != task_queue.end(); task_it++)
                    {
                        MemoryProfile p = *task_it;

                        auto ret = m_xy_plane->insert_xrange(pre_loc, p.end_layer_index, p.size, "weight_" + p.name);
                        if (ret.first < 0 || ret.second < 0)
                        {
                            fprintf(stderr, "preload scheduled weight size %d failed,start layer %d, ret={%d, %d}.\n", profile.size, profile.start_layer_index, ret.first, ret.second);
                            is_success = false;
                            break;
                        }
                        p.x = ret.first;
                        p.y = ret.second; // aligned
                        memory_schedule.insert({p.memory_index(), p});
                    }
                    task_queue.clear();
                    m_xy_plane->backup();

                    ret = m_xy_plane->insert_xrange(loading_x, profile.end_layer_index, profile.size, "weight_" + profile.name);

                    task_queue.push_back(profile);
                    pre_loc = ret.first;
                }

                loading_x = ret.first;
            }
            weight_count++;
            weight_it++;
        }

        if (!is_success)
        {
            // re-schedule this layer
            fprintf(stderr, "re-schedule layer %d.\n", i);
            m_xy_plane->restore();
            is_success = true;


            while (weight_count < m_weight_count)
            {
                auto profile = weight_it->second;
                if (profile.start_layer_index > i)
                    break;
                if (profile.memory_type != 0)
                    break;

                if (persistent_weights.find(weight_it->first) != persistent_weights.end())
                {
                    // already scheduled as persistent weights
                    // dicide xy and add to memory schedule now
                    profile.x = profile.start_layer_index;
                    profile.y = persistent_weights[weight_it->first];
                    memory_schedule.insert({profile.memory_index(), profile});

                    weight_count++;
                    weight_it++;
                    continue;
                }
                else
                {
                    auto ret = m_xy_plane->insert_xrange(profile.start_layer_index, profile.end_layer_index, profile.size, "weight_" + profile.name);
                    if (ret.first < 0 || ret.second < 0)
                    {
                        fprintf(stderr, "re-schedule weight size %d failed,start layer %d, ret={%d, %d}.\n", profile.size, profile.start_layer_index, ret.first, ret.second);
                        is_success = false;
                        break;
                    }
                    profile.x = ret.first;
                    profile.y = ret.second; // aligned
                    memory_schedule.insert({profile.memory_index(), profile});
                    m_xy_plane->backup();

                    loading_x = profile.x; // next loading starts not before x
                    weight_count++;
                    weight_it++;
                }
            }
        }
    }

    if (!task_queue.empty()) {
        m_xy_plane->restore();

        std::sort(task_queue.begin(), task_queue.end(), [](MemoryProfile a, MemoryProfile b) { return a.end_layer_index >= b.end_layer_index; });

        // insert_xtange
        for (std::vector<MemoryProfile>::iterator task_it = task_queue.begin(); task_it != task_queue.end(); task_it++)
        {
            MemoryProfile p = *task_it;

            auto ret = m_xy_plane->insert_xrange(pre_loc, p.end_layer_index, p.size, "weight_" + p.name);

            if (ret.first < 0 || ret.second < 0)
            {
                fprintf(stderr, "preload scheduled weight size %d failed,start layer %d, ret={%d, %d}.\n", p.size, p.start_layer_index, ret.first, ret.second);
                is_success = false;
                break;
            }

            p.x = ret.first;
            p.y = ret.second; // aligned
            memory_schedule.insert({p.memory_index(), p});
        }
        task_queue.clear();
    }


    std::string payout_path = std::string(path) + "xyplane_" + std::string(mode) + "_" + std::to_string(memory_budget) + ".payout";
    std::string budget_path = std::string(path) + "xyplane_" + std::string(mode) + "_" + std::to_string(memory_budget) + ".budget";
    m_xy_plane->save_payouts(payout_path.c_str(), get_layer_count());
    m_xy_plane->save_budgets(budget_path.c_str(), get_layer_count());

    if (!is_success)
    {
        fprintf(stderr, "schedule failed.\n");
        return -1;
    }

    // copy the schedule
    m_memory_schedule = memory_schedule;

    // offsets are persistent weights' offsets (values)
    m_persistent_offsets.clear();

    if (std::string(mode).find("no_persistent") == std::string::npos)
    {
        for (auto weight : persistent_weights)
        {
            m_persistent_offsets.push_back(weight.second);
        }
    }
    

    return 0;
}

#endif // FLEXNN_SCHEDULE_H