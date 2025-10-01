#ifndef POWER_LOGGER_H
#define POWER_LOGGER_H
#endif
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>
#include <system_error>
#include <unistd.h>
#include <cctype>
#include "mat.h"
#include "stdlib.h"
#include "errno.h"

const std::vector<std::vector<std::string> > device_nodes = {{"vdd_in", "0040", "1"}, {"vdd_cpu_gpu_cv", "0040", "2"}, {"vdd_soc", "0040", "3"}};

const std::string driver_dir = "/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon3/";

const std::string _valTypes[]
    = {"power", "voltage", "current"};

const std::string _valTypesFull[]
    = {"power [mW]", "voltage [mV]", "current [mA]"};


class PowerLogger
{
public:
    PowerLogger();
    PowerLogger(double _interval, int _log_level, std::string _path = NULL);
    void start();
    void stop();
    long get_process_rss();
    void _timer_func();
    std::vector<double> get_all_values();
    double read_value(std::string i2cAddr, std::string channel);
    std::vector<double> get_total_energy();
    std::vector<long> get_max_infer_memory_usage();
    long get_max_memory_usage();
    void record_event(std::string event);
    void save();
    void clear();
    void append();

public:
    FILE* out_file_power = NULL;
    FILE* out_file_infer = NULL;
    double interval;
    double start_time = -1;
    std::vector<std::pair<double, std::vector<double> > > data_log;
    std::vector<std::pair<double, long>> memory_log;
    std::vector<std::pair<double, std::string> > event_log;
    std::string device;
    std::vector<std::vector<std::string>> nodes;
    std::thread* current_thread;
    long max_memory_usage = 0;
    std::vector<long> max_infer_memory = {0};
    int config_num = 0;
    bool flag = true;
    pid_t pid = getpid();
    std::string path = "";
    int log_level = 0;
};

PowerLogger::PowerLogger(){}

PowerLogger::PowerLogger(double _interval,int _log_level, std::string _path)
{
    log_level = _log_level;
    path = _path;
    interval = _interval;
    device = "jetson_orin_nx";
    nodes = device_nodes;
    printf("pid is %d\n", pid);

    out_file_power = fopen(std::string(_path + "power_memory.txt").c_str(), "w+");
    out_file_infer = fopen(std::string(_path + "infer_time.txt").c_str(), "w+");
}

void PowerLogger::clear() {
    this->data_log.clear();
    this->event_log.clear();
    this->memory_log.clear();
}

void PowerLogger::append()
{
    out_file_power = fopen(std::string(path + "power_memory.txt").c_str(), "a");
    int length_power = data_log.size();
    int length_event = event_log.size();
    printf("will append power info length:%d ; event info length:%d\n", length_power, length_event);
    if (out_file_power)
    {
        if (length_event != 0)
        {
            int event_count = 0;
            int i = 0;
            while (i < length_power)
            {
                double time = data_log[i].first;
                double event_time = event_log[event_count].first;
                if (event_time >= time || event_count >= length_event)
                {
                    fprintf(out_file_power, "%lf:", time);
                    std::vector<double> data_info = data_log[i].second;
                    long memo = memory_log[i].second;
                    fprintf(out_file_power, "%ld,", memo);
                    for (int j = 0; j < data_info.size(); j++)
                    {
                        fprintf(out_file_power, "%lf", data_info[j]);
                        if (j != data_info.size() - 1) {
                            fprintf(out_file_power, ",");
                        } else {
                            fprintf(out_file_power, "\n");
                        }
                    }
                    
                    i++;
                }
                else
                {
                    if (log_level >= 2)
                        fprintf(out_file_power, "%lf:event:%s\n", event_time, event_log[event_count].second.c_str());
                    event_count++;
                }
            }

            while (event_count < length_event)
            {
                if (log_level >= 2)
                    fprintf(out_file_power, "%lf:event:%s\n", event_log[event_count].first, event_log[event_count].second.c_str());
                event_count++;
            }
        }
        else
        {
            for (int i = 0; i < length_power; i++)
            {
                fprintf(out_file_power, "%lf:", time);
                std::vector<double> data_info = data_log[i].second;
                long memo = memory_log[i].second;
                fprintf(out_file_power, "%ld,", memo);
                for (int j = 0; j < data_info.size(); j++)
                {
                    fprintf(out_file_power, "%lf", data_info[j]);
                    if (j != data_info.size() - 1)
                    {
                        fprintf(out_file_power, ",");
                    }
                    else
                    {
                        fprintf(out_file_power, "\n");
                    }
                }

                i++;
            }
        }
    }
    printf("power and memory data saved.\n");
    fclose(out_file_power);
}

void PowerLogger::record_event(std::string event)
{
    double time = flexnn::get_current_time() - this->start_time;

    this->event_log.push_back({time, event});
}

double PowerLogger::read_value(std::string i2cAddr, std::string channel)
{
    double res,voltage,current;
    char buffer[100];

    std::string type = "voltage";
    std::string val = "in";
    sprintf(buffer, "/sys/bus/i2c/drivers/ina3221/1-%s/hwmon/hwmon3/%s%s_input", i2cAddr.c_str(), val.c_str(), channel.c_str());

    FILE* f = fopen(buffer, "r");
    //perror("Error opening file");
    fscanf(f, "%lf", &voltage);
    fclose(f);

    type = "current";
    val = "curr";
    sprintf(buffer, "/sys/bus/i2c/drivers/ina3221/1-%s/hwmon/hwmon3/%s%s_input", i2cAddr.c_str(), val.c_str(), channel.c_str());

    f = fopen(buffer, "r");
    fscanf(f, "%lf", &current);
    fclose(f);

    return voltage * current / 1000;
}

void PowerLogger::stop()
{
    flag = false;
}

std::vector<double> PowerLogger::get_all_values()
{
    std::vector<double > res;
    for (int i = 0; i < this->nodes.size(); i++) {
        std::vector<std::string> node = nodes[i];
        res.push_back(read_value(node[1], node[2]));
    }

    return res;
}

long PowerLogger::get_process_rss()
{
    std::string path = "/proc/" + std::to_string(pid) + "/status";
    std::ifstream file(path);

    std::string line;
    while (std::getline(file, line))
    {
        if (line.find("VmRSS:") == 0) {
            size_t colon_pos = line.find(':');
            if (colon_pos == std::string::npos) continue;
            //size_t num_start = line.find_first_not_of("\0\t",colon_pos + 1);
            size_t num_start = line.find_first_of("0123456789",colon_pos + 1);
            if (num_start == std::string::npos) continue;
            size_t num_end = line.find_first_not_of("0123456789", num_start);
            if (num_end == std::string::npos || num_end < num_start) {
                num_end = line.length();
            }
            if (num_end - num_start == 0) {
                return 0;
            }
            else {
                std::string num_str = line.substr(num_start, num_end - num_start);
                long memory_usage = std::stol(num_str);
                max_memory_usage = std::max(max_memory_usage, memory_usage);
                file.close();
                return memory_usage;
            }
        }
    }

    return -1;
}

void PowerLogger::_timer_func()
{
    if (!flag) {
        return;
    }
    double time = flexnn::get_current_time() - this->start_time;

    this->data_log.push_back({time, get_all_values()});
    this->memory_log.push_back({time, get_process_rss()});
    this->start();
}

void PowerLogger::start()
{
    flag = true;
    int delay = this->interval;
    std::thread t([delay,this]() {
        //printf("sleep at : %lf\n", flexnn::get_current_time());
        std::this_thread::sleep_for(std::chrono::microseconds(delay));
        //printf("wake at : %lf\n", flexnn::get_current_time());
        this->_timer_func();
    });
    current_thread = &t;
    current_thread->detach();
    if (this->start_time < 0) {
        this->start_time = flexnn::get_current_time();
    }
}

void PowerLogger::save()
{
    bool infer = false;
    int infer_count = 0;
    int length_power = data_log.size();
    int length_event = event_log.size();

    int last_infer_num = -1;
    double last_infer_time = -1;
    long peak_memo = -1;

    printf("will save power info length:%d ; event info length:%d\n", length_power, length_event);
    if (out_file_power && out_file_infer)
    {
        if (length_event != 0) {
            int event_count = 0;
            int i = 0;
            while (i < length_power)
            {
                double time = data_log[i].first;
                double event_time = event_log[event_count].first;
                if (event_count >= length_event || event_time >= time)
                {
                    fprintf(out_file_power, "%lf:", time);
                    std::vector<double> data_info = data_log[i].second;
                    long memo = memory_log[i].second;

                    if (last_infer_num != -1)
                    {
                        if (memo > peak_memo)
                        {
                            peak_memo = memo;
                        }
                    }

                    if (infer) {
                        if (max_infer_memory[infer_count] < memo) {
                            max_infer_memory[infer_count] = memo;
                        }
                    }
                    fprintf(out_file_power, "%ld,", memo);

                    for (int j = 0; j < data_info.size(); j++)
                    {
                        fprintf(out_file_power, "%lf", data_info[j]);
                        if (j != data_info.size() - 1)
                        {
                            fprintf(out_file_power, ",");
                        }
                        else
                        {
                            fprintf(out_file_power, "\n");
                        }
                    }

                    i++;
                }
                else
                {
                    if (log_level >= 2)
                        fprintf(out_file_power, "%lf:event:%s\n", event_time, event_log[event_count].second.c_str());

                    if (event_log[event_count].second.find("this inference end.") != std::string::npos) {
                        fprintf(out_file_infer, "%lf,%d,%lf,%ld\n", event_time, last_infer_num, event_time - last_infer_time,peak_memo);
                        last_infer_num = -1;
                        peak_memo = -1;
                        last_infer_time = -1;
                    }

                    if (event_log[event_count].second.find("No.") != std::string::npos)
                    {   
                        int num = 0;
                        sscanf(event_log[event_count].second.c_str(), "No.%d",&num);

                        last_infer_num = num;
                        last_infer_time = event_time;

                        if (infer)
                        {
                            infer_count++;
                            max_infer_memory.push_back(0);
                        }
                        else {
                            infer = true;
                        }
                    }
                    else {
                        infer = false;
                    }
                    event_count++;
                }
            }

            while (event_count < length_event)
            {
                if (log_level >= 2)
                    fprintf(out_file_power, "%lf:event:%s\n", event_log[event_count].first, event_log[event_count].second.c_str());
                event_count++;
            }
        }
        else {
            for (int i = 0; i < length_power; i++)
            {
                fprintf(out_file_power, "%lf:", time);
                std::vector<double> data_info = data_log[i].second;
                long memo = memory_log[i].second;
                fprintf(out_file_power, "%ld,", memo);
                for (int j = 0; j < data_info.size(); j++)
                {
                    fprintf(out_file_power, "%lf", data_info[j]);
                    if (j != data_info.size() - 1)
                    {
                        fprintf(out_file_power, ",");
                    }
                    else
                    {
                        fprintf(out_file_power, "\n");
                    }
                }

                i++;
            }
        }
    }

    fclose(out_file_power);
    fclose(out_file_infer);
}

std::vector<double> PowerLogger::get_total_energy()
{
    std::vector<double> sum_energy(3,0.0);
    double pre_t = 0.0;

    for (int i = 0; i < this->data_log.size(); i++) {
        double t = this->data_log[i].first;
        for (int j = 0; j < this->data_log[i].second.size(); j++) {
            double power = this->data_log[i].second[j];
            sum_energy[j] += (t - pre_t) * power;
        }
        pre_t = t;
    }

    return sum_energy;
}

std::vector<long> PowerLogger::get_max_infer_memory_usage(){
    return this->max_infer_memory;
}

long PowerLogger::get_max_memory_usage()
{
    return this->max_memory_usage;
}