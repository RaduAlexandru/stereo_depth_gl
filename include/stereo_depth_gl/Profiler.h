#pragma once

#include <iostream>
#include <unordered_map>
#include <thread>
#include <chrono>
#include <atomic>
#include <string>
#include <vector>

#include "stereo_depth_gl/scope_exit.h"
#include "stereo_depth_gl/ringbuffer.h"


struct Stats{ //for each timer we store some stats so we can compute the avg, min, max an std-dev https://dsp.stackexchange.com/a/1187
    int nr_samples=0;
    float min=std::numeric_limits<float>::max(); //minimum time taken for that timer
    float max=std::numeric_limits<float>::min(); //maximum time taken for that timer
    float mean=0;
    float variance=0;
    float std_dev=0; //not really necesarry because we have the variance but it's nice to have
};


class Timer{
public:
    void start(){
        m_StartTime = std::chrono::high_resolution_clock::now();
        m_running = true;
    }

    bool stop(){
        //only stop if it was running otherwise it was already stopped before
        if (m_running){
            m_EndTime = std::chrono::high_resolution_clock::now();
            m_running = false;
            return true;
        }else{
            return false; //returns that it failed to stop the timer because it was already stopped before
        }
    }

    double elapsed_ms(){
        std::chrono::time_point<std::chrono::high_resolution_clock> endTime;

        if(m_running){
            endTime = std::chrono::high_resolution_clock::now();
        }else{
            endTime = m_EndTime;
        }

        return std::chrono::duration_cast<std::chrono::milliseconds>(endTime - m_StartTime).count();
    }

    double elapsed_s(){
        return elapsed_ms() / 1000.0;
    }


private:
    std::chrono::time_point<std::chrono::high_resolution_clock> m_StartTime;
    std::chrono::time_point<std::chrono::high_resolution_clock> m_EndTime;
    bool m_running = false;
};



class Profiler{
public:
    Profiler(){};
    void start_time( std::string name ){
        // std::thread::id thread_id= std::this_thread::get_id();
        // std::stringstream ss;
        // ss << thread_id;
        // std::string thread_id_string = ss.str();
        // std::string full_name=name+ thread_id_string;
        std::string full_name=name;


        m_timers[full_name].start();

    }

    void stop_time(std::string name){
        // std::thread::id thread_id= std::this_thread::get_id();
        // std::stringstream ss;
        // ss << thread_id;
        // std::string thread_id_string = ss.str();
        // std::string full_name=name+ thread_id_string;
        std::string full_name=name;

        //it's the first time we stopped this timer and doesn't have any recordings yet
        if(m_timings[full_name].empty()){
            m_ordered_timers.push_back(full_name);

            //the first time we time a functions it's usualy a warm up so the maximum can be quite big. We ignore this one so as to not squeue our stats
            m_timers[full_name].stop();
            double time_elapsed=m_timers[full_name].elapsed_ms();
            m_timings[full_name].push(time_elapsed);
            return; //don't store any stats, because this is the first time we time this function so it's likely to be bad
        }

        //get elapsed time for that timer and register it into the timings
        if(m_timers[full_name].stop()){  //we manage to stop is correctly and it was no stopped before
            double time_elapsed=m_timers[full_name].elapsed_ms();
            m_timings[full_name].push(time_elapsed);

            //we also store some evaluation things to be able to easily calculate min, max and std-dev
            //https://dsp.stackexchange.com/a/1187
            Stats& stats=m_stats[full_name];
            stats.nr_samples+=1;
            float prev_mean=stats.mean;
            stats.mean= stats.mean + (time_elapsed-stats.mean)/stats.nr_samples;
            stats.variance=stats.variance+ (time_elapsed - stats.mean)*(time_elapsed-prev_mean);
            stats.std_dev=std::sqrt(stats.variance);
            if(time_elapsed < stats.min){
                stats.min=time_elapsed;
            }
            if(time_elapsed > stats.max){
                stats.max=time_elapsed;
            }
        }




    }

    std::unordered_map<std::string, ringbuffer<float,100> > m_timings;  //contains the last N timings of the registers, for plotting in gui
    std::unordered_map<std::string, Timer> m_timers;  //contains the timers for the registers
    std::vector<std::string> m_ordered_timers;  //contains the timers in the order of insertion, useful for when we plot all of them, they will show in the order we inserted them
    std::unordered_map<std::string, Stats > m_stats;
private:


};



#define TIME_SCOPE_2(name, profiler)\
    profiler->start_time(name); \
    SCOPE_EXIT{profiler->stop_time(name);};

#define TIME_START_2(name,profiler) \
    profiler->start_time(name);

#define TIME_END_2(name,profiler) \
    profiler->stop_time(name);
