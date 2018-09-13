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

    float S=0; //used to calculate the variance and std_dev as explained here https://dsp.stackexchange.com/a/1187
};


class Timer{
public:
    //https://stackoverflow.com/a/40136853
    using precision = long double;
    using ratio = std::milli;
    using clock_t=std::chrono::high_resolution_clock;
    using duration_t = std::chrono::duration<precision, ratio>;
    using timepoint_t = std::chrono::time_point<clock_t, duration_t>;


    void start(){
        m_start_time = clock_t::now();
        m_running = true;
    }

    bool stop(){
        //only stop if it was running otherwise it was already stopped before
        if (m_running){
            m_end_time = clock_t::now() +m_duration_other_sections;
            m_duration_other_sections=duration_t::zero();
            m_running = false;
            return true;
        }else{
            return false; //returns that it failed to stop the timer because it was already stopped before
        }
    }

    bool pause(){
        if(stop()){ //if we managed to stop it correctly and it wasn't stopped before
            //if its running we stop the timer and save the time it took until now so we can sum it up the last time we do TIME_END
            m_duration_other_sections  += std::chrono::high_resolution_clock::now()-m_start_time;
        }
    }

    double elapsed_ms(){
        timepoint_t endTime;

        if(m_running){
            endTime = clock_t::now();
        }else{
            endTime = m_end_time;
        }

        return std::chrono::duration_cast<std::chrono::milliseconds>(endTime - m_start_time).count();
    }

    double elapsed_s(){
        return elapsed_ms() / 1000.0;
    }


private:
    timepoint_t m_start_time;
    timepoint_t m_end_time;
    duration_t m_duration_other_sections=duration_t::zero(); //each time you do a pause we accumulate the time it took for that section of start-pause. This will get summed up at the end for the last start-end
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
            stats.S=stats.S+ (time_elapsed - stats.mean)*(time_elapsed-prev_mean);
            if(stats.nr_samples>1){ //to avoid division by zero
                stats.std_dev=std::sqrt( stats.S/ (stats.nr_samples-1) );
                stats.variance=stats.S/ (stats.nr_samples-1);
            }
            if(time_elapsed < stats.min){
                stats.min=time_elapsed;
            }
            if(time_elapsed > stats.max){
                stats.max=time_elapsed;
            }

        }

    }


    void pause_time( std::string name ){
        // std::thread::id thread_id= std::this_thread::get_id();
        // std::stringstream ss;
        // ss << thread_id;
        // std::string thread_id_string = ss.str();
        // std::string full_name=name+ thread_id_string;
        std::string full_name=name;

        m_timers[full_name].pause();

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

//when you have to sections that are disjoin but you want to get the time it take for both, you cdo start-pause start-end
#define TIME_PAUSE_2(name,profiler) \
    profiler->pause_time(name);
