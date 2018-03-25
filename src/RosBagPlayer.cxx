#include "stereo_depth_cl/RosBagPlayer.h"

//c++
#include<iostream>


RosBagPlayer::RosBagPlayer():
        m_rosbag("rosbag play"),
        m_player_should_do_one_step(false),
        m_player_should_continue_after_step(false){
}
RosBagPlayer::~RosBagPlayer(){
    m_rosbag.kill(true);
}

void RosBagPlayer::play(std::string args){
    //Check for pause in the arguments
    m_paused= args.find("pause") != std::string::npos ? true : false;

    m_rosbag.add_arguments(args);
    m_rosbag.run();
}

void RosBagPlayer::pause(){
    m_rosbag.write(" ");
    m_paused=!m_paused;
}

bool RosBagPlayer::is_paused(){
    return m_paused;
}

void RosBagPlayer::kill(){
    std::cout << "stopping rosbag" << "\n";
    m_rosbag.kill(true);
}
