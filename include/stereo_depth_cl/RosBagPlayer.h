#pragma once

//c++
#include <iosfwd>
#include <sstream>

//My stuff
#include "tiny-process-library/process.hpp"


class RosBagPlayer{
public:
    RosBagPlayer();
    ~RosBagPlayer();
    RosBagPlayer(std::string args);
    void play(std::string args);
    void pause();
    bool is_paused();
    void kill();

    bool m_player_should_do_one_step;
    bool m_player_should_continue_after_step;

private:
    TinyProcessLib::Process m_rosbag;
    bool m_paused;

};
