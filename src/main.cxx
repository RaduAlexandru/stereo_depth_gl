#include <iostream>
//#include <memory>
//#include <chrono>
//#include <thread>

//GL
#include <GL/glad.h>
#include <GLFW/glfw3.h>


// //libigl
#ifdef WITH_VIEWER
    //set this to supress libigl viewer help
    //#define IGL_VIEWER_VIEWER_QUIET
    #include <igl/opengl/glfw/Viewer.h>

    //ImGui
    #include <imgui.h>
    #include "imgui_impl_glfw_gl3.h"

    //My stuff
    #include "stereo_depth_gl/Gui.h"
#endif

//ros
#include <ros/ros.h>
#include "stereo_depth_gl/RosTools.h"


//loguru
#define LOGURU_IMPLEMENTATION 1
#define LOGURU_NO_DATE_TIME 1
#define LOGURU_NO_UPTIME 1
#define LOGURU_REPLACE_GLOG 1
#include <loguru.hpp>

//configuru
#define CONFIGURU_IMPLEMENTATION 1
#define CONFIGURU_WITH_EIGEN 1
#define CONFIGURU_IMPLICIT_CONVERSIONS 1
#include <configuru.hpp>
using namespace configuru;


#include "stereo_depth_gl/Core.h"
#include "stereo_depth_gl/Profiler.h"

#ifdef WITH_VIEWER
    void switch_callbacks(
            std::shared_ptr<igl::opengl::glfw::Viewer> view); //Need to switch the callbacks so that the input goes to either libigl or imgui
#else
    void init_offscreen_context();
#endif

int main(int argc, char *argv[]) {


    loguru::init(argc, argv);
    loguru::g_stderr_verbosity = -1; //By default don't show any logs except warning, error and fatals

    ros::init(argc, argv, "stereo_depth_gl");
    ros::start(); //in order for the node to live otherwise it will die when the last node handle is killed

    LOG_S(INFO) << "Hello from main!";

    //Objects

    #ifdef WITH_VIEWER //we intializ the viewer which creates the context, otherwise we make an ofscreen window
        std::shared_ptr<igl::opengl::glfw::Viewer> view(new igl::opengl::glfw::Viewer);
        view->launch_init();  //creates the actual opengl window so that imgui can attach to it
        //Eyecandy options
        view->core.background_color << 0.2, 0.2, 0.2, 1.0;
        view->data().show_lines = false; //start with the mesh not showing wirefrae
    #else
        init_offscreen_context();
    #endif
    std::shared_ptr<Profiler> profiler(new Profiler());
    std::shared_ptr<Core> core(new Core());
    core->m_profiler=profiler;
    #ifdef WITH_VIEWER
        core->m_view=view;
        std::shared_ptr<Gui> gui(new Gui(core, view,profiler));
        gui->init_fonts(); //needs to be initialized inside the context
    #endif

    core->start(); //now that every object is linked correctly we can start




    //before starting with the main loop we check that none the parameters in the config file was left unchecked
    ros::NodeHandle private_nh("~");
    std::string config_file= getParamElseThrow<std::string>(private_nh, "config_file");
    Config cfg = configuru::parse_file(std::string(CMAKE_SOURCE_DIR)+"/config/"+config_file, CFG);
    cfg.check_dangling(); // Make sure we haven't forgot reading a key!

    while (ros::ok()) {
        // std::this_thread::sleep_for(std::chrono::milliseconds(1));

        // LOG_SCOPE(INFO,"main_loop");
        #ifdef WITH_VIEWER
            glfwPollEvents();
            ImGui_ImplGlfwGL3_NewFrame();
            switch_callbacks(view);  //needs to be done inside the context otherwise it segments faults
            gui->update();
        #endif


        core->update();


        #ifdef WITH_VIEWER
            if (core->m_viewer_initialized) {
               view->draw();
            } else {
               view->core.clear_framebuffers(); //if there is no mesh to draw then just put the color of the background
            }

            ImGui::Render();
            ImGui_ImplGlfwGL3_RenderDrawData(ImGui::GetDrawData());
            glfwSwapBuffers(view->window);
        #endif

    }




    // Cleanup
    #ifdef WITH_VIEWER
        view->launch_shut();
        ImGui_ImplGlfwGL3_Shutdown();
        ImGui::DestroyContext();
    #endif
    glfwTerminate();

    return 0;
}

#ifdef WITH_VIEWER
    void switch_callbacks(std::shared_ptr<igl::opengl::glfw::Viewer> view) {
        bool hovered_imgui = ImGui::IsMouseHoveringAnyWindow();
        if (hovered_imgui) {
            glfwSetMouseButtonCallback(view->window, ImGui_ImplGlfw_MouseButtonCallback);
            glfwSetScrollCallback(view->window, ImGui_ImplGlfw_ScrollCallback);
            glfwSetKeyCallback(view->window, ImGui_ImplGlfw_KeyCallback);
            glfwSetCharCallback(view->window, ImGui_ImplGlfw_CharCallback);
            glfwSetCharModsCallback(view->window, nullptr);
        } else {
            glfwSetMouseButtonCallback(view->window, glfw_mouse_press);
            glfwSetScrollCallback(view->window, glfw_mouse_scroll);
            glfwSetKeyCallback(view->window, glfw_key_callback);
            glfwSetCharModsCallback(view->window, glfw_char_mods_callback);
            glfwSetCharCallback(view->window, nullptr);
        }
    }
#else
    void init_offscreen_context(){
            /* Initialize the library */
        if (!glfwInit()){
            LOG(FATAL) << "init_context glfwInit failure";
        }
        // glfwWindowHint(GLFW_SAMPLES, 8);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 4);
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
        /* Create a ofscreen context and its OpenGL context */
        GLFWwindow* offscreen_context = glfwCreateWindow(640, 480, "", NULL, NULL);
        if (!offscreen_context) {
            glfwTerminate();
            LOG(FATAL) << "Failed to create the window or the GL context";
        }
        /* Make the window's context current */
        glfwMakeContextCurrent(offscreen_context);
        if (!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress))
        {
          LOG(FATAL) << ("Failed to load OpenGL and its extensions");
        }
    }

#endif
