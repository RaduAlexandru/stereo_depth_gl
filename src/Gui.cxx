#include "stereo_depth_gl/Gui.h"

//c++
#include <iostream>
#include <unordered_map>
#include <iomanip> // setprecision

//My stuff
#include "stereo_depth_gl/Core.h"
#include "stereo_depth_gl/Profiler.h"
#include "stereo_depth_gl/RosBagPlayer.h"
#include "stereo_depth_gl/MiscUtils.h"
#include "stereo_depth_gl/DataLoaderPNG.h"
// #include "stereo_depth_gl/SurfelSplatter.h"
#include "stereo_depth_gl/DepthEstimatorGL.h"

//imgui
#include "imgui_impl_glfw_gl3.h"
#include "curve.hpp"

//nativefiledialog
#include <nfd.h>

//loguru
//#include <loguru.hpp>

//libigl
#include <igl/opengl/glfw/Viewer.h>

// static int texturer_one_class_id=0;

Gui::Gui(std::shared_ptr<Core> core,
         std::shared_ptr<igl::opengl::glfw::Viewer> view,
         std::shared_ptr<Profiler> profiler) :
        m_show_demo_window(false),
        m_show_profiler_window(true),
        m_show_player_window(true){
    m_core = core;
    m_view = view;
    m_profiler=profiler;

    ImGui::CreateContext();
    ImGui_ImplGlfwGL3_Init(m_view->window, true);

    init_style();

    m_bg_color = ImColor(51, 51, 51, 255);
    m_mesh_color = ImColor(255, 215, 85, 255);
    foo[0].x = -1;
}

void Gui::update() {

    ImVec2 canvas_size = ImGui::GetIO().DisplaySize;

    ImGuiWindowFlags main_window_flags = 0;
    main_window_flags |= ImGuiWindowFlags_NoMove;
    ImGui::SetNextWindowSize(ImVec2(310, canvas_size.y));
    ImGui::SetNextWindowPos(ImVec2(0, 0));
    ImGui::Begin("Laser Mesher", nullptr, main_window_flags);
    ImGui::PushItemWidth(135);


    if (ImGui::CollapsingHeader("Viewer")) {
        //combo of the data list with names for each of them

        if(ImGui::ListBoxHeader("Scene meshes", m_core->m_scene.get_nr_meshes(), 4)){
            for (size_t i = 0; i < m_core->m_scene.get_nr_meshes(); i++) {

                //it's the one we have selected so we change the header color to a whiter value
                if(i==m_view->selected_data_index){
                    ImGui::PushStyleColor(ImGuiCol_Header,ImVec4(0.3f, 0.3f, 0.3f, 1.00f));
                }else{
                    ImGui::PushStyleColor(ImGuiCol_Header, ImGui::GetStyle().Colors[ImGuiCol_Header]);
                }

                //visibility changes the text color from green to red
                if(m_core->m_scene.get_mesh_with_idx(i).m_is_visible){
                    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.1f, 0.7f, 0.1f, 1.00f));  //green text
                }else{
                    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.8f, 0.1f, 0.1f, 1.00f)); //red text
                }




                if(ImGui::Selectable(m_core->m_scene.get_mesh_with_idx(i).name.c_str(), true, ImGuiSelectableFlags_AllowDoubleClick)){ //we leave selected to true so that the header appears and we can change it's colors
                    // m_view->data_list[i].visible=!m_view->data_list[i].visible;
                    if (ImGui::IsMouseDoubleClicked(0)){
                        m_core->m_scene.get_mesh_with_idx(i).m_is_visible=!m_core->m_scene.get_mesh_with_idx(i).m_is_visible;
                        m_core->m_scene.get_mesh_with_idx(i).m_visualization_should_change=true;
                    }
                    m_view->selected_data_index=i;
                }


                ImGui::PopStyleColor(2);
            }
        }
        ImGui::ListBoxFooter();


        if (ImGui::Button("CenterCamera")) {
            m_view->selected_data_index=0;
            m_view->core.align_camera_center(m_view->data().V);
        }
        // TODO not the best way of doing it because changing it supposes recomputing the whole mesh again
        if (ImGui::Checkbox("Show points", &m_core->m_scene.get_mesh_with_idx(m_view->selected_data_index).m_show_points)) {
            m_core->m_scene.get_mesh_with_idx(m_view->selected_data_index).m_visualization_should_change=true;
        }
        if (ImGui::Checkbox("Show mesh", &m_core->m_scene.get_mesh_with_idx(m_view->selected_data_index).m_show_mesh)) {
            m_core->m_scene.get_mesh_with_idx(m_view->selected_data_index).m_visualization_should_change=true;
        }
        if (ImGui::Checkbox("Show edges", &m_core->m_scene.get_mesh_with_idx(m_view->selected_data_index).m_show_edges)) {
            m_core->m_scene.get_mesh_with_idx(m_view->selected_data_index).m_visualization_should_change=true;
        }
        ImGui::SliderFloat("Line_width", &m_view->data().line_width, 0.6f, 5.0f);

        if (ImGui::Combo("Color type", &m_core->m_scene.get_mesh_with_idx(m_view->selected_data_index).m_color_type, m_core->m_color_types_desc, IM_ARRAYSIZE(m_core->m_color_types_desc))){
            m_core->m_scene.get_mesh_with_idx(m_view->selected_data_index).m_visualization_should_change=true;
        }
        if(ImGui::ColorEdit3("Mesh color",m_core->m_scene.get_mesh_with_idx(m_view->selected_data_index).m_mesh_color.data())){
            m_core->m_scene.get_mesh_with_idx(m_view->selected_data_index).m_visualization_should_change=true;
        }

       // //Imgui view.core options
       ImGui::Checkbox("Show texture", &m_view->data().show_texture);
       ImGui::Checkbox("Show faces", &m_view->data().show_faces);
       ImGui::Checkbox("Show lines", &m_view->data().show_lines);
       ImGui::Checkbox("Show vertid", &m_view->data().show_vertid);
       // ImGui::Checkbox("Show pointid", &m_view->data().show_pointid);
       ImGui::Checkbox("Show faceid", &m_view->data().show_faceid);
       ImGui::Checkbox("Invert_normals", &m_view->data().invert_normals);
       ImGui::SliderFloat("shininess", &m_view->data().shininess, 0.001f, 2.0f);
       ImGui::SliderFloat("point_size", &m_view->data().point_size, 1.0, 7.0);
       ImGui::SliderFloat("Line_width", &m_view->data().line_width, 0.6f, 5.0f);


       //global params applied to all meshes
       ImGui::SliderFloat("lighting_factor", &m_view->core.lighting_factor, 0.0f, 2.0f);
       if(ImGui::ColorEdit3("Bg color", (float*)&m_bg_color)){
           m_view->core.background_color << m_bg_color.x , m_bg_color.y, m_bg_color.z;
       }

    }


    ImGui::Separator();




    // if (ImGui::CollapsingHeader("Surfel Splatter")) {
    //     ImGui::PlotLines("Point cloud to mesh distances", m_core->m_splatter->m_nr_points_in_bucket.data(), m_core->m_splatter->m_nr_points_in_bucket.size());
    // }


    if (ImGui::CollapsingHeader("Depth Estimation")) {
        ImGui::Checkbox("Enable gl profiling", &m_core->m_depth_estimator_gl->m_gl_profiling_enabled);
        // if(ImGui::SliderInt("search_epi_method", &m_core->m_depth_estimator_gl->m_params.search_epi_method, 0, 1)){
        //     m_core->m_depth_estimator_gl->compute_depth_and_create_mesh();
        // }
        // if(ImGui::SliderFloat("mean_starting_depth", &m_core->m_depth_estimator_gl->m_mean_starting_depth, 0.5, 6)){
        //     m_core->m_depth_estimator_gl->compute_depth_and_create_mesh();
        // }
        // if(ImGui::SliderFloat("gradH_th", &m_core->m_depth_estimator_gl->m_params.gradH_th, 100000, 800000000)){
        //     m_core->m_depth_estimator_gl->compute_depth_and_create_mesh();
        // }
        // if(ImGui::SliderFloat("outlierTH", &m_core->m_depth_estimator_gl->m_params.outlierTH, 1, 300)){
        //     m_core->m_depth_estimator_gl->compute_depth_and_create_mesh();
        // }
        // if(ImGui::SliderFloat("overallEnergyTHWeight", &m_core->m_depth_estimator_gl->m_params.overallEnergyTHWeight, 0.1, 50)){
        //     m_core->m_depth_estimator_gl->compute_depth_and_create_mesh();
        // }
        // if(ImGui::SliderFloat("outlierTHSumComponent", &m_core->m_depth_estimator_gl->m_params.outlierTHSumComponent, 5, 500)){
        //     m_core->m_depth_estimator_gl->compute_depth_and_create_mesh();
        // }
        // if(ImGui::SliderFloat("huberTH", &m_core->m_depth_estimator_gl->m_params.huberTH, 0.1, 50)){
        //     m_core->m_depth_estimator_gl->compute_depth_and_create_mesh();
        // }
        // if(ImGui::SliderFloat("convergence_sigma2_thresh", &m_core->m_depth_estimator_gl->m_params.convergence_sigma2_thresh, 0.1, 300)){
        //     m_core->m_depth_estimator_gl->compute_depth_and_create_mesh();
        // }
        // if(ImGui::SliderFloat("eta", &m_core->m_depth_estimator_gl->m_params.eta, 0.1, 200)){
        //     m_core->m_depth_estimator_gl->compute_depth_and_create_mesh();
        // }
        //
        // //denoise things
        // if(ImGui::SliderInt("denoise_nr_iterations", &m_core->m_depth_estimator_gl->m_params.denoise_nr_iterations, 0, 400)){
        //     m_core->m_depth_estimator_gl->compute_depth_and_create_mesh();
        // }
        // if(ImGui::SliderFloat("denoise_depth_range", &m_core->m_depth_estimator_gl->m_params.denoise_depth_range, 1.0, 10.0)){
        //     m_core->m_depth_estimator_gl->compute_depth_and_create_mesh();
        // }
        // if(ImGui::SliderFloat("denoise_lambda", &m_core->m_depth_estimator_gl->m_params.denoise_lambda, 0.1, 10)){
        //     m_core->m_depth_estimator_gl->compute_depth_and_create_mesh();
        // }
        // if(ImGui::SliderFloat("denoise_L", &m_core->m_depth_estimator_gl->m_params.denoise_L, 0.1, 20)){
        //     m_core->m_depth_estimator_gl->compute_depth_and_create_mesh();
        // }
        // if(ImGui::SliderFloat("denoise_tau", &m_core->m_depth_estimator_gl->m_params.denoise_tau, 0.00001, 5)){
        //     m_core->m_depth_estimator_gl->compute_depth_and_create_mesh();
        // }
        // if(ImGui::SliderFloat("denoise_theta", &m_core->m_depth_estimator_gl->m_params.denoise_theta, 0.00001, 20)){
        //     m_core->m_depth_estimator_gl->compute_depth_and_create_mesh();
        // }
    }
    show_stereo_textures();
    show_rgb_textures();
    show_debug_textures();
    show_ref_frame_texture();




    ImGui::Separator();
    if (ImGui::CollapsingHeader("Misc")) {
        ImGui::SliderInt("log_level", &loguru::g_stderr_verbosity, -3, 9);
        if (ImGui::Button("Print profiling stats")){
            m_core->m_profiler->print_all_stats();

        }
    }


    ImGui::Separator();
    if (ImGui::CollapsingHeader("IO")) {
        if (ImGui::Button("Read mesh from file")){
            nfdchar_t *path = NULL;
            nfdresult_t result = NFD_OpenDialog( NULL, NULL, &path );
            if ( result == NFD_OKAY ) {
                puts("Success!");
                puts(path);
                Mesh mesh=m_core->read_mesh_from_file(std::string(path));
                m_core->m_scene.add_mesh(mesh,"from_file");
                free(path);
            }
        }

        ImGui::InputText("exported filename", m_core->m_exported_filename, IM_ARRAYSIZE(m_core->m_exported_filename));
        if (ImGui::Button("Write PLY")){
            m_core->write_ply();
        }
        if (ImGui::Button("Write OBJ")){
            m_core->write_obj();
        }
        if (ImGui::Button("Write PCD")){
            m_core->write_pcd();
        }
    }


    ImGui::Separator();
    ImGui::Text(("Nr of points: " + format_with_commas(m_core->m_scene.get_total_nr_vertices())).data());
    ImGui::Text(("Nr of triangles: " + format_with_commas(m_core->m_scene.get_total_nr_vertices())).data());
    ImGui::Text("Average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);




    // static float f = 0.0f;
    // ImGui::Text("Hello, world!");
    // ImGui::SliderFloat("float", &f, 0.0f, 1.0f);
    if (ImGui::Button("Test Window")) m_show_demo_window ^= 1;
    if (ImGui::Button("Profiler Window")) m_show_profiler_window ^= 1;
    if (ImGui::Button("Player Window")) m_show_player_window ^= 1;




    // if (ImGui::Curve("Das editor", ImVec2(400, 200), 10, foo))
    // {
    //   // foo[0].y=ImGui::CurveValue(foo[0].x, 5, foo);
    //   // foo[1].y=ImGui::CurveValue(foo[1].x, 5, foo);
    //   // foo[2].y=ImGui::CurveValue(foo[2].x, 5, foo);
    //   // foo[3].y=ImGui::CurveValue(foo[3].x, 5, foo);
    //   // foo[4].y=ImGui::CurveValue(foo[4].x, 5, foo);
    // }



    ImGui::End();


    if (m_show_profiler_window && m_profiler->m_timings.size()>0 ){
        ImGuiWindowFlags profiler_window_flags = 0;
        profiler_window_flags |= ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar;
        int nr_timings=m_profiler->m_timings.size();
        ImVec2 size(310,50*nr_timings);
        ImGui::SetNextWindowSize(size);
        ImGui::SetNextWindowPos(ImVec2(canvas_size.x -size.x , 0));
        ImGui::Begin("Profiler", nullptr, profiler_window_flags);
        ImGui::PushItemWidth(135);


        for (int i = 0; i < m_profiler->m_ordered_timers.size(); ++i){
            const std::string &name = m_profiler->m_ordered_timers[i];
            auto stats=m_profiler->m_stats[name];
            auto times=m_profiler->m_timings[name];


            // std::stringstream stream_exp;
            // stream_exp << std::fixed << std::setprecision(1) << stats.exp_avg();
            // std::string s_exp = stream_exp.str();
            std::stringstream stream_cma;
            stream_cma << std::fixed <<  std::setprecision(1) << stats.mean;
            std::string s_cma = stream_cma.str();

//        std::string title = times.first +  "\n" + "(" + s_exp + ")" + "(" + s_cma + ")";
            std::string title = name +  "\n" + "avg: " + s_cma + " ms";
            ImGui::PlotLines(title.data(), times.data() , times.size() ,times.get_front_idx() );
        }
        ImGui::End();
    }


    if (m_show_player_window){
        ImGuiWindowFlags player_window_flags = 0;
        player_window_flags |=  ImGuiWindowFlags_NoTitleBar;
        ImVec2 size(135,56);
        ImGui::SetNextWindowSize(size);
        ImGui::SetNextWindowPos(ImVec2(canvas_size.x -size.x , canvas_size.y -size.y ));
        ImGui::Begin("Player", nullptr, player_window_flags);
        ImGui::PushItemWidth(135);


        ImVec2 button_size(25,25);
        const char* icon_play = m_core->m_player_paused ? ICON_FA_PLAY : ICON_FA_PAUSE;
        if(ImGui::Button(icon_play,button_size)){
            m_core->m_player_paused^= 1;
        }
        ImGui::SameLine();
        if(ImGui::Button(ICON_FA_STEP_FORWARD,button_size)){
            m_core->m_player_should_do_one_step=true;
        }
        // ImGui::SameLine();
        // const char* icon_should_continue = m_core->m_player->m_player_should_continue_after_step? ICON_FA_STOP : ICON_FA_FAST_FORWARD;
        // if(ImGui::Button(icon_should_continue,button_size)){
        //     // //if it's paused, then start it
        //     // if (m_core->m_player->is_paused()){
        //     //     m_core->m_player->play();
        //     //     m_core->m_player->m_player_should_continue_after_step =true;
        //     //     m_core->m_player->m_player_should_do_one_step=true;
        //     // }else{
        //     //     m_core->m_player->pause();
        //     //     m_core->m_player->m_player_should_continue_after_step =false;
        //     //     m_core->m_player->m_player_should_do_one_step=false;
        //     // }
        //
        // }
        ImGui::SameLine();
        if(ImGui::Button(ICON_FA_UNDO,button_size)){
            m_core->m_loader_png->reset();
        }



        ImGui::End();
    }




    // 3. Show the ImGui test window. Most of the sample code is in ImGui::ShowTestWindow()
    if (m_show_demo_window) {
        ImGui::SetNextWindowPos(ImVec2(650, 20), ImGuiSetCond_FirstUseEver);
        ImGui::ShowDemoWindow(&m_show_demo_window);
    }


}


void Gui::show_stereo_textures(){
    //show the gray image with the gradients from each camera

    //show camera left
    if(m_core->m_depth_estimator_gl->m_frame_left.get_tex_id()==-1){
        return;
    }
    ImGuiWindowFlags gray_window_flags = 0;
    ImGui::Begin("cam_left_gray_grad", nullptr, gray_window_flags);
    ImGui::Image((ImTextureID)m_core->m_depth_estimator_gl->m_frame_left.get_tex_id(), ImGui::GetContentRegionAvail());
    ImGui::End();

    //show camera right
    if(m_core->m_depth_estimator_gl->m_frame_right.get_tex_id()==-1){
        return;
    }
    ImGui::Begin("cam_right_gray_grad", nullptr, gray_window_flags);
    ImGui::Image((ImTextureID)m_core->m_depth_estimator_gl->m_frame_right.get_tex_id(), ImGui::GetContentRegionAvail());
    ImGui::End();

}

void Gui::show_rgb_textures(){
    //show the rgb image from each camera

    //show camera left
    if(m_core->m_depth_estimator_gl->m_frame_rgb_left.get_tex_id()==-1){
        return;
    }
    ImGuiWindowFlags gray_window_flags = 0;
    ImGui::Begin("cam_left", nullptr, gray_window_flags);
    ImGui::Image((ImTextureID)m_core->m_depth_estimator_gl->m_frame_rgb_left.get_tex_id(), ImGui::GetContentRegionAvail());
    ImGui::End();

    //show camera right
    if(m_core->m_depth_estimator_gl->m_frame_rgb_right.get_tex_id()==-1){
        return;
    }
    ImGui::Begin("cam_right", nullptr, gray_window_flags);
    ImGui::Image((ImTextureID)m_core->m_depth_estimator_gl->m_frame_rgb_right.get_tex_id(), ImGui::GetContentRegionAvail());
    ImGui::End();

}


void Gui::show_debug_textures(){

    ImGuiWindowFlags debug_window_flags = 0;
    if(m_core->m_depth_estimator_gl->m_hessian_pointwise_tex.get_tex_id()!=-1){
        ImGui::Begin("m_hessian_pointwise_tex", nullptr, debug_window_flags);
        ImGui::Image((ImTextureID)m_core->m_depth_estimator_gl->m_hessian_pointwise_tex.get_tex_id(), ImGui::GetContentRegionAvail());
        ImGui::End();
    }
    if(m_core->m_depth_estimator_gl->m_hessian_blurred_tex.get_tex_id()!=-1){
        ImGui::Begin("m_hessian_blurred_tex", nullptr, debug_window_flags);
        ImGui::Image((ImTextureID)m_core->m_depth_estimator_gl->m_hessian_blurred_tex.get_tex_id(), ImGui::GetContentRegionAvail());
        ImGui::End();
    }
    if(m_core->m_depth_estimator_gl->m_high_hessian_tex.get_tex_id()!=-1){
        ImGui::Begin("m_high_hessian_tex", nullptr, debug_window_flags);
        ImGui::Image((ImTextureID)m_core->m_depth_estimator_gl->m_high_hessian_tex.get_tex_id(), ImGui::GetContentRegionAvail());
        ImGui::End();
    }
    if(m_core->m_depth_estimator_gl->m_debug_tex.get_tex_id()!=-1){
        ImGui::Begin("m_debug_tex", nullptr, debug_window_flags);
        ImGui::Image((ImTextureID)m_core->m_depth_estimator_gl->m_debug_tex.get_tex_id(), ImGui::GetContentRegionAvail());
        ImGui::End();
    }


}

void Gui::show_ref_frame_texture(){

    ImGuiWindowFlags debug_window_flags = 0;
    if(m_core->m_depth_estimator_gl->m_ref_frame_tex.get_tex_id()!=-1){
        ImGui::Begin("m_ref_frame_tex", nullptr, debug_window_flags);
        ImGui::Image((ImTextureID)m_core->m_depth_estimator_gl->m_ref_frame_tex.get_tex_id(), ImGui::GetContentRegionAvail());
        ImGui::End();
    }

}







void Gui::init_style() {
    //based on https://www.unknowncheats.me/forum/direct3d/189635-imgui-style-settings.html
    ImGuiStyle *style = &ImGui::GetStyle();

    style->WindowPadding = ImVec2(15, 15);
    style->WindowRounding = 0.0f;
    style->FramePadding = ImVec2(5, 5);
    style->FrameRounding = 4.0f;
    style->ItemSpacing = ImVec2(12, 8);
    style->ItemInnerSpacing = ImVec2(8, 6);
    style->IndentSpacing = 25.0f;
    style->ScrollbarSize = 8.0f;
    style->ScrollbarRounding = 9.0f;
    style->GrabMinSize = 5.0f;
    style->GrabRounding = 3.0f;

    style->Colors[ImGuiCol_Text] = ImVec4(0.80f, 0.80f, 0.83f, 1.00f);
    style->Colors[ImGuiCol_TextDisabled] = ImVec4(0.24f, 0.23f, 0.29f, 1.00f);
    style->Colors[ImGuiCol_WindowBg] = ImVec4(0.06f, 0.05f, 0.07f, 0.85f);
    style->Colors[ImGuiCol_ChildWindowBg] = ImVec4(0.07f, 0.07f, 0.09f, 1.00f);
    style->Colors[ImGuiCol_PopupBg] = ImVec4(0.07f, 0.07f, 0.09f, 1.00f);
    style->Colors[ImGuiCol_Border] = ImVec4(0.80f, 0.80f, 0.83f, 0.0f);
    style->Colors[ImGuiCol_BorderShadow] = ImVec4(0.92f, 0.91f, 0.88f, 0.00f);
    style->Colors[ImGuiCol_FrameBg] = ImVec4(0.10f, 0.09f, 0.12f, 1.00f);
    style->Colors[ImGuiCol_FrameBgHovered] = ImVec4(0.24f, 0.23f, 0.29f, 1.00f);
    style->Colors[ImGuiCol_FrameBgActive] = ImVec4(0.56f, 0.56f, 0.58f, 1.00f);
    style->Colors[ImGuiCol_TitleBg] = ImVec4(0.10f, 0.09f, 0.12f, 1.00f);
    style->Colors[ImGuiCol_TitleBgCollapsed] = ImVec4(0.10f, 0.09f, 0.12f, 1.00f);
    style->Colors[ImGuiCol_TitleBgActive] = ImVec4(0.07f, 0.07f, 0.09f, 1.00f);
    style->Colors[ImGuiCol_MenuBarBg] = ImVec4(0.10f, 0.09f, 0.12f, 1.00f);
    style->Colors[ImGuiCol_ScrollbarBg] = ImVec4(0.10f, 0.09f, 0.12f, 1.00f);
    style->Colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.80f, 0.80f, 0.83f, 0.31f);
    style->Colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.56f, 0.56f, 0.58f, 1.00f);
    style->Colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.06f, 0.05f, 0.07f, 1.00f);
    // style->Colors[ImGuiCol_ComboBg] = ImVec4(0.19f, 0.18f, 0.21f, 1.00f);
    style->Colors[ImGuiCol_CheckMark] = ImVec4(0.80f, 0.80f, 0.83f, 0.31f);
    style->Colors[ImGuiCol_SliderGrab] = ImVec4(0.80f, 0.80f, 0.83f, 0.31f);
    style->Colors[ImGuiCol_SliderGrabActive] = ImVec4(0.06f, 0.05f, 0.07f, 1.00f);
    style->Colors[ImGuiCol_Button] = ImVec4(0.10f, 0.09f, 0.12f, 1.00f);
    style->Colors[ImGuiCol_ButtonHovered] = ImVec4(0.24f, 0.23f, 0.29f, 1.00f);
    style->Colors[ImGuiCol_ButtonActive] = ImVec4(0.56f, 0.56f, 0.58f, 1.00f);
    style->Colors[ImGuiCol_Header] = ImVec4(0.10f, 0.09f, 0.12f, 1.00f);
    style->Colors[ImGuiCol_HeaderHovered] = ImVec4(0.56f, 0.56f, 0.58f, 0.35f);
    style->Colors[ImGuiCol_HeaderActive] = ImVec4(0.06f, 0.05f, 0.07f, 1.00f);
    style->Colors[ImGuiCol_Column] = ImVec4(0.56f, 0.56f, 0.58f, 1.00f);
    style->Colors[ImGuiCol_ColumnHovered] = ImVec4(0.24f, 0.23f, 0.29f, 1.00f);
    style->Colors[ImGuiCol_ColumnActive] = ImVec4(0.56f, 0.56f, 0.58f, 1.00f);
    style->Colors[ImGuiCol_ResizeGrip] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    style->Colors[ImGuiCol_ResizeGripHovered] = ImVec4(0.56f, 0.56f, 0.58f, 1.00f);
    style->Colors[ImGuiCol_ResizeGripActive] = ImVec4(0.06f, 0.05f, 0.07f, 1.00f);
    style->Colors[ImGuiCol_CloseButton] = ImVec4(0.40f, 0.39f, 0.38f, 0.16f);
    style->Colors[ImGuiCol_CloseButtonHovered] = ImVec4(0.40f, 0.39f, 0.38f, 0.39f);
    style->Colors[ImGuiCol_CloseButtonActive] = ImVec4(0.40f, 0.39f, 0.38f, 1.00f);
    style->Colors[ImGuiCol_PlotLines] = ImVec4(0.63f, 0.6f, 0.6f, 0.94f);
    style->Colors[ImGuiCol_PlotLinesHovered] = ImVec4(0.25f, 1.00f, 0.00f, 1.00f);
    style->Colors[ImGuiCol_PlotHistogram] = ImVec4(0.63f, 0.6f, 0.6f, 0.94f);
    style->Colors[ImGuiCol_PlotHistogramHovered] = ImVec4(0.25f, 1.00f, 0.00f, 1.00f);
    style->Colors[ImGuiCol_TextSelectedBg] = ImVec4(0.25f, 1.00f, 0.00f, 0.43f);
    style->Colors[ImGuiCol_ModalWindowDarkening] = ImVec4(1.00f, 0.98f, 0.95f, 0.73f);
}
