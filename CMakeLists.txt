cmake_minimum_required(VERSION 2.8.3)
project(stereo_depth_gl)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
list(APPEND CMAKE_MODULE_PATH /media/alex/Data/Programs_linux/libigl/shared/cmake/)
set(PROJECT_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include
                        ${CMAKE_SOURCE_DIR}/deps
                        ${CMAKE_SOURCE_DIR}/deps/imgui
                        ${CMAKE_SOURCE_DIR}/deps/imgui_ext
                        ${CMAKE_SOURCE_DIR}/deps/imgui/examples/opengl3_example
                        ${CMAKE_SOURCE_DIR}/deps/loguru
                        ${CMAKE_SOURCE_DIR}/deps/tiny-process-library
                        ${CMAKE_SOURCE_DIR}/deps/octree
                        ${CMAKE_SOURCE_DIR}/deps/dir_watcher
                        ${CMAKE_SOURCE_DIR}/deps/gl_lib
                        ${CMAKE_SOURCE_DIR}/deps/cv_lib)		# Header folder
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11 -std=gnu++11 -fext-numeric-literals")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w") #supress all warnings
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DAPC_CAPTURE_DEBUG=1")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -O2") #optimizations


# libigl options: choose between header only and compiled static library
# Header-only is preferred for small projects. For larger projects the static build
# considerably reduces the compilation times
option(LIBIGL_USE_STATIC_LIBRARY "Use libigl as static library" OFF)
option(LIBIGL_WITH_ANTTWEAKBAR       "Use AntTweakBar"    OFF)
option(LIBIGL_WITH_CGAL              "Use CGAL"           OFF)
option(LIBIGL_WITH_COMISO            "Use CoMiso"         OFF)
option(LIBIGL_WITH_CORK              "Use Cork"           OFF)
option(LIBIGL_WITH_EMBREE            "Use Embree"         OFF)
option(LIBIGL_WITH_LIM               "Use LIM"            OFF)
option(LIBIGL_WITH_MATLAB            "Use Matlab"         OFF)
option(LIBIGL_WITH_MOSEK             "Use MOSEK"          OFF)
option(LIBIGL_WITH_OPENGL            "Use OpenGL"         ON)
option(LIBIGL_WITH_OPENGL_GLFW       "Use GLFW"           ON)
option(LIBIGL_WITH_OPENGL_GLFW_IMGUI "Use ImGui"          OFF)
option(LIBIGL_WITH_PNG               "Use PNG"            OFF)
option(LIBIGL_WITH_PYTHON            "Use Python"         OFF)
option(LIBIGL_WITH_TETGEN            "Use Tetgen"         OFF)
option(LIBIGL_WITH_TRIANGLE          "Use Triangle"       OFF)
option(LIBIGL_WITH_VIEWER            "Use OpenGL viewer"  ON)
option(LIBIGL_WITH_XML "Use XML" OFF)
if(LIBIGL_WITH_CGAL) # Do not remove or move this block, the cgal build system fails without it
  find_package(CGAL REQUIRED COMPONENTS Core)
  set(CGAL_DONT_OVERRIDE_CMAKE_FLAGS TRUE CACHE BOOL "CGAL's CMAKE Setup is super annoying ")
  include(${CGAL_USE_FILE})
endif()

find_package(LIBIGL REQUIRED)
find_package(catkin REQUIRED)
find_package(OpenMP REQUIRED)
#find_package(OpenGL REQUIRED)
find_package(GLM REQUIRED)
# find_package(nanoflann)
# find_package(Boost 1.64 REQUIRED COMPONENTS system thread filesystem)
# find_package(kqueue REQUIRED)
find_package(Ceres REQUIRED)
find_package(Boost REQUIRED COMPONENTS system thread filesystem)
find_package(kqueue REQUIRED)
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  message_generation
  image_transport
  tf
  tf_conversions
)

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# fix for https://github.com/ros-perception/vision_opencv/issues/193
# find_package(OpenCV 3 REQUIRED)
find_package(OpenCV REQUIRED)
# if (${OpenCV_VERSION} MATCHES "3.3.1")
#   foreach(__cvcomponent ${OpenCV_LIB_COMPONENTS})
#     set (__original_cvcomponent ${__cvcomponent})
#     if(NOT __cvcomponent MATCHES "^opencv_")
#       set(__cvcomponent opencv_${__cvcomponent})
#     endif()
#     if (TARGET ${__cvcomponent})
#       set_target_properties(${__cvcomponent} PROPERTIES
#           MAP_IMPORTED_CONFIG_DEBUG ""
#           MAP_IMPORTED_CONFIG_RELEASE ""
#           MAP_IMPORTED_CONFIG_RELWITHDEBINFO ""
#           MAP_IMPORTED_CONFIG_MINSIZEREL ""
#       )
#     endif()
#   endforeach(__cvcomponent)
# endif()



include_directories(${PROJECT_INCLUDE_DIR} ${catkin_INCLUDE_DIRS})
# include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})
# include_directories("/media/alex/Data/Programs_linux/boost/boost_1_64_0")
# include_directories(${GLFW_INCLUDE_DIR})
#include_directories(${OPENGL_INCLUDE_DIR})
include_directories(${GLM_INCLUDE_DIRS})
# include_directories(${nanoflann_INCLUDE_DIRS})
# include_directories(${KQUEUE_INCLUDE_DIRS})
include_directories(${CERES_INCLUDE_DIR})
#add_definitions(${GLM_DEFINITIONS})
#include_directories("/media/alex/Data/Master/SHK/c_ws/src/laser_mesher/deps/libigl/external/nanogui/ext/glew/include/")


catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS
    roscpp
    geometry_msgs
    std_msgs
    message_generation
  	image_transport
    pcl_ros
    pcl_conversions
    eigen_conversions
    tf
    tf_conversions
)




file(GLOB IMGUI_SRC ${CMAKE_SOURCE_DIR}/deps/imgui/*.c* ${CMAKE_SOURCE_DIR}/deps/imgui/examples/opengl3_example/imgui_impl_glfw_gl3.cpp)
file(GLOB MY_SRC ${CMAKE_SOURCE_DIR}/src/*.c* )


#TINY PROCESS LIBRARY
set (TINY_PROCESS_LIB_SOURCES   ${CMAKE_SOURCE_DIR}/deps/tiny-process-library/process.cpp
                                ${CMAKE_SOURCE_DIR}/deps/tiny-process-library/process_unix.cpp )
add_library( TINY_PROCESS_LIB ${TINY_PROCESS_LIB_SOURCES} )
target_link_libraries(TINY_PROCESS_LIB ${CMAKE_THREAD_LIBS_INIT})
set_target_properties(TINY_PROCESS_LIB PROPERTIES LINKER_LANGUAGE C)


# #Boost 1.64 libraries
# file(GLOB Boost_libs_164 /media/alex/Data/Programs_linux/boost/boost_1_64_0/*.so /media/alex/Data/Programs_linux/boost/boost_1_64_0/*.a)


add_executable(stereo_depth_gl ${MY_SRC}  deps/GL/glad.c deps/dir_watcher/dir_watcher.cpp  ${IMGUI_SRC}   )
target_link_libraries(stereo_depth_gl ${catkin_LIBRARIES}  ${OpenCV_LIBS}  TINY_PROCESS_LIB
          ${KQUEUE_LIBRARIES}
          igl::core igl::opengl igl::opengl_glfw
          ceres
)
#definitions for cmake variables that are necesarry during runtime
target_compile_definitions(stereo_depth_gl PRIVATE AWESOMEFONT_DIR="${CMAKE_SOURCE_DIR}/deps/imgui_ext/fontawesome-webfont.ttf")
target_compile_definitions(stereo_depth_gl PRIVATE -DTRILIBRARY -DANSI_DECLARATORS)
