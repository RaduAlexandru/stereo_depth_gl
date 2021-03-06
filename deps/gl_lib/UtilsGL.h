#pragma once

//my stuff
#include "Texture2D.h"
#include "Texture2DArray.h"

//eigen
#include <Eigen/Dense>

//GL
#include <GL/glad.h>

//c++
#include <iostream>

//loguru
#include <loguru.hpp>

//https://blog.noctua-software.com/opencv-opengl-projection-matrix.html
inline Eigen::Matrix4f intrinsics_to_opengl_proj(const Eigen::Matrix3f& K, const int width, const int height, float znear=0.1f, float zfar=100.0f){
    //apllying glscale like here solves the flipping issue https://www.opengl.org/discussion_boards/showthread.php/144492-Flip-entire-framebuffer-upside-down
    //effectivelly we flip m(1,1) and m(2,1)

    Eigen::Matrix4f m;
    float fx=K(0,0);
    float fy=K(1,1);
    float cx=K(0,2);
    float cy=K(1,2);


    //attempt 4 (http://ksimek.github.io/2013/06/03/calibrated_cameras_in_opengl)
    Eigen::Matrix4f persp = Eigen::Matrix4f::Zero();
    persp(0,0) = fx;                      persp(0,2) = -cx;
                        persp(1,1) = fy;  persp(1,2) = -cy;
                                          persp(2,2) = (znear+zfar); persp(2,3) = znear*zfar;
                                          persp(3,2) = -1.0;

     Eigen::Matrix4f ortho = Eigen::Matrix4f::Zero();
     ortho(0,0) =  2.0/width; ortho(0,3) = -1;
     ortho(1,1) =  2.0/height; ortho(1,3) = -1;
     ortho(2,2) = -2.0/(zfar-znear); ortho(2,3) = -(zfar+znear)/(zfar-znear);
     ortho(3,3) =  1.0;

     m = ortho*persp;
     //need t flip the z axis for some reason
     m(0,2)=-m(0,2);
     m(1,2)=-m(1,2);
     m(2,2)=-m(2,2);
     m(3,2)=-m(3,2);

    return m;
}

// OpenGL-error callback function
// Used when GL_ARB_debug_output is supported
inline void APIENTRY debug_func(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length,
	const GLchar* message, GLvoid const* userParam)
{
	std::string srcName;
	switch(source)
	{
	case GL_DEBUG_SOURCE_API_ARB: srcName = "API"; break;
	case GL_DEBUG_SOURCE_WINDOW_SYSTEM_ARB: srcName = "Window System"; break;
	case GL_DEBUG_SOURCE_SHADER_COMPILER_ARB: srcName = "ShaderProgram Compiler"; break;
	case GL_DEBUG_SOURCE_THIRD_PARTY_ARB: srcName = "Third Party"; break;
	case GL_DEBUG_SOURCE_APPLICATION_ARB: srcName = "Application"; break;
	case GL_DEBUG_SOURCE_OTHER_ARB: srcName = "Other"; break;
	}

	std::string errorType;
	switch(type)
	{
	case GL_DEBUG_TYPE_ERROR_ARB: errorType = "Error"; break;
	case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR_ARB: errorType = "Deprecated Functionality"; break;
	case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR_ARB: errorType = "Undefined Behavior"; break;
	case GL_DEBUG_TYPE_PORTABILITY_ARB: errorType = "Portability"; break;
	case GL_DEBUG_TYPE_PERFORMANCE_ARB: errorType = "Performance"; break;
	case GL_DEBUG_TYPE_OTHER_ARB: errorType = "Other"; break;
	}

	std::string typeSeverity;
	switch(severity)
	{
	case GL_DEBUG_SEVERITY_HIGH_ARB: typeSeverity = "High"; break;
	case GL_DEBUG_SEVERITY_MEDIUM_ARB: typeSeverity = "Medium"; break;
	case GL_DEBUG_SEVERITY_LOW_ARB: typeSeverity = "Low"; break;
	}

    int loguru_severity=1;
    if(typeSeverity == "High"){
        loguru_severity=loguru::Verbosity_ERROR;
    }else if (typeSeverity == "Medium"){
        loguru_severity=loguru::Verbosity_WARNING;
    }else if (typeSeverity == "Low"){
        loguru_severity=loguru::Verbosity_4;
    }else{
        loguru_severity=loguru::Verbosity_4;
    }

    VLOG(loguru_severity) << message;

    // if(typeSeverity == "High"){
    //     auto st = loguru::stacktrace(1);
    //     if (!st.empty()) {
    //         RAW_LOG_F(ERROR, "Stack trace:\n%s", st.c_str());
    //     }
    // }

	// printf("%s from %s,\t%s priority\nMessage: %s\n",
	// 	errorType.c_str(), srcName.c_str(), typeSeverity.c_str(), message);
}

inline void print_supported_extensions(){
    GLint ExtensionCount = 0;
	glGetIntegerv(GL_NUM_EXTENSIONS, &ExtensionCount);
	for(GLint i = 0; i < ExtensionCount; ++i){
        std::cout << std::string((char const*)glGetStringi(GL_EXTENSIONS, i)) << '\n';
    }
}
inline void bind_for_sampling(const GLenum texture_target, const GLuint texture_id, const GLint texture_unit, const GLint shader_location){
    //https://www.opengl.org/discussion_boards/showthread.php/174926-when-to-use-glActiveTexture
    GLenum cur_texture_unit = GL_TEXTURE0 + texture_unit;
    glActiveTexture(cur_texture_unit);
    glBindTexture(texture_target, texture_id);
    glUniform1i(shader_location, texture_unit);
}

inline void bind_for_sampling(const gl::Texture2D& texture, const GLint texture_unit, const GLint shader_location){
    //https://www.opengl.org/discussion_boards/showthread.php/174926-when-to-use-glActiveTexture
    GLenum cur_texture_unit = GL_TEXTURE0 + texture_unit;
    glActiveTexture(cur_texture_unit);
    texture.bind();
    glUniform1i(shader_location, texture_unit);
}

inline void bind_for_sampling(const gl::Texture2DArray& texture, const GLint texture_unit, const GLint shader_location){
    //https://www.opengl.org/discussion_boards/showthread.php/174926-when-to-use-glActiveTexture
    GLenum cur_texture_unit = GL_TEXTURE0 + texture_unit;
    glActiveTexture(cur_texture_unit);
    texture.bind();
    glUniform1i(shader_location, texture_unit);
}

inline void CheckOpenGLError(const char* stmt, const char* fname, int line)
{
    GLenum err = glGetError();
    //  const GLubyte* sError = gluErrorString(err);

    if (err != GL_NO_ERROR){
        printf("OpenGL error %08x, at %s:%i - for %s.\n", err, fname, line, stmt);
        exit(1);
    }
}

// GL Check Macro. Will terminate the program if a GL error is detected.
#define GL_C(stmt) do {					\
	stmt;						\
	CheckOpenGLError(#stmt, __FILE__, __LINE__);	\
} while (0)


// //https://stackoverflow.com/a/20545775
// #include <glm/gtx/string_cast.hpp>
// #include <type_traits>
// #include <utility>
// template <typename GLMType, typename = decltype(glm::to_string(std::declval<GLMType>()))>
// inline std::ostream& operator<<(std::ostream& out, const GLMType& g)
// {
//     return out << glm::to_string(g);
// }

auto file_to_string = [](const std::string &filename)->std::string{
    std::ifstream t(filename);
    if (t.is_open()) {
        return std::string((std::istreambuf_iterator<char>(t)),
                std::istreambuf_iterator<char>());
    }else{
        LOG(FATAL) << "Failed to open file " << filename;
        return "";
    }
};
