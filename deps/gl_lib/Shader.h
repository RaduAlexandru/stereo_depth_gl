#include <GL/glad.h>

#include <iostream>
#include <fstream>

namespace gl{

    inline GLuint load_shader( const std::string & src,const GLenum type) {
        if(src.empty()){
            return (GLuint) 0;
        }

        GLuint s = glCreateShader(type);
        if(s == 0){
            fprintf(stderr,"Error: load_shader() failed to create shader.\n");
            return 0;
        }
        // Pass shader source string
        const char *c = src.c_str();
        glShaderSource(s, 1, &c, NULL);
        glCompileShader(s);
        return s;
    }

    inline GLuint program_init( const std::string &vertex_shader_string,
                        const std::string &fragment_shader_string){

        using namespace std;
        GLuint vertex_shader = load_shader(vertex_shader_string,GL_VERTEX_SHADER);
        GLuint fragment_shader = load_shader(fragment_shader_string, GL_FRAGMENT_SHADER);

        if (!vertex_shader || !fragment_shader)
            return false;

        GLuint program_shader = glCreateProgram();

        glAttachShader(program_shader, vertex_shader);
        glAttachShader(program_shader, fragment_shader);

        glLinkProgram(program_shader);

        GLint status;
        glGetProgramiv(program_shader, GL_LINK_STATUS, &status);

        if (status != GL_TRUE){
            char buffer[512];
            glGetProgramInfoLog(program_shader, 512, NULL, buffer);
            cerr << "Linker error: " << endl << buffer << endl;
            program_shader = 0;
            return -1;
        }

        return program_shader;

    }

    inline GLuint program_init_from_files( const std::string &vertex_shader_filename,
                                   const std::string &fragment_shader_filename){



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

    return program_init(file_to_string(vertex_shader_filename),
                       file_to_string(fragment_shader_filename));

    }


    //for compute shaders
    inline GLuint program_init( const std::string &compute_shader_string){

        using namespace std;
        GLuint compute_shader = load_shader(compute_shader_string,GL_COMPUTE_SHADER);

        if (!compute_shader)
            return false;

        GLuint program_shader = glCreateProgram();

        glAttachShader(program_shader, compute_shader);

        glLinkProgram(program_shader);

        GLint status;
        glGetProgramiv(program_shader, GL_LINK_STATUS, &status);

        if (status != GL_TRUE){
            char buffer[512];
            glGetProgramInfoLog(program_shader, 512, NULL, buffer);
            cerr << "Linker error: " << endl << buffer << endl;
            program_shader = 0;
            return -1;
        }

        return program_shader;

    }


    inline GLuint program_init_from_files( const std::string &compute_shader_filename){



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

    return program_init(file_to_string(compute_shader_filename));

    }

    //for program with vert, frag and geom
    inline GLuint program_init( const std::string &vertex_shader_string,
                                const std::string &fragment_shader_string,
                                const std::string &geom_shader_string){

        using namespace std;
        GLuint vertex_shader = load_shader(vertex_shader_string,GL_VERTEX_SHADER);
        GLuint fragment_shader = load_shader(fragment_shader_string, GL_FRAGMENT_SHADER);
        GLuint geom_shader = load_shader(geom_shader_string, GL_GEOMETRY_SHADER);

        if (!vertex_shader || !fragment_shader)
            return false;

        GLuint program_shader = glCreateProgram();

        glAttachShader(program_shader, vertex_shader);
        glAttachShader(program_shader, fragment_shader);
        glAttachShader(program_shader, geom_shader);

        glLinkProgram(program_shader);

        GLint status;
        glGetProgramiv(program_shader, GL_LINK_STATUS, &status);

        if (status != GL_TRUE){
            char buffer[512];
            glGetProgramInfoLog(program_shader, 512, NULL, buffer);
            cerr << "Linker error: " << endl << buffer << endl;
            program_shader = 0;
            return -1;
        }

        return program_shader;

    }

    inline GLuint program_init_from_files( const std::string &vertex_shader_filename,
                                           const std::string &fragment_shader_filename,
                                           const std::string &geom_shader_filename){



        auto file_to_string = [](const std::string &filename)->std::string{
            std::ifstream t(filename);
            return std::string((std::istreambuf_iterator<char>(t)),
                               std::istreambuf_iterator<char>());
        };

        return program_init(file_to_string(vertex_shader_filename),
                            file_to_string(fragment_shader_filename),
                            file_to_string(geom_shader_filename));

    }








}
