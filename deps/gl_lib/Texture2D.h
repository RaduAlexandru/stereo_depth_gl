#pragma once
#include <GL/glad.h>

#include <iostream>

namespace gl{
    class Texture2D{
    public:
        Texture2D():
            m_tex_id(-1),
            m_pbo_id(-1),
            m_tex_storage_initialized(false),
            m_pbo_storage_initialized(false){
            glGenTextures(1,&m_tex_id);
            glGenBuffers(1, &m_pbo_id);
        }

        ~Texture2D(){
            glDeleteTextures(1, &m_tex_id);
            glDeleteBuffers(1, &m_pbo_id);
        }

        void set_wrap_mode(const GLenum wrap_mode){
            glBindTexture(GL_TEXTURE_2D, m_tex_id);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrap_mode);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrap_mode);
        }

        void set_filter_mode(const GLenum filter_mode){
            glBindTexture(GL_TEXTURE_2D, m_tex_id);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filter_mode);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter_mode);
        }
        void upload_data(GLint internal_format, GLsizei width, GLsizei height, GLenum format, GLenum type, const void* data_ptr, int size_bytes){
            // bind the texture and PBO
            glBindTexture(GL_TEXTURE_2D, m_tex_id);
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo_id);

            if(!m_pbo_storage_initialized){
                glBufferData(GL_PIXEL_UNPACK_BUFFER, size_bytes, NULL, GL_STREAM_DRAW); //allocate storage for pbo
                m_pbo_storage_initialized=true;
            }
            if(!m_tex_storage_initialized){
                glTexImage2D(GL_TEXTURE_2D, 0, internal_format,width,height,0,format,type,0); //allocate storage texture
                m_tex_storage_initialized=true;
            }


            //update pbo
            glBufferData(GL_PIXEL_UNPACK_BUFFER, size_bytes, 0, GL_STREAM_DRAW); //orphan the pbo storage
            GLubyte* ptr = (GLubyte*)glMapBuffer(GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY);
            if(ptr){
                memcpy(ptr, data_ptr, size_bytes);
                glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER); // release pointer to mapping buffer
            }

            // copy pixels from PBO to texture object (this returns inmediatelly and lets the GPU perform DMA at a later time)
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, format, type, 0);

            // it is good idea to release PBOs with ID 0 after use.
            // Once bound with 0, all pixel operations behave normal ways.
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        }


        void upload_without_pbo(GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format, GLenum type, const void* data_ptr){
            glBindTexture(GL_TEXTURE_2D, m_tex_id);
            glTexSubImage2D(GL_TEXTURE_2D,level,xoffset,yoffset,width,height,format,type,data_ptr);
        }


        //allocate inmutable texture storage
        void allocate_tex_storage_inmutable(GLenum internal_format,  GLsizei width, GLsizei height){
            std::cout << "we are binding" << '\n';
            glBindTexture(GL_TEXTURE_2D, m_tex_id);
            std::cout << "m_tex_id " << m_tex_id << '\n';
            std::cout << "we are allocating widthxheight" << width << " " << height << " with format " <<  internal_format << '\n';
            glTexStorage2D(GL_TEXTURE_2D, 1, internal_format, width, height);
            m_tex_storage_initialized=true;
        }

        //uploads data from cpu to pbo (on gpu) will take some cpu time to perform the memcpy
        void upload_to_pbo(const void* data_ptr, int size_bytes){
            // bind the PBO
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo_id);


            //allocate storage for pbo
            if(!m_pbo_storage_initialized){
                glBufferData(GL_PIXEL_UNPACK_BUFFER, size_bytes, NULL, GL_STREAM_DRAW); //allocate storage for pbo
                m_pbo_storage_initialized=true;
            }


            //update pbo
            glBufferData(GL_PIXEL_UNPACK_BUFFER, size_bytes, 0, GL_STREAM_DRAW); //orphan the pbo storage
            GLubyte* ptr = (GLubyte*)glMapBuffer(GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY);
            if(ptr){
                memcpy(ptr, data_ptr, size_bytes);
                glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER); // release pointer to mapping buffer
            }

            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        }

        //uploads the data of the pbo (on the gpu already) to the texture (also on the gpu). Should return inmediatelly
        void upload_pbo_to_tex( GLsizei width, GLsizei height,
                                GLenum format, GLenum type){
            // bind the PBO and texture
            glBindTexture(GL_TEXTURE_2D, m_tex_id);
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo_id);

            // copy pixels from PBO to texture object (this returns inmediatelly and lets the GPU perform DMA at a later time)
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, format, type, 0);


            // it is good idea to release PBOs with ID 0 after use.
            // Once bound with 0, all pixel operations behave normal ways.
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        }

        //uploads the data of the pbo (on the gpu already) to the texture (also on the gpu). Should return inmediatelly
        void upload_pbo_to_tex_no_binds( GLsizei width, GLsizei height,
                                         GLenum format, GLenum type){

            // copy pixels from PBO to texture object (this returns inmediatelly and lets the GPU perform DMA at a later time)
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, format, type, 0);

        }


        void bind() const{
            glBindTexture(GL_TEXTURE_2D, m_tex_id);
        }

        int get_tex_id() const{
            return m_tex_id;
        }

        bool get_tex_storage_initialized (){
            return m_tex_storage_initialized;
        }

    private:
        GLuint m_tex_id;
        GLuint m_pbo_id;
        bool m_tex_storage_initialized;
        bool m_pbo_storage_initialized;


    };
}
