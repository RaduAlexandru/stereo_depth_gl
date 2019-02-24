#pragma once

#include <iostream>
#include <vector>
#include <time.h>
#include <fstream>

#include <Eigen/Dense>

#include <opencv2/highgui/highgui.hpp>

// //loguru
#define LOGURU_NO_DATE_TIME 1
#define LOGURU_NO_UPTIME 1
#define LOGURU_REPLACE_GLOG 1
#include <loguru.hpp>

#include <configuru.hpp>


typedef std::vector<float> row_type_f;
typedef std::vector<row_type_f> matrix_type_f;

typedef std::vector<double> row_type_d;
typedef std::vector<row_type_f> matrix_type_d;

typedef std::vector<int> row_type_i;
typedef std::vector<row_type_i> matrix_type_i;

typedef std::vector<bool> row_type_b;
typedef std::vector<row_type_b> matrix_type_b;

// Converts degrees to radians.
#define degreesToRadians(angleDegrees) (angleDegrees * M_PI / 180.0)

// Converts radians to degrees.
#define radiansToDegrees(angleRadians) (angleRadians * 180.0 / M_PI)

//Best answer of https://stackoverflow.com/questions/5731863/mapping-a-numeric-range-onto-another
inline float lerp(float input, float input_start, float input_end, float output_start, float output_end) {
    return output_start + ((output_end - output_start) / (input_end - input_start)) * (input - input_start);
}

//Adapted from https://stackoverflow.com/questions/7706339/grayscale-to-red-green-blue-matlab-jet-color-scale
//given a certain value from a range between vmin and vmax we output the corresponding jet colormap color
inline std::vector<float> jet_color(double v, double vmin, double vmax) {
    std::vector<float> c = {1.0, 1.0, 1.0}; // white
    double dv;

    if (v < vmin)
        v = vmin;
    if (v > vmax)
        v = vmax;
    dv = vmax - vmin;

    if (v < (vmin + 0.25 * dv)) {
        c[0] = 0;
        c[1] = 4 * (v - vmin) / dv;
    } else if (v < (vmin + 0.5 * dv)) {
        c[0] = 0;
        c[2] = 1 + 4 * (vmin + 0.25 * dv - v) / dv;
    } else if (v < (vmin + 0.75 * dv)) {
        c[0] = 4 * (v - vmin - 0.5 * dv) / dv;
        c[2] = 0;
    } else {
        c[1] = 1 + 4 * (vmin + 0.75 * dv - v) / dv;
        c[2] = 0;
    }

    return (c);
}

//https://stackoverflow.com/questions/7276826/c-format-number-with-commas
template<class T>
std::string format_with_commas(T value) {
    std::stringstream ss;
    ss.imbue(std::locale(""));
    ss << std::fixed << value;
    return ss.str();
}

//returns a random float in the range between a and b
inline float rand_float(float a, float b) {
    float random = ((float) rand()) / (float) RAND_MAX;
    float diff = b - a;
    float r = random * diff;
    return a + r;
}

inline Eigen::Vector3d random_color() {
    Eigen::Vector3d color;
    color(0) = rand_float(0.0, 1.0);
    color(1) = rand_float(0.0, 1.0);
    color(2) = rand_float(0.0, 1.0);
    return color;
}

inline void removeRow(Eigen::MatrixXd& matrix, unsigned int rowToRemove)
{
    unsigned int numRows = matrix.rows()-1;
    unsigned int numCols = matrix.cols();

    if( rowToRemove < numRows )
        matrix.block(rowToRemove,0,numRows-rowToRemove,numCols) = matrix.block(rowToRemove+1,0,numRows-rowToRemove,numCols);

    matrix.conservativeResize(numRows,numCols);
}

inline void removeColumn(Eigen::MatrixXd& matrix, unsigned int colToRemove)
{
    unsigned int numRows = matrix.rows();
    unsigned int numCols = matrix.cols()-1;

    if( colToRemove < numCols )
        matrix.block(0,colToRemove,numRows,numCols-colToRemove) = matrix.block(0,colToRemove+1,numRows,numCols-colToRemove);

    matrix.conservativeResize(numRows,numCols);
}

inline void EigenToFile(Eigen::MatrixXd& src, std::string pathAndName)
{
      std::ofstream fichier(pathAndName);
      if(fichier.is_open())  // si l'ouverture a réussi
      {
        // instructions
        fichier << src << "\n";
        fichier.close();  // on referme le fichier
      }
      else  // sinon
      {
        std::cerr << "Erreur à l'ouverture !" << std::endl;
      }
 }


//Needed because % is not actually modulo in c++ and it may yield unexpected valued for negative numbers
//https://stackoverflow.com/questions/12276675/modulus-with-negative-numbers-in-c
inline int mod(int k, int n) {
    return ((k %= n) < 0) ? k+n : k;
}

inline Eigen::MatrixXd vec2eigen( const std::vector<Eigen::VectorXd>& std_vec ){
    //TODO not the most efficient way, can be done using map but that may break alignment: https://stackoverflow.com/questions/40852757/c-how-to-convert-stdvector-to-eigenmatrixxd

    if(std_vec.size()==0){
        Eigen::MatrixXd eigen_vec;
        return eigen_vec;
    }

    int dim=std_vec[0].size();
    Eigen::MatrixXd eigen_vec(std_vec.size(),dim);
    for (int i = 0; i < std_vec.size(); ++i) {
        eigen_vec.row(i)=std_vec[i];
    }
    return eigen_vec;
}

inline Eigen::MatrixXi vec2eigen( const std::vector<Eigen::VectorXi>& std_vec, bool debug=false){
    //TODO not the most efficient way, can be done using map but that may break alignment: https://stackoverflow.com/questions/40852757/c-how-to-convert-stdvector-to-eigenmatrixxd

    if(std_vec.size()==0){
        Eigen::MatrixXi eigen_vec;
        return eigen_vec;
    }

    int dim=std_vec[0].size();
    Eigen::MatrixXi eigen_vec(std_vec.size(),dim);
    for (int i = 0; i < std_vec.size(); ++i) {
        eigen_vec.row(i)=std_vec[i];
    }
    return eigen_vec;
}

inline Eigen::MatrixXi vec2eigen( const std::vector<bool>& std_vec ){
    //TODO not the most efficient way, can be done using map but that may break alignment: https://stackoverflow.com/questions/40852757/c-how-to-convert-stdvector-to-eigenmatrixxd
    Eigen::MatrixXi eigen_vec (std_vec.size(),1);
    for (int i = 0; i < std_vec.size(); ++i) {
        eigen_vec(i)= std_vec[i] ? 1 : 0;
    }
    return eigen_vec;
}

inline Eigen::MatrixXd vec2eigen( const std::vector<double>& std_vec ){
    //TODO not the most efficient way, can be done using map but that may break alignment: https://stackoverflow.com/questions/40852757/c-how-to-convert-stdvector-to-eigenmatrixxd
    Eigen::MatrixXd eigen_vec(std_vec.size(),1);
    for (int i = 0; i < std_vec.size(); ++i) {
        eigen_vec(i)=std_vec[i];
    }
    return eigen_vec;
}

inline Eigen::MatrixXd vec2eigen( const std::vector<Eigen::Vector3d>& std_vec ){
    //TODO not the most efficient way, can be done using map but that may break alignment: https://stackoverflow.com/questions/40852757/c-how-to-convert-stdvector-to-eigenmatrixxd

    if(std_vec.size()==0){
        Eigen::MatrixXd eigen_vec;
        return eigen_vec;
    }

    Eigen::MatrixXd eigen_vec(std_vec.size(),3);
    for (int i = 0; i < std_vec.size(); ++i) {
        eigen_vec.row(i)=std_vec[i];
    }
    return eigen_vec;
}

inline Eigen::MatrixXi vec2eigen( const std::vector<Eigen::Vector3i>& std_vec){
    //TODO not the most efficient way, can be done using map but that may break alignment: https://stackoverflow.com/questions/40852757/c-how-to-convert-stdvector-to-eigenmatrixxd

    if(std_vec.size()==0){
        Eigen::MatrixXi eigen_vec;
        return eigen_vec;
    }

    Eigen::MatrixXi eigen_vec(std_vec.size(),3);
    for (int i = 0; i < std_vec.size(); ++i) {
        eigen_vec.row(i)=std_vec[i];
    }
    return eigen_vec;
}

inline Eigen::MatrixXi vec2eigen( const std::vector<Eigen::Vector2i>& std_vec){
    //TODO not the most efficient way, can be done using map but that may break alignment: https://stackoverflow.com/questions/40852757/c-how-to-convert-stdvector-to-eigenmatrixxd

    if(std_vec.size()==0){
        Eigen::MatrixXi eigen_vec;
        return eigen_vec;
    }

    Eigen::MatrixXi eigen_vec(std_vec.size(),2);
    for (int i = 0; i < std_vec.size(); ++i) {
        eigen_vec.row(i)=std_vec[i];
    }
    return eigen_vec;
}

//filters the rows of an eigen matrix and returns only those for which the mask is equal to the keep
template <class T>
inline T filter_impl(std::vector<int>&indirection, std::vector<int>&inverse_indirection, const T& eigen_mat, const std::vector<bool> mask, const bool keep ){
    if(eigen_mat.rows()!=mask.size()){
        LOG(WARNING) << "filter: Eigen matrix and mask don't have the same size: " << eigen_mat.rows() << " and " << mask.size() ;
        return eigen_mat;
    }

    int nr_elem_to_keep=std::count(mask.begin(), mask.end(), keep);
    T new_eigen_mat( nr_elem_to_keep, eigen_mat.cols() );

    indirection.resize(eigen_mat.rows(),-1);
    inverse_indirection.resize(nr_elem_to_keep, -1);


    int insert_idx=0;
    for (int i = 0; i < eigen_mat.rows(); ++i) {
        if(mask[i]==keep){
            new_eigen_mat.row(insert_idx)=eigen_mat.row(i);
            indirection[i]=insert_idx;
            inverse_indirection[insert_idx]=i;
            insert_idx++;
        }
    }

    return new_eigen_mat;

}

template <class T>
inline T filter( const T& eigen_mat, const std::vector<bool> mask, const bool keep ){

    std::vector<int> indirection;
    std::vector<int> inverse_indirection;
    return filter_impl(indirection, inverse_indirection, eigen_mat, mask, keep);

}

template <class T>
inline T filter_return_indirection(std::vector<int>&indirection, const T& eigen_mat, const std::vector<bool> mask, const bool keep ){

    std::vector<int> inverse_indirection;
    return filter_impl(indirection, inverse_indirection, eigen_mat, mask, keep);

}

template <class T>
inline T filter_return_inverse_indirection(std::vector<int>&inverse_indirection, const T& eigen_mat, const std::vector<bool> mask, const bool keep ){

    std::vector<int> indirection;
    return filter_impl(indirection, inverse_indirection, eigen_mat, mask, keep);

}

template <class T>
inline T filter_return_both_indirection(std::vector<int>&indirection, std::vector<int>&inverse_indirection, const T& eigen_mat, const std::vector<bool> mask, const bool keep ){

    return filter_impl(indirection, inverse_indirection, eigen_mat, mask, keep);

}

//gets rid of the faces that are redirected to a -1 or edges that are also indirected into a -1
inline Eigen::MatrixXi filter_apply_indirection(const std::vector<int>&indirection, const Eigen::MatrixXi& eigen_mat ){

    if(!eigen_mat.size()){  //it's empty
        return eigen_mat;
    }

    if(eigen_mat.maxCoeff() > indirection.size()){
        LOG(FATAL) << "filter apply_indirection: eigen_mat is indexing indirection at a higher position than allowed" << eigen_mat.maxCoeff() << " " << indirection.size();
    }

    std::vector<Eigen::VectorXi> new_eigen_mat_vec;

    for (int i = 0; i < eigen_mat.rows(); ++i) {

        Eigen::VectorXi row= eigen_mat.row(i);
        bool should_keep=true;
        for (int j = 0; j < row.size(); ++j) {
            if (indirection[row(j)]==-1){
                //it points to a an already removed point so we will not keep it
                should_keep=false;
            }else{
                //it point to a valid vertex so we change the idx so that it point to that one
                row(j) = indirection[row(j)];
            }
        }

        if(should_keep){
            new_eigen_mat_vec.push_back(row);
        }

    }

    return vec2eigen(new_eigen_mat_vec);

}

//gets rid of the faces that are redirected to a -1 or edges that are also indirected into a -1 AND also returns a mask (size eigen_mat x 1) with value of TRUE for those which were keps
inline Eigen::MatrixXi filter_apply_indirection_return_mask(std::vector<bool>& mask_kept, const std::vector<int>&indirection, const Eigen::MatrixXi& eigen_mat, bool debug=false){

    if(!eigen_mat.size()){  //it's empty
        return eigen_mat;
    }

    if(eigen_mat.maxCoeff() > indirection.size()){
                LOG(FATAL) << "filter apply_indirection: eigen_mat is indexing indirection at a higher position than allowed" << eigen_mat.maxCoeff() << " " << indirection.size();
    }

    std::vector<Eigen::VectorXi> new_eigen_mat_vec;
    mask_kept.resize(eigen_mat.rows(),false);

    for (int i = 0; i < eigen_mat.rows(); ++i) {
        LOG_IF_S(INFO,debug) << "getting row " << i;
        Eigen::VectorXi row= eigen_mat.row(i);
        LOG_IF_S(INFO,debug) << "row is" << row;
        bool should_keep=true;
        for (int j = 0; j < row.size(); ++j) {
            LOG_IF_S(INFO,debug) << "value at column " << j << " is " << row(j);
            LOG_IF_S(INFO,debug) << "indirection has size " << indirection.size();
            LOG_IF_S(INFO,debug) << "value of indirection at is  " <<  indirection[row(j)];
            if (indirection[row(j)]==-1){
                //it points to a an already removed point so we will not keep it
                should_keep=false;
            }else{
                //it point to a valid vertex so we change the idx so that it point to that one
                row(j) = indirection[row(j)];
            }
        }

        if(should_keep){
            LOG_IF_S(INFO,debug) << "pushing new row " <<  row;
            new_eigen_mat_vec.push_back(row);
            LOG_IF_S(INFO,debug) << "setting face " << i << " to kept";
            mask_kept[i]=true;
        }

    }

    LOG_IF_S(INFO,debug) << "finished, doing a vec2eigen ";
    return vec2eigen(new_eigen_mat_vec, debug);

}



template <class T>
inline T concat(const T& mat_1, const T& mat_2){

    if(mat_1.cols()!=mat_2.cols() && mat_1.cols()!=0 && mat_2.cols()!=0){
        LOG(FATAL) << "concat: Eigen matrices don't have the same nr of columns: " << mat_1.cols() << " and " << mat_2.cols() ;
    }


    T mat_new(mat_1.rows() + mat_2.rows(), mat_1.cols());
    mat_new << mat_1, mat_2;
    return mat_new;
}

//swap a Vector2, if the input is not a vector2, swaps the first 2 elements
template <class T>
inline T swap(const T& vec){
    T swapped=vec;
    swapped(0)=vec(1);
    swapped(1)=vec(0);
    return swapped;
}

//https://stackoverflow.com/questions/1903954/is-there-a-standard-sign-function-signum-sgn-in-c-c
template <typename T> int sign(T val) {
    return (T(0) < val) - (val < T(0));
}

inline void set_default_color(Eigen::MatrixXd& C){
    C.col(0).setConstant(0.41);
    C.col(1).setConstant(0.58);
    C.col(2).setConstant(0.59);
}

inline Eigen::MatrixXd default_color(int nr_rows){
    Eigen::MatrixXd C(nr_rows,3);
    C.col(0).setConstant(0.41);
    C.col(1).setConstant(0.58);
    C.col(2).setConstant(0.59);
    return C;

}

inline bool XOR(bool a, bool b)
{
    return (a + b) % 2;
}

//convert an OpenCV type to a string value
inline std::string type2string(int type) {
    std::string r;

    unsigned char depth = type & CV_MAT_DEPTH_MASK;
    unsigned char chans = 1 + (type >> CV_CN_SHIFT);

    switch ( depth ) {
        case CV_8U:  r = "8U"; break;
        case CV_8S:  r = "8S"; break;
        case CV_16U: r = "16U"; break;
        case CV_16S: r = "16S"; break;
        case CV_32S: r = "32S"; break;
        case CV_32F: r = "32F"; break;
        case CV_64F: r = "64F"; break;
        default:     r = "User"; break;
    }

    r += "C";
    r += (chans+'0');

    return r;
}

inline void create_alpha_mat(const cv::Mat& mat, cv::Mat_<cv::Vec4b>& dst){
    std::vector<cv::Mat> matChannels;
    cv::split(mat, matChannels);

  // create alpha channel
  // cv::Mat alpha = matChannels.at(0) + matChannels.at(1) + matChannels.at(2);
  // matChannels.push_back(alpha);
    cv::Mat alpha=cv::Mat(mat.rows,mat.cols, CV_8UC1);
    alpha.setTo(cv::Scalar(255));
    matChannels.push_back(alpha);

    cv::merge(matChannels, dst);
}

inline cv::Mat_<cv::Vec4b> create_alpha_mat(const cv::Mat& mat){
    std::vector<cv::Mat> matChannels;
    cv::split(mat, matChannels);

    // create alpha channel
    // cv::Mat alpha = matChannels.at(0) + matChannels.at(1) + matChannels.at(2);
    // matChannels.push_back(alpha);
    cv::Mat alpha=cv::Mat(mat.rows,mat.cols, CV_8UC1);
    alpha.setTo(cv::Scalar(255));
    matChannels.push_back(alpha);

    cv::Mat_<cv::Vec4b> out;
    cv::merge(matChannels, out);
    return out;
}


//https://stackoverflow.com/a/217605
#include <algorithm>
#include <functional>
#include <cctype>
#include <locale>

// trim from start (in place)
static inline void ltrim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(),
            std::not1(std::ptr_fun<int, int>(std::isspace))));
}

// trim from end (in place)
static inline void rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(),
            std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
}

// trim from both ends (in place)
static inline void trim(std::string &s) {
    ltrim(s);
    rtrim(s);
}

// trim from start (copying)
static inline std::string ltrim_copy(std::string s) {
    ltrim(s);
    return s;
}

// trim from end (copying)
static inline std::string rtrim_copy(std::string s) {
    rtrim(s);
    return s;
}

// trim from both ends (copying)
static inline std::string trim_copy(std::string s) {
    trim(s);
    return s;
}

//https://stackoverflow.com/a/37454181
inline std::vector<std::string> split(const std::string& str, const std::string& delim){
    std::vector<std::string> tokens;
    size_t prev = 0, pos = 0;
    do{
        pos = str.find(delim, prev);
        if (pos == std::string::npos) pos = str.length();
        std::string token = str.substr(prev, pos-prev);
        if (!token.empty()) tokens.push_back(token);
        prev = pos + delim.length();
    }
    while (pos < str.length() && prev < str.length());
    return tokens;
}


// inline std::string file_to_string (const std::string &filename){
//     std::ifstream t(filename);
//     return std::string((std::istreambuf_iterator<char>(t)),
//             std::istreambuf_iterator<char>());
// }

//sometimes you want to allocate memory that is multiple of 64 bytes, so therefore you want to allocate more memory but you need a nr that is divisible by 64
inline int round_up_to_nearest_multiple(const int number, const int divisor){
 return number - number % divisor + divisor * !!(number % divisor);
}

//clamp a value between a min and a max
template <class T>
inline T clamp(const T val, const T min, const T max){
    return std::min(std::max(val, min),max);
}


inline int next_power_of_two(int x) { // https://github.com/LWJGL/lwjgl3-wiki/wiki/2.6.1.-Ray-tracing-with-OpenGL-Compute-Shaders-(Part-I)
  x--;
  x |= x >> 1; // handle 2 bit numbers
  x |= x >> 2; // handle 4 bit numbers
  x |= x >> 4; // handle 8 bit numbers
  x |= x >> 8; // handle 16 bit numbers
  x |= x >> 16; // handle 32 bit numbers
  x++;
  return x;
}


namespace configuru {
    // template<>
    // inline Eigen::Vector2f as(const configuru::Config& config)
    // {
    //     auto&& array = config.as_array();
    //     config.check(array.size() == 2, "Expected Vector2f");
    //     return {(float)array[0], (float)array[1]};
    // }
    //
    // template<>
    // inline Eigen::Affine3f as(const configuru::Config& config)
    // {
    //     auto&& array = config.as_array();
    //     if(array.size()!= 12 && array.size()!=16){
    //         config.on_error("Expected matrix of size 3x4 or 4x4");
    //     }
    //
    //     //if we have 16 elements, the last 4 of them should be row containing 0,0,0,1
    //     if(array.size()==16){
    //         bool bad=false;
    //         bad |= array[12]!=0.0;
    //         bad |= array[13]!=0.0;
    //         bad |= array[14]!=0.0;
    //         bad |= array[15]!=1.0;
    //         if(bad){
    //             std::cout << " last row is "<< array[12] << " " << array[13] << " " << array[14] << " " << array[15] << '\n';
    //             config.on_error("For an affine matrix of size 4x4 the last row should be 0,0,0,1");
    //         }
    //     }
    //
    //     Eigen::Affine3f mat;
    //     mat.matrix()<< (float)array[0], (float)array[1], (float)array[2], (float)array[3],
    //     (float)array[4], (float)array[5], (float)array[6], (float)array[7],
    //     (float)array[8], (float)array[9], (float)array[10], (float)array[11];
    //     return mat;
    //     // return {(float)array[0], (float)array[1]};
    // }


    // template<typename T>
    // explicit operator Eigen::Matrix< T , 2 , 1> const
    // {
    //     const auto& array = as_array();
    //     // Eigen::Matrix< T , 2 , 1> vec;
    //     config.check(array.size() == 2, "Expected vector of 2 elements");
    //     return {(float)array[0], (float)array[1]};
    //     // std::vector<T> ret;
    //     // ret.reserve(array.size());
    //     // for (auto&& config : array) {
    //     //     ret.push_back(static_cast<T>(config));
    //     // }
    //     // return ret;
    // }
}
