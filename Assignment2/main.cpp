// clang-format off
#include <iostream>
#include <opencv2/opencv.hpp>
#include "rasterizer.hpp"
#include "global.hpp"
#include "Triangle.hpp"


constexpr double MY_PI = 3.1415926;

Eigen::Matrix4f get_view_matrix(Eigen::Vector3f eye_pos)
{
    Eigen::Matrix4f view = Eigen::Matrix4f::Identity();

    Eigen::Matrix4f translate;
    translate << 1,0,0,-eye_pos[0],
                 0,1,0,-eye_pos[1],
                 0,0,1,-eye_pos[2],
                 0,0,0,1;

    view = translate*view;

    return view;
}

Eigen::Matrix4f get_model_matrix(float rotation_angle)
{
    Eigen::Matrix4f model = Eigen::Matrix4f::Identity();
    float rotation_angle_radian = rotation_angle * MY_PI / 180;
    model(0, 0) = cos(rotation_angle_radian);
    model(0, 1) = -sin(rotation_angle_radian);
    model(1, 0) = sin(rotation_angle_radian);
    model(1, 1) = cos(rotation_angle_radian);

    return model;
}

Eigen::Matrix4f get_projection_matrix(float eye_fov, float aspect_ratio, float zNear, float zFar)
{
    Eigen::Matrix4f projection = Eigen::Matrix4f::Identity();
    //zNear = -zNear;
    //zFar = -zFar;

    float t = tan((eye_fov * MY_PI / 180) / 2) * fabs(zNear);
    float r = aspect_ratio * t;
    float l = -r;
    float b = -t;

    Eigen::Matrix4f translate = Eigen::Matrix4f::Identity();
    translate(0, 3) = -(r + l) / 2;
    translate(1, 3) = -(t + b) / 2;
    translate(2, 3) = -(zNear + zFar) / 2;

    Eigen::Matrix4f scale = Eigen::Matrix4f::Identity();
    scale(0, 0) = 2 / (r - l);
    scale(1, 1) = 2 / (t - b);
    scale(2, 2) = 2 / (zNear - zFar);

    Eigen::Matrix4f ortho = scale * translate;

    Eigen::Matrix4f persp2ortho = Eigen::Matrix4f::Zero();

    persp2ortho(0, 0) = zNear;
    persp2ortho(1, 1) = zNear;
    persp2ortho(2, 2) = zNear + zFar;
    persp2ortho(2, 3) = -zNear * zFar;
    persp2ortho(3, 2) = 1;
    projection = ortho * persp2ortho;

    return projection;
}

int main(int argc, const char** argv)
{
    float angle = 0;
    bool command_line = false;
    std::string filename = "output.png";

    if (argc == 2)
    {
        command_line = true;
        filename = std::string(argv[1]);
    }

    rst::rasterizer r(700, 700);

    Eigen::Vector3f eye_pos = {0,0,5};


    std::vector<Eigen::Vector3f> pos
    {
            {2, 0, -2},
            {0, 2, -2},
            {-2, 0, -2},
            {3.5, -1, -5},
            {2.5, 1.5, -5},
            {-1, 0.5, -5}
    };

    std::vector<Eigen::Vector3i> ind
    {
            {0, 1, 2},
            {3, 4, 5}
    };

    std::vector<Eigen::Vector3f> cols
    {
            {217.0, 238.0, 185.0},
            {217.0, 238.0, 185.0},
            {217.0, 238.0, 185.0},
            {185.0, 217.0, 238.0},
            {185.0, 217.0, 238.0},
            {185.0, 217.0, 238.0}
    };

    auto pos_id = r.load_positions(pos);
    auto ind_id = r.load_indices(ind);
    auto col_id = r.load_colors(cols);

    int key = 0;
    int frame_count = 0;

    if (command_line)
    {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);

        r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45, 1, 0.1, 50));

        r.draw(pos_id, ind_id, col_id, rst::Primitive::Triangle);
        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);
        cv::cvtColor(image, image, cv::COLOR_RGB2BGR);

        cv::imwrite(filename, image);

        return 0;
    }

    while(key != 27)
    {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);

        r.set_model(get_model_matrix(30));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45, 1, 0.1, 50));

        r.draw(pos_id, ind_id, col_id, rst::Primitive::Triangle);

        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);
        cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
        cv::imshow("image", image);
        key = cv::waitKey(10);

        std::cout << "frame count: " << frame_count++ << '\n';
    }

    return 0;
}
// clang-format on