#include<cmath>
#include<eigen3/Eigen/Core>
#include<eigen3/Eigen/Dense>
#include<iostream>

using namespace std;
using namespace Eigen;

int main(){

    double a = M_PI / 4;
    Vector3f v(2, 1, 1);
    Matrix3f m {{cos(a), -sin(a), 1}, {sin(a), cos(a), 2}, {0, 0, 1}};

    v = m * v;
    cout << v << endl;

    return 0;
}