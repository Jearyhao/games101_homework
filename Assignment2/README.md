#作业完成情况

-代码能编译运行
-正确测试点在三角形内
-正确实现三角形栅格化算法
-正确实现 z-buffer 算法, 将三角形按顺序画在屏幕上
-利用super-sampling处理Anti-aliasing

#函数实现功能

insideTriangle（） 利用二维坐标矢量的叉乘计算公式，AB，BC，CA分别与AP，BP，CP叉乘，若结果同为正或同为负，则判定点落在三角形内

rasterizer类 要实现SSAA，每个像素需要维护一个sample list，因此要对rasterizer类进行改写，增加两个一维数组frame_buf_2xSSAA和depth_buf_2xSSAA用来维护每个样本的深度值。因为加入2个新变量，所以要对初始化函数rasterizer()和clear()函数进行增添超采样帧缓存和深度缓冲的构造和清除有关的内容。并且利用宏定义可以自行设置是否采用SSAA超采样

rasterize_triangle()

首先找出三角形x,y轴的最大和最小值以便确定bounding box的大小
为了方便遍历，对最大值向上取整，最小值向下取整。
定义一个数组存储每个像素的4个采样点
循环遍历bounding box，判断采样点是否在三角形内，若z_interpolated小于采样点的深度值，则替换depth_buf_2xSSAA对应数组元素的值并将颜色写入frame_buf_2xSSAA对应数组元素的值
若4个采样点不全在三角形内，则将depth_buf_2xSSAA的对应数组元素改为frame_buf_2xSSAA对应数组元素的最小值，将frame_buf_2xSSAA对应数组元素的值取平均得到颜色的值，通过set_pixel设置当前像素的颜色