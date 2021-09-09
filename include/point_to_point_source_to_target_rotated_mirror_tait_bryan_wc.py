from sympy import *
from tait_bryan_R_utils import *

x_t, y_t, z_t = symbols('x_t y_t z_t')
px, py, pz = symbols('px py pz')
om, fi, ka = symbols('om fi ka')
pxc, pyc, pzc = symbols('pxc pyc pzc')
omc, fic, kac = symbols('omc fic kac')
om_mirror = symbols('om_mirror')
ray_dir_x, ray_dir_y, ray_dir_z, ray_length = symbols('ray_dir_x ray_dir_y ray_dir_z ray_length')
plane_a, plane_b, plane_c, plane_d = symbols('plane_a plane_b plane_c plane_d')

position_symbols = [px, py, pz]
orientation_symbols = [om, fi, ka]
position_symbols_cal = [pyc, pzc]
orientation_symbols_cal = [fic, kac]
plane_symbols = [plane_a, plane_b, plane_c, plane_d]
all_symbols = position_symbols + orientation_symbols + plane_symbols + position_symbols_cal + orientation_symbols_cal

mcal=matrix44FromTaitBryan(pxc, pyc, pzc, omc, fic, kac)

m_rot_mirror=mcal * matrix44FromTaitBryan(0, 0, 0, om_mirror, 0, 0)

plane = Matrix([[plane_a, plane_b, plane_c, plane_d]])

R_cw=m_rot_mirror[:-1,:-1].transpose()
T_wc=Matrix([0, 0, 0]).vec()
T_cw=-R_cw*T_wc
RT_cw=Matrix.hstack(R_cw, T_cw)
m_rot_mirror_inv=Matrix.vstack(RT_cw, Matrix([[0,0,0,1]]))

plane_t = plane * m_rot_mirror_inv

plane_a = plane_t[0]
plane_b = plane_t[1]
plane_c = plane_t[2]
plane_d = plane_t[3]


a = plane_a * ray_dir_x + plane_b * ray_dir_y + plane_c * ray_dir_z

intersection_x = - ray_dir_x * (plane_d/a)
intersection_y = - ray_dir_y * (plane_d/a)
intersection_z = - ray_dir_z * (plane_d/a)

n=Matrix([plane_a, plane_b, plane_c]).vec()
d=Matrix([ray_dir_x, ray_dir_y, ray_dir_z]).vec()
rd=2*d.dot(n)*n-d 

ll = ray_length - sqrt(intersection_x * intersection_x + intersection_y * intersection_y + intersection_z * intersection_z)
x_s=-(intersection_x + rd[0] * ll)
y_s=-(intersection_y + rd[1] * ll)
z_s=-(intersection_y + rd[2] * ll)

point_source = Matrix([x_s, y_s, z_s, 1]).vec()
point_target = Matrix([x_t, y_t, z_t]).vec()

transformed_point_source = (matrix44FromTaitBryan(px, py, pz, om, fi, ka) * point_source)[:-1,:]
target_value = Matrix([0,0,0]).vec()
model_function = transformed_point_source-point_target

delta = target_value - model_function
delta_jacobian=delta.jacobian(all_symbols)
print(delta)
print(delta_jacobian)
print(point_source)

with open("point_to_point_source_to_target_rotated_mirror_tait_bryan_wc_jacobian.h",'w') as f_cpp:  
    f_cpp.write("inline void transform_point_rotated_mirror_tait_bryan_wc(double &x, double &y, double &z, double &px, double &py, double &pz, double &om, double &fi, double &ka, double &ray_dir_x, double &ray_dir_y, double &ray_dir_z, double &ray_length, double &plane_a, double &plane_b, double &plane_c, double &plane_d, double &om_mirror, double &pxc, double &pyc, double &pzc, double &omc, double &fic, double &kac)\n")
    f_cpp.write("{")
    f_cpp.write("x = %s;\n"%(ccode(transformed_point_source[0])))
    f_cpp.write("y = %s;\n"%(ccode(transformed_point_source[1])))
    f_cpp.write("z = %s;\n"%(ccode(transformed_point_source[2])))
    f_cpp.write("}")
    f_cpp.write("\n")
    f_cpp.write("inline void point_to_point_source_to_target_rotated_mirror_tait_bryan_wc_jacobian(Eigen::Matrix<double, 3, 14, Eigen::RowMajor> &j, double &px, double &py, double &pz, double &om, double &fi, double &ka, double &ray_dir_x, double &ray_dir_y, double &ray_dir_z, double &ray_length, double &plane_a, double &plane_b, double &plane_c, double &plane_d, double &x_t, double &y_t, double &z_t, double &om_mirror, double &pxc, double &pyc, double &pzc, double &omc, double &fic, double &kac)\n")
    f_cpp.write("{")
    for i in range (3):
        for j in range (14):
            f_cpp.write("j.coeffRef(%d,%d) = %s;\n"%(i,j, ccode(delta_jacobian[i,j])))
    f_cpp.write("}")
    f_cpp.write("\n")
    f_cpp.write("inline void point_to_point_source_to_target_rotated_mirror_tait_bryan_wc(Eigen::Matrix<double, 3, 1> &delta, double &px, double &py, double &pz, double &om, double &fi, double &ka, double &ray_dir_x, double &ray_dir_y, double &ray_dir_z, double &ray_length, double &plane_a, double &plane_b, double &plane_c, double &plane_d, double &x_t, double &y_t, double &z_t, double &om_mirror, double &pxc, double &pyc, double &pzc, double &omc, double &fic, double &kac)\n")
    f_cpp.write("{")
    for i in range (3):
        for j in range (1):
            f_cpp.write("delta.coeffRef(%d,%d) = %s;\n"%(i,j, ccode(delta[i,j])))
    f_cpp.write("}")
    f_cpp.write("\n")





