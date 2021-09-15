#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/voxel_grid.h>

#include <vector>
#include <fstream>
#include <iostream>
#include <GL/freeglut.h>
#include <boost/algorithm/string.hpp>
#include <string>
#include <sstream>

#include <ceres/ceres.h>

#include "../include/point_to_point_source_to_target_rotated_mirror_tait_bryan_wc_jacobian.h"

#include "imgui.h"
#include "imgui_impl_glut.h"
#include "imgui_impl_opengl2.h"

#include "utils.h"
#include "cost_fun.h"

const unsigned int window_width = 1920;
const unsigned int window_height = 1080;
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -10.0;
float translate_x, translate_y = 0.0;

std::vector<std::vector<DataStream>> data_streams;
Plane mirror;
Sophus::SE3d mirror_cal = Sophus::SE3d(Eigen::Matrix4d::Identity());
std::vector<Sophus::SE3d> instrument_poses;
std::vector<pcl::PointCloud<pcl::PointXYZ>> pcs_global;
float search_radious = 0.3;
pcl::PointCloud<pcl::PointXYZ> ground_truth;
std::vector<std::pair<Eigen::Vector3d,Eigen::Vector3d>> nns_to_draw;
bool shown_nn = false;

bool initGL(int *argc, char **argv);
void display();
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void reshape(int w, int h);
void printHelp();

pcl::PointCloud<pcl::PointXYZ> get_pc_global(const Plane& mirror, const Sophus::SE3d& instrument_pose, const std::vector<DataStream>& data_stream);
std::vector<std::pair<int,int>> nns(const pcl::PointCloud<pcl::PointXYZ>& pc1, const pcl::PointCloud<pcl::PointXYZ>& pc2, float radius);

struct imgui_data_type{
    Eigen::Matrix<float,6,1> mirror_cal;
    Eigen::Matrix<float,4,1> abcd;
    std::vector<Eigen::Matrix<float,6,1>> poses;
};
imgui_data_type imgui_data;
bool load_data_stream(std::vector<DataStream> &data, const std::string &filename)
{
	std::ifstream f;
	f.open(filename.c_str());
	if(f.good()) {
		std::string s;
		getline(f,s);
		int counter = 0;
		while(!f.eof())	{
			getline(f,s);
			std::vector<std::string> strs;
			boost::split(strs,s,boost::is_any_of(","));

			if(strs.size() == 6){
				DataStream ds;
				std::istringstream(strs[0]) >> ds.x;
				std::istringstream(strs[1]) >> ds.y;
				std::istringstream(strs[2]) >> ds.z;
				std::istringstream(strs[3]) >> ds.angle_rad;
				std::istringstream(strs[4]) >> ds.intensity;
				std::istringstream(strs[5]) >> ds.timestamp;

				Eigen::Vector3d v(ds.x, ds.y, ds.z);
				if(v.norm()>3){
					data.push_back(ds);
				}
			}
			counter++;
		}
		f.close();
	}else{
		return false;
	}

	////////////////////decimation
	Sophus::SE3d intrumentent_pose;
	pcl::PointCloud<pcl::PointXYZ> pc = get_pc_global(mirror, intrumentent_pose, data);

	pcl::PointCloud<pcl::PointXYZL>::Ptr pre_filteredcloud (new pcl::PointCloud<pcl::PointXYZL>);
	pcl::PointCloud<pcl::PointXYZL>::Ptr post_filteredcloud_m (new pcl::PointCloud<pcl::PointXYZL>);

	pre_filteredcloud->resize(pc.size());
	for (int i =0; i < pc.size(); i++){
		const auto & p1 = pc[i];
		pcl::PointXYZL p2;
		p2.label = i;
		p2.getArray3fMap() = Eigen::Vector3f{static_cast<float>(p1.x),
											 static_cast<float>(p1.y),
											 static_cast<float>(p1.z)};
		(*pre_filteredcloud)[i] = p2;
	}

	pcl::VoxelGrid<pcl::PointXYZL> sor0;
	sor0.setInputCloud (pre_filteredcloud);
	sor0.setLeafSize (0.1,0.1,0.1);
	sor0.filter (*post_filteredcloud_m);


	std::vector<DataStream> ds2;
	ds2.resize(post_filteredcloud_m->size());

	for (int i =0; i < ds2.size(); i++){
		const auto & p1 = (*post_filteredcloud_m)[i];
		const auto & p2 = data[p1.label];
		ds2[i] = p2;
	}

	data = ds2;

	return true;
}

int main(int argc, char *argv[]){
	pcl::io::loadPCDFile("../data/ground_truth.pcd", ground_truth);

	mirror.a = -0.794;
	mirror.b = -0.419;
	mirror.c = 0.438;
	Eigen::Vector3d v(mirror.a, mirror.b, mirror.c);
	v=v/v.norm();
	mirror.d = 0.01;

    mirror_cal = Sophus::SE3d(Eigen::Matrix4d::Identity());

	std::vector<std::string> file_names;

	file_names.push_back("../data/log1630860825.csv");
	file_names.push_back("../data/log1630860867.csv");

	TaitBryanPose pose;
	pose.px = 27.9894;
	pose.py = -2.36887;
	pose.pz = -0.37047;
	pose.om = 1.44997;
	pose.fi = -1.48019;
	pose.ka = 1.45228;
	instrument_poses.push_back(rotating_mirror::TaitBryanPoseToSE3(pose));

	pose.px = 37.6912;
	pose.py = -2.46042;
	pose.pz = -0.538841;
	pose.om = -1.23007;
	pose.fi = -1.54447;
	pose.ka = -1.23228;
    instrument_poses.push_back(rotating_mirror::TaitBryanPoseToSE3(pose));

    for (int i =0; i< instrument_poses.size(); i++)
    {
        imgui_data.poses.push_back(instrument_poses[i].log().cast<float>());
    }
    imgui_data.mirror_cal = mirror_cal.log().cast<float>();
    imgui_data.abcd = mirror.getABCD().cast<float>();

	for(size_t i = 0 ; i < file_names.size(); i++){
		std::cout << "loading " << i << std::endl;
		std::vector<DataStream> data;
		load_data_stream(data, file_names[i]);
		data_streams.push_back(data);
	}

	for(size_t i = 0; i < data_streams.size(); i++){
		pcl::PointCloud<pcl::PointXYZ> pc_global = get_pc_global(mirror, instrument_poses[i], data_streams[i]);
		pcs_global.push_back(pc_global);
	}

	if (false == initGL(&argc, argv)) {
		return 4;
	}

	printHelp();
	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	glutMainLoop();
}


bool initGL(int *argc, char **argv) {
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(window_width, window_height);
	glutCreateWindow("calibration");
	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutMotionFunc(motion);

	// default initialization
	glClearColor(1.0, 1.0, 1.0, 1.0);
	glEnable(GL_DEPTH_TEST);

	// viewport
	glViewport(0, 0, window_width, window_height);

	// projection
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60.0, (GLfloat) window_width / (GLfloat) window_height, 0.01,
			10000.0);
	glutReshapeFunc(reshape);
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls

    ImGui::StyleColorsDark();
    ImGui_ImplGLUT_Init();
    ImGui_ImplGLUT_InstallFuncs();
    ImGui_ImplOpenGL2_Init();
	return true;
}

void  update()
{
    for(size_t i = 0; i < data_streams.size(); i++){
        pcl::PointCloud<pcl::PointXYZ> pc_global = get_pc_global(mirror, instrument_poses[i], data_streams[i]);
        pcs_global[i] = pc_global;
    }
}
void display() {
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(translate_x, translate_y, translate_z);
	glRotatef(rotate_x, 1.0, 0.0, 0.0);
	glRotatef(rotate_y, 0.0, 0.0, 1.0);

	glBegin(GL_LINES);
		glColor3f(1.0f, 0.0f, 0.0f);
		glVertex3f(0.0f, 0.0f, 0.0f);
		glVertex3f(1.0f, 0.0f, 0.0f);

		glColor3f(0.0f, 1.0f, 0.0f);
		glVertex3f(0.0f, 0.0f, 0.0f);
		glVertex3f(0.0f, 1.0f, 0.0f);

		glColor3f(0.0f, 0.0f, 1.0f);
		glVertex3f(0.0f, 0.0f, 0.0f);
		glVertex3f(0.0f, 0.0f, 1.0f);
	glEnd();

	glColor3f(1,0,0);
	glBegin(GL_POINTS);
	for(size_t i = 0 ; i < pcs_global.size(); i++){
		for(size_t j = 0 ; j < pcs_global[i].size(); j++){
			glVertex3f(pcs_global[i][j].x, pcs_global[i][j].y, pcs_global[i][j].z);
		}
	}
	glEnd();

	glColor3f(0,1,0);
	glBegin(GL_POINTS);
	for(size_t i = 0 ; i < ground_truth.size(); i++){
		glVertex3f(ground_truth[i].x, ground_truth[i].y, ground_truth[i].z);
	}
	glEnd();

	if(shown_nn){
		glColor3f(0,0,1);
		glBegin(GL_LINES);
		for (auto &n : nns_to_draw)
		{
			glVertex3f(n.first.x(),n.first.y(),n.first.z());
			glVertex3f(n.second.x(),n.second.y(),n.second.z());
		}
		glEnd();
	}
    ImGui_ImplOpenGL2_NewFrame();
    ImGui_ImplGLUT_NewFrame();

    ImGui::Begin("Demo Window2");
    if (ImGui::Button("ma+")){ mirror.a+=0.01; update();}
    ImGui::SameLine();
    if (ImGui::Button("ma-")){ mirror.a-=0.01; update();}

    if (ImGui::Button("mb+")){ mirror.b+=0.01; update();}
    ImGui::SameLine();
    if (ImGui::Button("mb-")){ mirror.b-=0.01; update();}

    if (ImGui::Button("mc+")){ mirror.c+=0.01; update();}
    ImGui::SameLine();
    if (ImGui::Button("mc-")){ mirror.c-=0.01; update();}

    if (ImGui::Button("md+")){ mirror.d+=0.01; update();}
    ImGui::SameLine();
    if (ImGui::Button("md-")){ mirror.d-=0.01; update();}

    ImGui::Text("------");
    for (int i=0; i <6 ; i++) {
        std::string f1 = "mcal_"+std::to_string(i)+"+";
        std::string f2 = "mcal_"+std::to_string(i)+"-";
        if (ImGui::Button(f1.c_str())) {
            auto c = mirror_cal.log();
            c[i] += 0.01;
            mirror_cal = Sophus::SE3d::exp(c);
            update();
        }
        ImGui::SameLine();
        if (ImGui::Button(f2.c_str())) {
            auto c = mirror_cal.log();
            c[i] -= 0.01;
            mirror_cal = Sophus::SE3d::exp(c);
            update();
        }
    }

    for (int j =0; j < instrument_poses.size(); j++)
    {
        for (int i=0; i <6 ; i++) {
            std::string f1 = "p"+std::to_string(j)+"_"+std::to_string(i)+"+";
            std::string f2 = "p"+std::to_string(j)+"_"+std::to_string(i)+"-";
            if (ImGui::Button(f1.c_str())) {
                auto c = instrument_poses[j].log();
                c[i] += 0.01;
                instrument_poses[j] = Sophus::SE3d::exp(c);
                update();
            }
            ImGui::SameLine();
            if (ImGui::Button(f2.c_str())) {
                auto c = instrument_poses[j].log();
                c[i] -= 0.01;
                instrument_poses[j] = Sophus::SE3d::exp(c);
                update();
            }
        }
    }

    if (ImGui::Button("export pcd")){
        pcl::PointCloud<pcl::PointXYZ> pc_save;
        for (int i=0; i < pcs_global.size();i++)
        {
            for (int j=0; j < pcs_global[i].size();j++)
            {
                pc_save.push_back(pcs_global[i][j]);
            }
        }

        pcl::io::savePCDFile("pcd_exported2.pcd", pc_save);
    }

    ImGui::Text("Text");
    ImGui::End();


    ImGui::Render();
    ImGui_ImplOpenGL2_RenderDrawData(ImGui::GetDrawData());
	glutSwapBuffers();
    glutPostRedisplay();
}



void keyboard(unsigned char key, int /*x*/, int /*y*/) {
	switch (key) {
		case (27): {
			glutDestroyWindow(glutGetWindow());
			return;
		}
		case 'c':{

            std::cout << "optimize" << std::endl;
            ceres::Problem problem;
            Eigen::Vector4d  plane {mirror.a,mirror.b,mirror.c,mirror.d};
            std::vector<Sophus::Vector6d> params_before;
            problem.AddParameterBlock(plane.data(), 4, new LocalParameterizationPlane());
            for (int i =0; i< instrument_poses.size();i++)
            {
                problem.AddParameterBlock(instrument_poses[i].data(), Sophus::SE3d::num_parameters, new LocalParameterizationSE3());
                params_before.push_back(instrument_poses[i].log());
            }
            problem.AddParameterBlock(mirror_cal.data(), Sophus::SE3d::num_parameters, new LocalParameterizationSE3());
            problem.SetParameterBlockConstant(mirror_cal.data());

            const Sophus::Vector6d mirror_pose_before = mirror_cal.log();
            for(size_t i = 0 ; i < data_streams.size(); i++){
                auto nn = nns (pcs_global[i],ground_truth, search_radious);

                std::cout << "nn.size(): " << nn.size() << std::endl;



                for (auto & n : nn) {
                    Sophus::SE3d &instrument_pose_i = instrument_poses[i];

                    Eigen::Vector3d ray_i(data_streams[i][n.first].x, data_streams[i][n.first].y,
                                          data_streams[i][n.first].z);

                    Eigen::Vector3d traget_i(ground_truth[n.second].x, ground_truth[n.second].y,
                                             ground_truth[n.second].z);

                    double om_mirror = data_streams[i][n.first].angle_rad;
                    ceres::LossFunction *loss = nullptr;// new ceres::CauchyLoss(0.2);

                    ceres::CostFunction * cost_function = MirrorOptimization::Create(ray_i, traget_i, om_mirror);
                    problem.AddResidualBlock(cost_function, loss, plane.data(), instrument_poses[i].data(), mirror_cal.data());

                }

            }
            ceres::Solver::Options options;
            options.linear_solver_type = ceres::DENSE_QR;
            options.minimizer_progress_to_stdout = true;
            options.max_num_iterations = 50;
            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);
            std::cout << summary.FullReport() << "\n";
            std::cout << "mirror diff "  << plane.x()-mirror.a <<", "<< plane.y()-mirror.b <<", "<< plane.z()-mirror.c <<", "<< plane.w()-mirror.d <<"\n";
            for (int i=0; i <  params_before.size();i++)
            {
                std::cout << " -> pose "<< i <<"diff "  << (instrument_poses[i].log()-params_before[i]).transpose() << std::endl;
            }
            std::cout << " mirror diff "  << (mirror_pose_before-mirror_cal.log()).transpose() << std::endl;
            for(size_t i = 0; i < data_streams.size(); i++){
                pcl::PointCloud<pcl::PointXYZ> pc_global = get_pc_global(mirror, instrument_poses[i], data_streams[i]);
                pcs_global[i] = pc_global;
            }
            break;

		}
	}
	printHelp();
	glutPostRedisplay();
}


void mouse(int glut_button, int state, int x, int y) {

    ImGuiIO& io = ImGui::GetIO();
    io.MousePos = ImVec2((float)x, (float)y);
    int button = -1;
    if (glut_button == GLUT_LEFT_BUTTON) button = 0;
    if (glut_button == GLUT_RIGHT_BUTTON) button = 1;
    if (glut_button == GLUT_MIDDLE_BUTTON) button = 2;
    if (button != -1 && state == GLUT_DOWN)
        io.MouseDown[button] = true;
    if (button != -1 && state == GLUT_UP)
        io.MouseDown[button] = false;

    if (!io.WantCaptureMouse)
    {
        if (state == GLUT_DOWN) {
            mouse_buttons |= 1 << glut_button;
        } else if (state == GLUT_UP) {
            mouse_buttons = 0;
        }
        mouse_old_x = x;
        mouse_old_y = y;
    }

}

void motion(int x, int y) {
    ImGuiIO& io = ImGui::GetIO();
    io.MousePos = ImVec2((float)x, (float)y);

    if (!io.WantCaptureMouse)
    {
        float dx, dy;
        dx = (float) (x - mouse_old_x);
        dy = (float) (y - mouse_old_y);
        if (mouse_buttons & 1) {
            rotate_x += dy * 0.2f;
            rotate_y += dx * 0.2f;
        } else if (mouse_buttons & 4) {
            translate_z += dy * 0.05f;
        } else if (mouse_buttons & 3) {
            translate_x += dx * 0.05f;
            translate_y -= dy * 0.05f;
        }
        mouse_old_x = x;
        mouse_old_y = y;
    }
    glutPostRedisplay();
}


void reshape(int w, int h) {
	glViewport(0, 0, (GLsizei) w, (GLsizei) h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60.0, (GLfloat) w / (GLfloat) h, 0.01, 10000.0);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}

void printHelp() {
	std::cout << "-------help-------" << std::endl;
	std::cout << "c: calibrate" << std::endl;
}


pcl::PointCloud<pcl::PointXYZ> get_pc_global(const Plane& mirror, const Sophus::SE3d & instrument_pose, const std::vector<DataStream>& data_stream)
{
	pcl::PointCloud<pcl::PointXYZ> pc_global;
    Eigen::Vector4d  plane{ mirror.a, mirror.b, mirror.c, mirror.d};

	for(auto &ds:data_stream){
		double x;
		double y;
		double z;
		Eigen::Vector3d ray(ds.x, ds.y, ds.z);
		double angle_rad = ds.angle_rad;
        Eigen::Matrix<double,3,1> g = getPointOfRay<double>(plane, instrument_pose,mirror_cal, ray, angle_rad);
		pcl::PointXYZ p;
		p.getArray3fMap() = g.cast<float>();
		pc_global.push_back(p);
	}
	return pc_global;
}

std::vector<std::pair<int,int>> nns(const pcl::PointCloud<pcl::PointXYZ>& pc1, const pcl::PointCloud<pcl::PointXYZ>& pc2, float radius)
{
	nns_to_draw.clear();

    std::vector<std::pair<int,int>> result;

    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    for(size_t i = 0; i < pc2.size(); i++){
        cloud->push_back(pc2[i]);
    }
    int K = 1;

    std::vector<int> pointIdxRadiusSearch;
    std::vector<float> pointRadiusSquaredDistance;

    kdtree.setInputCloud (cloud);
    for(size_t k = 0; k < pc1.size(); k++){
        if ( kdtree.radiusSearch (pc1[k], radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0 ){
            for (std::size_t i = 0; i < pointIdxRadiusSearch.size (); ++i){

            	std::pair<Eigen::Vector3d,Eigen::Vector3d>  n = std::pair<Eigen::Vector3d,Eigen::Vector3d>(pc2[pointIdxRadiusSearch[i]].getArray3fMap().cast<double>(),
                        pc1[k].getArray3fMap().cast<double>());

            		Eigen::Vector3d n1 = pc1[k].getArray3fMap().cast<double>();
					Eigen::Vector3d n2 = pc2[pointIdxRadiusSearch[i]].getArray3fMap().cast<double>() ;

            		if((n2-n1).norm() > 0.001){
            			result.emplace_back( k, pointIdxRadiusSearch[i]);
						nns_to_draw.emplace_back(n1, n2);
            		}

                break;
            }
        }
    }
    std::cout << "nns.size(): " << result.size() << std::endl;
    return result;
}




