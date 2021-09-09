#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <vector>
#include <fstream>
#include <iostream>
#include <GL/freeglut.h>
#include <boost/algorithm/string.hpp>
#include <string>
#include <sstream>

#include <Eigen/Eigen>

#include "../include/point_to_point_source_to_target_rotated_mirror_tait_bryan_wc_jacobian.h"

inline double cauchy(double delta, double b){
	return 1.0 / (M_PI * b *( 1.0 + ((delta)/b) * ((delta)/b) ) );
}

struct DataStream{
	double x;
	double y;
	double z;
	double angle_rad;
	double intensity;
	double timestamp;
};

struct TaitBryanPose
{
	double px;
	double py;
	double pz;
	double om;
	double fi;
	double ka;

	TaitBryanPose(){
		px = py = pz = om = fi = ka = 0.0;
	}
};

struct Plane{
	double a;
	double b;
	double c;
	double d;
	Eigen::Affine3d m;

	float distance_to_plane(Eigen::Vector3d p){
		return a * p.x() + b * p.y() + c * p.z() + d;
	}

	void from_m_to_abcd(){
		a = m(0,2);
		b = m(1,2);
		c = m(2,2);
		d = -a * m(0,3) - b * m(1,3) - c * m(2,3);
	}

};

const unsigned int window_width = 1920;
const unsigned int window_height = 1080;
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -10.0;
float translate_x, translate_y = 0.0;

std::vector<std::vector<DataStream>> data_streams;
Plane mirror;
TaitBryanPose mirror_cal;
std::vector<TaitBryanPose> instrument_poses;
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

pcl::PointCloud<pcl::PointXYZ> get_pc_global(const Plane& mirror, const TaitBryanPose& instrument_pose, const std::vector<DataStream>& data_stream);
std::vector<std::pair<int,int>> nns(const pcl::PointCloud<pcl::PointXYZ>& pc1, const pcl::PointCloud<pcl::PointXYZ>& pc2, float radius);

inline Eigen::Affine3d affine_matrix_from_pose_tait_bryan(const TaitBryanPose& pose)
{
	Eigen::Affine3d m = Eigen::Affine3d::Identity();

	double sx = sin(pose.om);
	double cx = cos(pose.om);
	double sy = sin(pose.fi);
	double cy = cos(pose.fi);
	double sz = sin(pose.ka);
	double cz = cos(pose.ka);

	m(0,0) = cy * cz;
	m(1,0) = cz * sx * sy + cx * sz;
	m(2,0) = -cx * cz * sy + sx * sz;

	m(0,1) = -cy * sz;
	m(1,1) = cx * cz - sx * sy * sz;
	m(2,1) = cz * sx + cx * sy * sz;

	m(0,2) = sy;
	m(1,2) = -cy * sx;
	m(2,2) = cx * cy;

	m(0,3) = pose.px;
	m(1,3) = pose.py;
	m(2,3) = pose.pz;

	return m;
}

inline TaitBryanPose pose_tait_bryan_from_affine_matrix(Eigen::Affine3d m){
	TaitBryanPose pose;

	pose.px = m(0,3);
	pose.py = m(1,3);
	pose.pz = m(2,3);

	if (m(0,2) < 1) {
		if (m(0,2) > -1) {
			//case 1
			pose.fi = asin(m(0,2));
			pose.om = atan2(-m(1,2), m(2,2));
			pose.ka = atan2(-m(0,1), m(0,0));

			return pose;
		}
		else //r02 = −1
		{
			//case 2
			// not a unique solution: thetaz − thetax = atan2 ( r10 , r11 )
			pose.fi = -M_PI / 2.0;
			pose.om = -atan2(m(1,0), m(1,1));
			pose.ka = 0;
			return pose;
		}
	}
	else {
		//case 3
		// r02 = +1
		// not a unique solution: thetaz + thetax = atan2 ( r10 , r11 )
		pose.fi = M_PI / 2.0;
		pose.om = atan2(m(1,0), m(1,1));
		pose.ka = 0.0;
		return pose;
	}

	return pose;
}

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
	TaitBryanPose instrument_pose;
	pcl::PointCloud<pcl::PointXYZ> pc = get_pc_global(mirror, instrument_pose, data);

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

	//from calibration
	//a,b,c,d: -0.795092 -0.419594 0.437916 0.00453302
	//pose 0
	//27.99 -2.44216 -0.551882 1.5202 -1.47909 1.51403
	//pose 1
	//37.6979 -2.41142 -0.537749 -1.37268 -1.54515 -1.38618
	//mirror_cal: 0 0 0 0 0.0017875 0.00588813



	pcl::io::loadPCDFile("../data/ground_truth.pcd", ground_truth);

	mirror.a =  -0.795092;
	mirror.b = -0.419594;
	mirror.c = 0.437916;
	Eigen::Vector3d v(mirror.a, mirror.b, mirror.c);
	v=v/v.norm();
	mirror.d = 0.00453302;


	mirror_cal.px = 0;
	mirror_cal.py = 0;
	mirror_cal.pz = 0;
	mirror_cal.om = 0;
	mirror_cal.fi = 0.0017875;
	mirror_cal.ka = 0.00588813;

	std::vector<std::string> file_names;
	TaitBryanPose pose;

	file_names.push_back("../data/log1630860661.csv");
	pose.px = 2.60492;
	pose.py = -10.2245;
	pose.pz = -0.632075;
	pose.om = 1.56086;
	pose.fi = -1.49079;
	pose.ka = 1.56083;
	instrument_poses.push_back(pose);

	file_names.push_back("../data/log1630860904.csv");
	pose.px = 46.0542;
	pose.py = -2.08642;
	pose.pz = -0.597237;
	pose.om = 1.43695;
	pose.fi = -1.49007;
	pose.ka = 1.43652;
	instrument_poses.push_back(pose);

	file_names.push_back("../data/log1630861009.csv");
	pose.px = 65.6144;
	pose.py = -7.95168;
	pose.pz = -0.617178;
	pose.om = -3.10159;
	pose.fi = -1.56159;
	pose.ka = 3.14159;
	instrument_poses.push_back(pose);



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
	glutCreateWindow("validation");
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

	return true;
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

	glutSwapBuffers();
}


void keyboard(unsigned char key, int /*x*/, int /*y*/) {
	switch (key) {
		case (27): {
			glutDestroyWindow(glutGetWindow());
			return;
		}
	case 'x':{
		for (int i =0; i< pcs_global.size(); i++){
			pcl::io::savePCDFile("site"+std::to_string(i)+".pcd", pcs_global[i]);
		}
		break;
	}

		case 'v':{

			for(int iter = 0; iter < 50; iter++){

				std::cout << "optimize" << std::endl;

				std::vector<Eigen::Triplet<double>> tripletListA;
				std::vector<Eigen::Triplet<double>> tripletListP;
				std::vector<Eigen::Triplet<double>> tripletListB;

				double plane_a = mirror.a;
				double plane_b = mirror.b;
				double plane_c = mirror.c;
				double plane_d = mirror.d;


				for(size_t i = 0 ; i < data_streams.size(); i++){
					auto nn = nns (pcs_global[i],ground_truth, search_radious);

					for (auto & n : nn)
					{
						TaitBryanPose instrument_pose_i = instrument_poses[i];

						Eigen::Vector3d ray_i(data_streams[i][n.first].x, data_streams[i][n.first].y, data_streams[i][n.first].z);
						double ray_i_length = ray_i.norm();
						Eigen::Vector3d ray_i_n = ray_i/ray_i.norm();

						double x_t = ground_truth[n.second].x;
						double y_t = ground_truth[n.second].y;
						double z_t = ground_truth[n.second].z;

						double om_mirror = data_streams[i][n.first].angle_rad;

						Eigen::Matrix<double, 3, 1> delta;
						point_to_point_source_to_target_rotated_mirror_tait_bryan_wc(
								delta,
								instrument_pose_i.px,
								instrument_pose_i.py,
								instrument_pose_i.pz,
								instrument_pose_i.om,
								instrument_pose_i.fi,
								instrument_pose_i.ka,
								ray_i_n.x(),
								ray_i_n.y(),
								ray_i_n.z(),
								ray_i_length,
								plane_a,
								plane_b,
								plane_c,
								plane_d,
								x_t,
								y_t,
								z_t,
								om_mirror,
								mirror_cal.px,
								mirror_cal.py,
								mirror_cal.pz,
								mirror_cal.om,
								mirror_cal.fi,
								mirror_cal.ka);

						Eigen::Matrix<double, 3, 14, Eigen::RowMajor> jacobian;

						point_to_point_source_to_target_rotated_mirror_tait_bryan_wc_jacobian(
								jacobian,
								instrument_pose_i.px,
								instrument_pose_i.py,
								instrument_pose_i.pz,
								instrument_pose_i.om,
								instrument_pose_i.fi,
								instrument_pose_i.ka,
								ray_i_n.x(),
								ray_i_n.y(),
								ray_i_n.z(),
								ray_i_length,
								plane_a,
								plane_b,
								plane_c,
								plane_d,
								x_t,
								y_t,
								z_t,
								om_mirror,
								mirror_cal.px,
								mirror_cal.py,
								mirror_cal.pz,
								mirror_cal.om,
								mirror_cal.fi,
								mirror_cal.ka);

						int ir = tripletListB.size();
						int ic = i * 6;

						tripletListA.emplace_back(ir     , ic + 0, -jacobian(0,0));
						tripletListA.emplace_back(ir     , ic + 1, -jacobian(0,1));
						tripletListA.emplace_back(ir     , ic + 2, -jacobian(0,2));
						tripletListA.emplace_back(ir     , ic + 3, -jacobian(0,3));
						tripletListA.emplace_back(ir     , ic + 4, -jacobian(0,4));
						tripletListA.emplace_back(ir     , ic + 5, -jacobian(0,5));


						tripletListA.emplace_back(ir + 1 , ic + 0, -jacobian(1,0));
						tripletListA.emplace_back(ir + 1 , ic + 1, -jacobian(1,1));
						tripletListA.emplace_back(ir + 1 , ic + 2, -jacobian(1,2));
						tripletListA.emplace_back(ir + 1 , ic + 3, -jacobian(1,3));
						tripletListA.emplace_back(ir + 1 , ic + 4, -jacobian(1,4));
						tripletListA.emplace_back(ir + 1 , ic + 5, -jacobian(1,5));


						tripletListA.emplace_back(ir + 2 , ic + 0, -jacobian(2,0));
						tripletListA.emplace_back(ir + 2 , ic + 1, -jacobian(2,1));
						tripletListA.emplace_back(ir + 2 , ic + 2, -jacobian(2,2));
						tripletListA.emplace_back(ir + 2 , ic + 3, -jacobian(2,3));
						tripletListA.emplace_back(ir + 2 , ic + 4, -jacobian(2,4));
						tripletListA.emplace_back(ir + 2 , ic + 5, -jacobian(2,5));


						tripletListP.emplace_back(ir    , ir    ,  1);
						tripletListP.emplace_back(ir + 1, ir + 1,  1);
						tripletListP.emplace_back(ir + 2, ir + 2,  1);


						tripletListB.emplace_back(ir    , 0,  delta(0,0));
						tripletListB.emplace_back(ir + 1, 0,  delta(1,0));
						tripletListB.emplace_back(ir + 2, 0,  delta(2,0));
					}
				}


				int state_vec_l = data_streams.size() * 6;
				std::cout << "state_vec_l " << state_vec_l << std::endl;

				Eigen::SparseMatrix<double> matA(tripletListB.size(), state_vec_l);
				Eigen::SparseMatrix<double> matP(tripletListB.size(), tripletListB.size());
				Eigen::SparseMatrix<double> matB(tripletListB.size(), 1);

				matA.setFromTriplets(tripletListA.begin(), tripletListA.end());
				matP.setFromTriplets(tripletListP.begin(), tripletListP.end());
				matB.setFromTriplets(tripletListB.begin(), tripletListB.end());

				Eigen::SparseMatrix<double> AtPA(state_vec_l, state_vec_l);
				Eigen::SparseMatrix<double> AtPB(state_vec_l, 1);

				{
				Eigen::SparseMatrix<double> AtP = matA.transpose() * matP;
				AtPA = (AtP) * matA;
				AtPB = (AtP) * matB;
				}

				Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>> solver(AtPA);

				Eigen::SparseMatrix<double> x = solver.solve(AtPB);

				std::vector<double> h_x;

				h_x.resize(state_vec_l);

				for(size_t i = 0 ; i < state_vec_l; i++)h_x[i] = 0;

				for (int k=0; k<x.outerSize(); ++k){
					for (Eigen::SparseMatrix<double>::InnerIterator it(x,k); it; ++it){
						h_x[it.row()] = it.value();
					}
				}

				std::cout << "h_x.size() " << h_x.size() << std::endl;
				std::cout << "results" << std::endl;
				for(size_t i = 0 ; i < h_x.size(); i++){
					std::cout << i << " " << h_x[i] << std::endl;
				}

				if(h_x.size() == state_vec_l){
					int counter = 0;


					for (int i=0; i < instrument_poses.size(); i++)
					{
						instrument_poses[i].px += h_x[counter++]*0.5;
						instrument_poses[i].py += h_x[counter++]*0.5;
						instrument_poses[i].pz += h_x[counter++]*0.5;
						instrument_poses[i].om += h_x[counter++]*0.5;
						instrument_poses[i].fi += h_x[counter++]*0.5;
						instrument_poses[i].ka += h_x[counter++]*0.5;
					}

					std::cout << "a,b,c,d: " << mirror.a << " " << mirror.b << " " << mirror.c << " " << mirror.d << std::endl;
					for (int i=0; i < instrument_poses.size(); i++)
					{
						std::cout << instrument_poses[i].px << " " << instrument_poses[i].py << " " << instrument_poses[i].pz << " " <<
								instrument_poses[i].om << " " << instrument_poses[i].fi << " " << instrument_poses[i].ka << std::endl;
					}

					std::cout << "mirror_cal: " << mirror_cal.px << " " << mirror_cal.py << " " << mirror_cal.pz << " " <<
							mirror_cal.om << " " << mirror_cal.fi << " " << mirror_cal.ka << std::endl;

					for(size_t i = 0; i < data_streams.size(); i++){
						pcl::PointCloud<pcl::PointXYZ> pc_global = get_pc_global(mirror, instrument_poses[i], data_streams[i]);
						pcs_global[i] = pc_global;
					}
				}else{
					std::cout << "OPTIMIZATION FAILED" << std::endl;
				}
			}
			break;
		}
	}

	printHelp();
	glutPostRedisplay();
}


void mouse(int button, int state, int x, int y) {
	if (state == GLUT_DOWN) {
		mouse_buttons |= 1 << button;
	} else if (state == GLUT_UP) {
		mouse_buttons = 0;
	}

	mouse_old_x = x;
	mouse_old_y = y;
}

void motion(int x, int y) {
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
	std::cout << "v: validate" << std::endl;
}


pcl::PointCloud<pcl::PointXYZ> get_pc_global(const Plane& mirror, const TaitBryanPose& instrument_pose, const std::vector<DataStream>& data_stream)
{
	pcl::PointCloud<pcl::PointXYZ> pc_global;
	double px = instrument_pose.px;
	double py = instrument_pose.py;
	double pz = instrument_pose.pz;
	double om = instrument_pose.om;
	double fi = instrument_pose.fi;
	double ka = instrument_pose.ka;
	double a = mirror.a;
	double b = mirror.b;
	double c = mirror.c;
	double d = mirror.d;

	for(auto &ds:data_stream){
		double x;
		double y;
		double z;
		Eigen::Vector3d ray(ds.x, ds.y, ds.z);
		double length = ray.norm();
		double angle_rad = ds.angle_rad;

		ray = ray / ray.norm();

		transform_point_rotated_mirror_tait_bryan_wc(x, y, z, px, py, pz, om, fi, ka,
				ray.x(), ray.y(), ray.z(), length,
				a, b, c, d, angle_rad, mirror_cal.px, mirror_cal.py, mirror_cal.pz, mirror_cal.om, mirror_cal.fi, mirror_cal.ka);

		pcl::PointXYZ p;
		p.x = x;
		p.y = y;
		p.z = z;
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




