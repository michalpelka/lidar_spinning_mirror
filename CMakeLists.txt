cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(mandeye-ls)

set(CMAKE_CXX_STANDARD 11)
set(PROJECT_VERSION 1.0.0)

find_package(OpenGL REQUIRED)
if(OpenGL_FOUND)
	message("OpenGL found")
	message ("OPENGL_LIBRARIES: " ${OPENGL_LIBRARIES})
endif()

find_package(GLUT REQUIRED)
if(GLUT_FOUND)
	message("GLUT found")
	message ("GLUT_glut_LIBRARY: " ${GLUT_glut_LIBRARY})
endif()

find_package( Eigen3 REQUIRED)
if(EIGEN3_FOUND)
	message("EIGEN3 found")
	message ("EIGEN3_INCLUDE_DIR: " ${EIGEN3_INCLUDE_DIR})
	message ("EIGEN3_INCLUDE_DIRS: " ${EIGEN3_INCLUDE_DIRS})
	message ("Eigen3_DIR: " ${Eigen3_DIR})
endif()

find_package(PCL 1.5 REQUIRED)


set(Boost_USE_STATIC_LIBS        OFF)
set(Boost_USE_MULTITHREADED      ON)
set(Boost_USE_STATIC_RUNTIME    OFF)
find_package(Boost REQUIRED COMPONENTS system filesystem thread iostreams) 
if(Boost_FOUND)
    message ("Boost_LIBRARIES: " ${Boost_LIBRARIES})
    message ("Boost_INCLUDE_DIRS: " ${Boost_INCLUDE_DIRS})
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")

find_package(Ceres REQUIRED)
include_directories( ${CERES_INCLUDE_DIRS})

include_directories(
	${EIGEN3_INCLUDE_DIR} 
	${EIGEN3_INCLUDE_DIRS} 
	${Eigen3_DIR}
        ${Boost_INCLUDE_DIRS} 
	${PCL_INCLUDE_DIRS}
	3rd/Sophus include
)

include_directories(3rd/imgui 3rd/imgui/backends ${PCL_INCLUDE_DIRS})
set(ImGui_Glut_src 3rd/imgui/imgui.cpp 3rd/imgui/imgui_demo.cpp 3rd/imgui/imgui_draw.cpp 3rd/imgui/imgui_widgets.cpp
		3rd/imgui/backends/imgui_impl_glut.cpp 3rd/imgui/backends/imgui_impl_opengl2.cpp)



add_executable(calibration
	src/calibration.cpp
	src/utils.cpp
		${ImGui_Glut_src}
)


target_link_libraries(calibration
   	${OPENGL_LIBRARIES}
	${GLUT_glut_LIBRARY}
	${PCL_LIBRARIES}
	${CERES_LIBRARIES}
)

add_executable(validation
	src/validation.cpp
)

target_link_libraries(validation
   	${OPENGL_LIBRARIES}
	${GLUT_glut_LIBRARY}
	${PCL_LIBRARIES}
)




