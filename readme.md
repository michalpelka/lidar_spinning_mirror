# Lidar spinning mirror

## Repository goal
At this moment repository contains C++ code with optimizer that optimizes a geometry 
of the Livox Mid-40 LiDAR's field of view. Prototype of the device allows to spin a 
reflector with speed from 1RPM to 120RPM. Device has time synchronization that enables 
precise measurment. With presented method of calibration a fesh apporach for LiDAR's FOV 
reshaping is presented.


## Demo 5 RPM
![Demo 5 rpm](doc/railway_slow.gif)
[Download pcd, 5 RPM](https://storage.googleapis.com/dataset_sensors_pub/pcd_samples/railway_slow.pcd)

## Demo 120 RPM
![Demo 120 rpm](doc/parking_fast.gif)
[Download pcd, 120 RPM](https://storage.googleapis.com/dataset_sensors_pub/pcd_samples/parking_fast_1sec.pcd)
[Download pcd, 5 RPM](https://storage.googleapis.com/dataset_sensors_pub/pcd_samples/parking_fast_1sec.pcd)

# Mechanical design 
![Photo](doc/system_photo.jpg)

System is build with a tilted reflectorthat spins. The angle is read by a contacless encoder.
An electronic PCB produce synchronization signal for Livox Mid-40.

## Plans
 - Develop open hardware version of calibrated system using more avaible components (e.g. clasical encoder and DC motor and widely aviable STM32 board )
 - Release firmware and ROS-node
