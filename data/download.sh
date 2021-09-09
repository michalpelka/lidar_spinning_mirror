#! /bin/sh

wget https://storage.googleapis.com/dataset_sensors_pub/ground_truth.tar.xz
tar -xf ground_truth.tar.xz
wget https://storage.googleapis.com/dataset_sensors_pub/rotated_mirror_livox.tar.xz
tar -xf rotated_mirror_livox.tar.xz

rm -r *.tar.xz