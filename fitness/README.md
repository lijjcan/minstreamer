# Fitness Motion Quality Evaluation

This example demonstrates a simple C++ application using OpenCV for pose
estimation. It captures frames from a camera, detects joint positions with
the OpenPose model provided in OpenCV's DNN module and evaluates the motion.

The code is intended as a starting point and requires OpenCV with DNN
support. You will need the OpenPose `.prototxt` and `.caffemodel` files.

## Build

```
make
```

## Usage

```
./fitness_app <proto> <model>
```

where `<proto>` is the path to the OpenPose prototxt file and `<model>` is
the path to the caffe weight file.
