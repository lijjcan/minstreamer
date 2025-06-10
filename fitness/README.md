# Fitness Motion Quality Evaluation

This example demonstrates a simple Qt and OpenCV application for
evaluating fitness poses. The GUI allows loading an image, detects joint
positions with the OpenPose model from OpenCV's DNN module and evaluates
the motion.

The code is intended as a starting point and requires OpenCV with DNN
support as well as Qt5. You will need the OpenPose `.prototxt` and
`.caffemodel` files.

## Build

```
make
```

## Usage

```
./fitness_app <proto> <model>
```

where `<proto>` is the path to the OpenPose prototxt file and `<model>` is
the path to the caffe weight file. After launching, click **Open Image** to
select a picture to analyse. The detected joints and an elbow angle
evaluation are drawn on the image.
