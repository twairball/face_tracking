# Face Tracking Demo

Detect and track faces from a webcam (required).

## Install

`pip install -r requirements.txt`

## Run

Run main program, you should see webcam screen, with Green or Blue box for faces detected. 

Blue = Detected face 
Green = Box from previous detection, updated via tracking. 

````
python face_tracking.py`
````

Optionally use `-i` argument to set different detection intervals, in seconds. Default=6.

````
python face_tracking.py -i 3
````

## Sample

![sample](./sample.gif)


# LICENSE

MIT


