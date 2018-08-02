import cv2
import sys, datetime
from time import sleep


def draw_boxes(frame, boxes, color=(0,255,0)):
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), color, 2)
    return frame

def resize_image(image, size_limit=500.0):
    max_size = max(image.shape[0], image.shape[1])
    if max_size > size_limit:
        scale = size_limit / max_size
        _img = cv2.resize(image, None, fx=scale, fy=scale)
        return _img
    return image

class FaceDetect():

    def __init__(self, cascPath="./haarcascade_frontalface_default.xml"):
        self.faceCascade = cv2.CascadeClassifier(cascPath)

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        return faces

class FaceTracker():
    
    def __init__(self, frame, face):
        (x,y,w,h) = face
        self.face = (x,y,w,h)
        self.tracker = cv2.TrackerKCF_create()
        self.tracker.init(frame, self.face)
    
    def update(self, frame):
        self.face = self.tracker.update(frame)
        return self.face


# update states
DETECT_INTERVAL= 6
LAST_DETECT = datetime.datetime.now()

def should_run_detector():
    current = datetime.datetime.now()
    seconds = (current - LAST_DETECT).seconds
    should = seconds > DETECT_INTERVAL
    # print("LAST: %s, current: %s, seconds: %s, should: %s" % (LAST_DETECT, current, seconds, should))
    return should


def run():
    video_capture = cv2.VideoCapture(0)

    # exit if video not opened
    if not video_capture.isOpened():
        print('Cannot open video')
        sys.exit()
    
    # read first frame
    ok, frame = video_capture.read()
    if not ok:
        print('Error reading video')
        sys.exit()

    # init detector    
    detector = FaceDetect()

    def get_faces_trackers(frame):
        print("running detector...")
        # get faces 
        faces = detector.detect(frame)

        global LAST_DETECT
        LAST_DETECT = datetime.datetime.now()

        # get trackers
        # trackers = [FaceTracker(frame, face) for face in faces]
        trackers = []
        for face in faces:
            print("adding tracker for face: ", face)
            (x,y,w,h) = face
            tracker = cv2.TrackerKCF_create()
            tracker.init(frame, (x,y,w,h))
            trackers.append(tracker)
        return faces, trackers

    # start detection
    # read a couple frames to cold-start
    faces = None
    while faces is None:
        _, frame = video_capture.read()
        faces, trackers = get_faces_trackers(frame)
        print("cold start: %s" % faces)

    draw_boxes(frame, faces)
    
    ##
    ## main loop
    ##
    while True:
        # Capture frame-by-frame
        _, frame = video_capture.read()

        # check if tracking or detect
        if should_run_detector():
            faces, trackers = get_faces_trackers(frame)
            draw_boxes(frame, faces)
        else:
            boxes = [t.update(frame)[1] for t in trackers]
            print("tracking: %s" % boxes)
            draw_boxes(frame, boxes, (255,0,0)) # draw blue for tracking

        # Display the resulting frame
        cv2.imshow('Video', frame)

        # quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run()