# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 22:05:20 2020

@author: Arpita Kumari

FACE MASK DETECTOR PROJECT @ Technocolabs
"""


from flask import Flask, render_template, Response, redirect, url_for
from camera import Camera

app = Flask(__name__)
camera  = None

def get_camera():
    global camera
    if not camera:
        camera  = Camera()
    return camera
@app.route('/')
def route():
    return redirect(url_for("index"))

# the starting route
@app.route('/index/')
def index():
    return render_template('untitled2.html')

# the function which generates frame by calling function camera_stream
def gen_frame(camera):
    while True:
        frame = camera.get_feed()
        #frame = camera_stream(frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# function to concat all video frames and display on web-app
@app.route('/video_feed')
def video_feed():
    camera  = get_camera()
    return Response(gen_frame(camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# start the host on local server
if __name__ == '__main__':
    app.run(threaded=False)
