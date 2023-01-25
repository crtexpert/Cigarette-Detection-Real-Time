import imp
import os.path
import time
from tkinter.ttk import *
import tkinter as tk
import importlib
import customtkinter
import customtkinter as ctk
from os import startfile
from matplotlib import pyplot as plt
import torch
import numpy as np
import cv2
from PIL import Image, ImageTk
from tkinter import filedialog

app = tk.Tk()
app.geometry("1280x1080")
app.title("Cigarette Detection App")
ctk.set_appearance_mode("dark")
imp.reload(torch.hub)
homePageImage = Image.open("icon/homePageIcon.png")
resized_image = homePageImage.resize((700,700), Image.ANTIALIAS)
newHomePageImage = ImageTk.PhotoImage(resized_image)
iconImage = tk.PhotoImage(file = 'icon/smoke-detector.png')
app.iconphoto(False, iconImage)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/exp5/weights/last.pt',verbose=False,force_reload=True)

def home_page():

    home_frame = tk.Frame(content_frame)
    lb = tk.Label(home_frame, image=newHomePageImage)
    lb.pack()
    home_frame.pack(expand=True)

def indicate(page):
    delete_page()
    page()
#
#   Menubar conf.
#

menuBar_frame = tk.Frame(app, bg='#507A5A')
menuBar_frame.pack(side=tk.LEFT)
menuBar_frame.pack_propagate(False)
menuBar_frame.configure(width=180, height=1080)

#
#   Button conf.
#

#Home Button

home_btn = tk.Button(menuBar_frame, text='Home', font=(("Times New Roman"), 15, 'italic'),
                     fg='#FFFFFF', bd=4, bg='#476B4F',command=lambda: indicate(home_page))
home_btn.config(width=15, height=3)
home_btn.place(x=0, y=0)

#Real Time Detection Button

rtd_btn = tk.Button(menuBar_frame, text='Real-Time Detection', font=(("Times New Roman"), 15, 'italic'),
                     fg='#FFFFFF', bd=4, bg='#476B4F',command=lambda: indicate(realTimeDetection_page))
rtd_btn.config(width=15, height=3)
rtd_btn.place(x=0, y=85)

#Video Detection Button

vd_btn = tk.Button(menuBar_frame, text='Video Detection', font=(("Times New Roman"), 15, 'italic'),
                     fg='#FFFFFF', bd=4, bg='#476B4F',command=lambda: indicate(videoDetection_page))
vd_btn.config(width=15, height=3)
vd_btn.place(x=0, y=170)

#
#  Content frame kontrolleri.
#
content_frame = tk.Frame(app, highlightbackground='#9DE3B1',
                              highlightthickness=3, )
content_frame.pack(side=tk.LEFT)
content_frame.pack_propagate(False)
content_frame.configure(width=1100, height=1080)


def delete_page():
    for frame in content_frame.winfo_children():
        frame.destroy()


def realTimeDetection_page():

    videoFrame = tk.Frame(content_frame)


    video = ctk.CTkLabel(videoFrame, text=None)
    video.pack()

    camera = cv2.VideoCapture(0)
    videoFrame.pack(expand=True)


    def realTimeDetectFunc():
        ptime = 0
        if videoFrame.winfo_exists() == False:
            cv2.releaseCapture()
            return
        else:
            ret, frame = camera.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(frame)
            detections = results.pandas().xyxy[0].to_dict(orient="records")
            if len(results.xywh[0]) > 0:
                detectionConf = results.xywh[0][0][4]
                detectionNum = results.xywh[0][0][5]
                if detectionNum == 0 and detectionConf > 0.1:

                    for detection in detections:
                        x = int(detection['xmin'])
                        w = int(detection['xmax'])
                        y = int(detection['ymin'])
                        h = int(detection['ymax'])

                        frame[y:h, x:w] = cv2.GaussianBlur(frame[y:h, x:w], (5001, 5001), cv2.BORDER_DEFAULT)
            cTime = time.time()
            fpsx = 1 / (cTime - ptime)
            ptime = cTime
            image = np.squeeze(results.render())
            imageArray = Image.fromarray(image)
            imageTk    = ImageTk.PhotoImage(imageArray)
            video.imageTk = imageTk
            video.configure(image=imageTk)
            video.after(3, realTimeDetectFunc)

    realTimeDetectFunc()

def videoDetection_page():

    videoFrame = tk.Frame(content_frame)
    video = ctk.CTkLabel(videoFrame, text=None)
    video.pack()
    videoFrame.pack(expand=True)
    selectButton = tk.Button(content_frame, text='Select File', font=(("Times New Roman"), 15, 'italic'),
                         fg='#FFFFFF', bd=4, bg='#476B4F', command=lambda: openvideo())
    selectButton.config(width=15, height=3)
    selectButton.place(x=content_frame.winfo_width()/2-90,y=20)

    def openvideo():
        file = filedialog.askopenfile(initialdir="/videos", title="Select A Video File", filetypes=(("mp4 files", "*.mp4"), ("all files", "*.*")))
        if file:
            filepath = os.path.abspath(file.name)
            detectvideo(filepath)

    def detectvideo(filename):
        camera = cv2.VideoCapture(filename)
        bar = Progressbar(content_frame, orient=tk.HORIZONTAL, length=content_frame.winfo_width()-40)
        bar.place(x=20, y=content_frame.winfo_height()/2)
        percentage = tk.StringVar()
        percentage.set(" ")
        percentageLabel = tk.Label(content_frame, textvariable=percentage)
        percentageLabel.place(x=30, y=content_frame.winfo_height()/2 - 20)
        height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = camera.get(cv2.CAP_PROP_FPS)
        videoWriter = cv2.VideoWriter(os.path.join('output_video', 'output.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        def realTimeDetectFunc():
            ptime = 0
            index = 0
            for frameid in range(int(camera.get(cv2.CAP_PROP_FRAME_COUNT))):
                ret, frame = camera.read()
                results = model(frame)
                results.hide_conf = True
                results.hide_labels = True
                detections = results.pandas().xyxy[0].to_dict(orient="records")
                if len(results.xywh[0]) > 0:
                    detectionConf = results.xywh[0][0][4]
                    detectionNum = results.xywh[0][0][5]
                    if detectionNum == 0 and detectionConf > 0.1:

                        for detection in detections:
                            x = int(detection['xmin'])
                            w = int(detection['xmax'])
                            y = int(detection['ymin'])
                            h = int(detection['ymax'])

                            frame[y:h, x:w] = cv2.GaussianBlur(frame[y:h, x:w], (5001, 5001), cv2.BORDER_DEFAULT)

                #cv2.imshow('CigaretteDetection', np.squeeze(results.render()))

                videoWriter.write(np.squeeze(results.render()))
                index+=1
                bar['value'] += 100 / int(camera.get(cv2.CAP_PROP_FRAME_COUNT))
                percentage.set(str(int((index / camera.get(cv2.CAP_PROP_FRAME_COUNT)) *100))+"%")
                videoFrame.update_idletasks()
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            percentage.set("Done.")
            camera.release()
            videoWriter.release()
            startfile(os.path.join('output_video', 'output.mp4'))


        realTimeDetectFunc()
app.mainloop()
