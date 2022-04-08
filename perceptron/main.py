import datetime
import shutil
from tkinter import *
from PIL import Image, ImageTk
import numpy as np
import os

from ml.Perceptron import Perceptron

root = Tk()
root.title("Vowel Recognition")
root.geometry("500x350")

perceptron = Perceptron()
perceptron.update()

def open_popup():
    top = Toplevel()
    top.title("Output Correction")
    top.geometry("400x150")
    label = Label(top, text="Select the correct vowel identification:", font=('Helvetica 12'))
    v_btn = Button(top, text="vowel", font=('Helvetica 14 bold'), command=lambda: update_and_save(top, "vowel_recognition.png","1-"+ str(datetime.datetime.now()) + ".png"))
    nv_btn = Button(top, text="not a vowel", font=('Helvetica 14 bold'), command=lambda: update_and_save(top, "vowel_recognition.png","0-"+ str(datetime.datetime.now()) + ".png"))
    label.pack(pady=20)
    v_btn.place(x=90, y=60)
    nv_btn.place(x=200, y=60)

# on submit button click show pop up output
def incorrect():
    open_popup()
    perceptron.update()

def submit():
    output_value.text = ""
    save_png(wn, "vowel_recognition.ps")
    img = Image.open("vowel_recognition.png")
    img = img.convert('L')
    data_img = np.array(img)
    data_img = np.where(data_img > 0, 0, 1)
    data_img = data_img.flatten()
    output_txt = None
    if(perceptron.predict(data_img)):
        output_txt = "vowel"
    else:
        output_txt = "not a vowel"
    output_value.config(text=output_txt)
    

def save_png(canvas, fileName):
    canvas.update()
    canvas.postscript(file=fileName, colormode='color')
    img = Image.open(fileName)
    res = img.resize((35, 35), Image.Resampling.NEAREST)
    res.save("vowel_recognition.png", "PNG")
    update_img_label("vowel_recognition.png")
    
def update_img_label(path):
    img = Image.open(path)
    img = img.resize((100, 100), Image.Resampling.NEAREST)
    photo = ImageTk.PhotoImage(img)
    img_label.configure(image=photo)
    img_label.image = photo



def update_and_save(event, existing_file, new_file, dir_name="ml/img-datasets/", ):
    shutil.copy(existing_file, dir_name + new_file)
    event.destroy()

def paint(event):
    x1, y1 = (event.x-15), (event.y-15)
    x2, y2 = (event.x+15), (event.y+15)
    color = "black"
    wn.create_oval(x1, y1, x2, y2, fill=color)

def clear_canvas(event):
    wn.delete("all")


wn = Canvas(root, width=250, height=250, bg='white')
btn = Button(root, text="Submit", command=lambda: submit())
incorrect_button = Button(root, text="Incorrect", command=lambda: incorrect())
clr_btn = Button(root, text="Clear", command=lambda: clear_canvas(wn))
output_label = Label(root, text="Output: ")
output_value = Label(root, text="", font=('Helvetica 14 bold'))
txt_label = Label(root, text="Input 35x35")
img_label = Label(root)
    

wn.bind('<B1-Motion>', paint)
wn.place(x=20, y=20)
output_label.place(x=300, y=150)
output_value.place(x=300, y=200)
txt_label.place(x=300, y=125)
img_label.place(x=300, y=20)

btn.place(x=190, y=280)
clr_btn.place(x=120, y=280)
incorrect_button.place(x=300, y=280)
root.mainloop()