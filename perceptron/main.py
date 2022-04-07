from cgitb import small
from tkinter import *
from PIL import Image, ImageTk
import numpy as np

root = Tk()
root.title("Vowel Recognition")
root.geometry("700x350")

def paint(event):
    x1, y1 = (event.x-10), (event.y-10)
    x2, y2 = (event.x+10), (event.y+10)
    color = "black"
    wn.create_oval(x1, y1, x2, y2, fill=color)

def save_png(canvas, fileName):
    canvas.update()
    canvas.postscript(file=fileName, colormode='color')
    img = Image.open(fileName)
    # resize image to 35x35
    # small_img = img.resize((35, 35), Image.BILINEAR)
    # res = small_img.resize((200, 200), Image.NEAREST)
    res = img.resize((35, 35), Image.Resampling.NEAREST)
    res.save("vowel_recognition.png", "PNG")

    imge = Image.open('vowel_recognition.png')
    imge = imge.resize((100, 100), Image.Resampling.NEAREST)
    photo = ImageTk.PhotoImage(imge)
    label = Label(root, image=photo)
    label2 = Label(root, text="Input 35x35")
    label.image = photo
    label.place(x=300, y=20)
    label2.place(x=300, y=120)

    imge = Image.open('vowel_recognition.png')
    grayImage = imge.convert('L')
    array = np.array(grayImage)
    array = np.where(array > 0, 0, 1)
    array = array.flatten()

def clear_canvas(event):
    wn.delete("all")


wn = Canvas(root, width=250, height=250, bg='white')
btn = Button(root, text="Submit", command=lambda: save_png(wn, "vowel_recognition.ps"))
clr_btn = Button(root, text="Clear", command=lambda: clear_canvas(wn))
wn.bind('<B1-Motion>', paint)

wn.place(x=20, y=20)
btn.place(x=190, y=280)
clr_btn.place(x=120, y=280)

root.mainloop()


# perceptron