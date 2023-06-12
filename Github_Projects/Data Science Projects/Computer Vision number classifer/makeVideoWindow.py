import numpy as np
import pyautogui
import cv2
import tkinter
from customtkinter import *
from multiprocessing import Process


set_appearance_mode("Dark") 

win = CTk()
win.title("Snipping Tool")
win.attributes("-alpha", 0.90)
win.geometry("370x200")
win.resizable(False, False)

# we need a seperate semi-transparent window for rectangular snip
board = tkinter.Toplevel(win)
# hide this window at start (we will only open it in snip mode)
board.withdraw()
# we need a GLOBAL canvas object to draw upon
canvas = None
startX,startY=None,None #to track starting pos
curX,curY=None,None #to track mouse cur pos
# we need a rectangle to define the selection snip
snipRect = None   

# hide the main window and open the board
def enterSnipMode():
    global canvas
    win.withdraw()
    board.deiconify()
    canvas = tkinter.Canvas(board,cursor='cross',bg='grey11')
    canvas.pack(fill=BOTH, expand=YES)
    canvas.bind("<ButtonPress-1>", mousePress)
    canvas.bind("<B1-Motion>", mouseMove)
    canvas.bind("<ButtonRelease-1>", mouseRelease)
    board.bind("<Escape>", escapeSnipMode)
    board.lift()
    board.attributes('-fullscreen', True)
    board.attributes('-alpha', 0.25)
    board.attributes("-topmost", True)

def rectSnip(x, y, w, h):
    while True:
		# Take screenshot using PyAutoGUI
        img = pyautogui.screenshot(region=(x, y, w, h))

		# Convert the screenshot to a numpy array
        frame = np.array(img)

		# Convert it from BGR(Blue, Green, Red) to
		# RGB(Red, Green, Blue)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		
		# Optional: Display the recording screen
        cv2.imshow('Live', frame)
		
		# Stop recording when we press 'q'
        if cv2.waitKey(1) == ord('q'):
            break

    
        

def mouseMove(event):
    global snipRect,startX,startY,curX,curY
    startX,startY = (event.x, event.y)
    # expand rectangle as you drag the mouse
    canvas.coords(snipRect, startX, startY, curX, curY)
    return 'break'

def mousePress(event):
    global snipRect,startX,startY,curX,curY
    startX=curX=canvas.canvasx(event.x)
    startY=curY=canvas.canvasy(event.y)
    snipRect = canvas.create_rectangle(startX,startY, 2, 2, outline='red', width=3, fill="white")
    return 'break'

def escapeSnipMode(_):
    canvas.destroy()
    board.withdraw()
    win.deiconify()

def mouseRelease(event):
    global startX,startY,curX,curY
    p1 = Process(target = rectSnip, args = [startX, startY, curX, curY])

    
    # for left-down, right-up, right-down and left-up
    if startX <= curX and startY <= curY:
        p1 = Process(target = rectSnip, args = [startX, startY, curX - startX, curY - startY])

    elif startX >= curX and startY >= curY:
        p1 = Process(target = rectSnip, args = [curX, curY, startX - curX, startY - curY])

    elif startX >= curX and startY <= curY:
        p1 = Process(target = rectSnip, args = [curX, startY, startX - curX, curY - startY])

    elif startX <= curX and startY >= curY:
        p1 = Process(target = rectSnip, args = [startX, curY, curX - startX, startY - curY])

    

    if __name__ == '__main__':
        p1.start()
        

    escapeSnipMode(0)
    return 'break'


#We need this because when multiprocessing each child process imports the parent script, causing this to be rerun
if __name__ == '__main__':

    #GUI Layout

    CTkLabel(master=win,text='').grid(row=0,column=0,columnspan=2)


    CTkButton(master=win, text="Make Window",height=48,command=enterSnipMode).grid(row=3,column=0,columnspan=2,pady=10,padx=30,sticky='we')

    win.mainloop()