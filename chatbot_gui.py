import argparse
from tkinter import *

import tensorflow as tf
from keras.models import load_model

from generative_model import getAnswer
#from generative_model_eng import getAnswer
#from retrieval_model import getAnswer

selected_model = {
    1: './retrieval.h5',
    2: './generative.h5',
    3: './generative_eng.h5'
}
#model = load_model(selected_model.get(1))
model = load_model(selected_model.get(2))

def send():
    msg = EntryBox.get("1.0", 'end-1c').strip()
    EntryBox.delete("0.0", END)

    if msg != '':
        ChatBox.config(state=NORMAL)
        ChatBox.insert(END, "Вы: " + msg + '\n\n')
        ChatBox.config(foreground="#446665", font=("Verdana", 12))

        res = getAnswer(msg)

        ChatBox.insert(END, "Бот: " + res + '\n\n')

        ChatBox.config(state=DISABLED)
        ChatBox.yview(END)


print('Чат-бот 6138 запущен')
print()
print('Версия TensorFlow: v{}'.format(tf.__version__))

# Creating tkinter GUI
root = Tk()
root.title("6138 Чат-бот (Плешаков, Ширяева)")
root.geometry("400x500")
icon = PhotoImage(file='./data/krasnyy-samoletik-64x48.png')
root.tk.call('wm', 'iconphoto', root._w, icon)
root.resizable(width=False, height=False)

# Create Chat window
ChatBox = Text(root, bd=0, bg="white", height="8", width="50", font="Arial", )
ChatBox.insert(END,
               "Бот: " + "Привет. Я чат-бот Навуходоносор, " + '\n\n' +
                         "я учусь на случайных диалогах. Пообщаемся?" + '\n\n')

# Bind scrollbar to Chat window
scrollbar = Scrollbar(root, command=ChatBox.yview, cursor="heart")
ChatBox['yscrollcommand'] = scrollbar.set

# Create Button to send message
SendButton = Button(root, image=icon, bd=0, fg='#000000', command=send)

# Create the box to enter message
EntryBox = Text(root, bd=0, bg="white", width="29", height="5", font="Arial")
# EntryBox.bind("<Return>", send)

# Place all components on the screen
scrollbar.place(x=376, y=6, height=386)
ChatBox.place(x=6, y=6, height=386, width=370)
EntryBox.place(x=6, y=401, height=90, width=300)
SendButton.place(x=312, y=401, height=90)
root.mainloop()


