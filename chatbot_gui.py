import argparse
from tkinter import *

import tensorflow as tf
from keras.models import load_model
from keras_seq2seq import getAnswer
from sequential_with_intents import getAnswer

global model

def parseArgs():
    """
    Parse the arguments from the given command line
    Args:
        args (list<str>): List of arguments to parse. If None, the default sys.argv will be parsed
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=int, default=1,
                        help='tag to differentiate which model to store/load: '
                             '1 for generative seq2seq, 2 for rule-based rnn with intents')

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
               "Бот: " + "Привет. Я чат-бот Навуходоносор, я учусь на случайных диалогах. Пообщаемся?" + '\n\n')
# ChatBox.config(state=DISABLED)

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

print('Чат-бот 6138 запущен')
print()
print('Версия TensorFlow: v{}'.format(tf.__version__))
args = sys.argv[1:]
selected_model = {
    1: './seq2seq.h5',
    2: './sequential.h5'
}
if len(args) == 2 and args[0] == '-model':
    model = load_model(selected_model.get(args[1]))
else:
    model = load_model(selected_model.get(2))