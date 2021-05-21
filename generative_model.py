import os
import re

import numpy as np
from keras.layers import Input, LSTM, Dense
from keras.models import Model
from keras.models import load_model

# Number of input-response pairs in training dataset
# The more the better, limited only by your computation capacity
# My PC can process ~ 500 pairs, on 1000 it gets stuck
pairs_num = 200
# Dimensionality (number of features)
dimensionality = 256
# The batch size
batch_size = 10
# The number of training epochs, for good quality it should be at least 500
# My PC need ~a day to run 400 epochs
epochs = 60

data_path = "./data/human_text.txt"
data_path2 = "./data/robot_text.txt"
# Defining lines as a list of each line
with open(data_path, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')
with open(data_path2, 'r', encoding='utf-8') as f:
    lines2 = f.read().split('\n')
lines = [re.sub(r"\[\w+]", 'hi', line) for line in lines]
lines = [" ".join(re.findall(r"\w+", line)) for line in lines]
lines2 = [re.sub(r"\[\w+]", '', line) for line in lines2]
lines2 = [" ".join(re.findall(r"\w+", line)) for line in lines2]
# Grouping lines by response pair
pairs = list(zip(lines, lines2))
# random.shuffle(pairs)


input_docs = []
target_docs = []
input_tokens = set()
target_tokens = set()
for line in pairs[:pairs_num]:
    input_doc, target_doc = line[0], line[1]
    # Appending each input sentence to input_docs
    input_docs.append(input_doc)
    # Splitting words from punctuation
    target_doc = " ".join(re.findall(r"[\w']+|[^\s\w]", target_doc))
    # Redefine target_doc below and append it to target_docs
    target_doc = '<START> ' + target_doc + ' <END>'
    target_docs.append(target_doc)

    # Now we split up each sentence into words and add each unique word to our vocabulary set
    for token in re.findall(r"[\w']+|[^\s\w]", input_doc):
        if token not in input_tokens:
            input_tokens.add(token)
    for token in target_doc.split():
        if token not in target_tokens:
            target_tokens.add(token)
input_tokens = sorted(list(input_tokens))
target_tokens = sorted(list(target_tokens))
num_encoder_tokens = len(input_tokens)
num_decoder_tokens = len(target_tokens)

input_features_dict = dict(
    [(token, i) for i, token in enumerate(input_tokens)])
target_features_dict = dict(
    [(token, i) for i, token in enumerate(target_tokens)])

reverse_input_features_dict = dict(
    (i, token) for token, i in input_features_dict.items())
reverse_target_features_dict = dict(
    (i, token) for token, i in target_features_dict.items())

max_encoder_seq_length = max([len(re.findall(r"[\w']+|[^\s\w]", input_doc)) for input_doc in input_docs])
max_decoder_seq_length = max([len(re.findall(r"[\w']+|[^\s\w]", target_doc)) for target_doc in target_docs])

encoder_input_data = np.zeros(
    (len(input_docs), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')
decoder_input_data = np.zeros(
    (len(input_docs), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(input_docs), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')

for line, (input_doc, target_doc) in enumerate(zip(input_docs, target_docs)):
    for timestep, token in enumerate(re.findall(r"[\w']+|[^\s\w]", input_doc)):
        # Assign 1. for the current line, timestep, & word in encoder_input_data
        encoder_input_data[line, timestep, input_features_dict[token]] = 1.

    for timestep, token in enumerate(target_doc.split()):
        decoder_input_data[line, timestep, target_features_dict[token]] = 1.
        if timestep > 0:
            decoder_target_data[line, timestep - 1, target_features_dict[token]] = 1.

print(pairs[:5])
print(input_docs[:5])


# Encoder
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder_lstm = LSTM(dimensionality, return_state=True)
encoder_outputs, state_hidden, state_cell = encoder_lstm(encoder_inputs)
encoder_states = [state_hidden, state_cell]
# Decoder
decoder_inputs = Input(shape=(None, num_decoder_tokens))
decoder_lstm = LSTM(dimensionality, return_sequences=True, return_state=True)
decoder_outputs, decoder_state_hidden, decoder_state_cell = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Model
if not os.path.isfile('generative.h5'):
    training_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    # Compiling
    training_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'],
                           sample_weight_mode='temporal')
    training_model.summary()
    # Training
    training_model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=batch_size,
                       epochs=epochs,
                       validation_split=0.2)
    training_model.save('generative.h5')
    print("model created")

training_model = load_model('generative.h5')

encoder_inputs = training_model.input[0]
encoder_outputs, state_h_enc, state_c_enc = training_model.layers[2].output
encoder_states = [state_h_enc, state_c_enc]

# single encoder input is a question, represented as a sequence
# of integers padded with zeros
encoder_model = Model(encoder_inputs, encoder_states)

latent_dim = 256
# two inputs for the state vectors returned by encoder
decoder_state_input_hidden = Input(shape=(latent_dim,))
decoder_state_input_cell = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_hidden, decoder_state_input_cell]
# these state vectors are used as an initial state
# for LSTM layer in the inference decoder
# third input is the Embedding layer as explained above
decoder_outputs, state_hidden, state_cell = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_hidden, state_cell]
# Dense layer is used to return OHE predicted word
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)


def decode_response(test_input):
    # Getting the output states to pass into the decoder
    states_value = encoder_model.predict(test_input)
    # Generating empty target sequence of length 1
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Setting the first token of target sequence with the start token
    target_seq[0, 0] = target_features_dict['<START>']

    # A variable to store our response word by word
    decoded_sentence = ''

    stop_condition = False

    while not stop_condition:
        # feed the state vectors and 1-word target sequence
        # to the decoder to produce predictions for the next word
        output_tokens, hidden_state, cell_state = decoder_model.predict([target_seq] + states_value)
        # sample the next word using these predictions
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_features_dict[sampled_token_index]
        # append the sampled word to the target sequence
        for word, index in target_features_dict.items():
            if sampled_token_index == index:
                if word != '<END>':
                    decoded_sentence += ' {}'.format(sampled_token)
                sampled_token = word
        # repeat until we generate the end-of-sequence word 'end'
        # or we hit the length of answer limit
        if (sampled_token == '<END>' or len(decoded_sentence.split()) > max_decoder_seq_length):
            stop_condition = True
        # Update the target sequence
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0] = sampled_token_index
        # Update states
        states_value = [hidden_state, cell_state]
    return decoded_sentence


class ChatBot:
    negative_responses = ("нет", "не-а", "ну нет", "нееее", "никак", "извини")
    exit_commands = ("выход", "пауза", "закрыть", "пока", "до свидания", "до скорого", "стоп")

    # Method to start the conversation
    def start_chat(self):
        user_response = input("Привет. Я чат-бот Навуходоносор, я учусь на случайных диалогах. "
                              "Пообщаемся?\n")

        if user_response in self.negative_responses:
            print("Ок, до новых встреч!")
            return
        self.chat(user_response)

    # Method to handle the conversation
    def chat(self, reply):
        while not self.make_exit(reply):
            reply = input(self.generate_response(reply) + "\n")

    # Method to convert user input into a matrix
    def string_to_matrix(self, user_input):
        tokens = re.findall(r"[\w']+|[^\s\w]", user_input)
        user_input_matrix = np.zeros(
            (1, max_encoder_seq_length, num_encoder_tokens),
            dtype='float32')
        for timestep, token in enumerate(tokens):
            if token in input_features_dict:
                user_input_matrix[0, timestep, input_features_dict[token]] = 1.
        return user_input_matrix

    # Method that will create a response using seq2seq model we built
    def generate_response(self, user_input):
        if user_input in self.negative_responses:
            print("Ок, до новых встреч!")
            return
        input_matrix = self.string_to_matrix(user_input)
        chatbot_response = decode_response(input_matrix)
        # Remove <START> and <END> tokens from chatbot_response
        chatbot_response = chatbot_response.replace("<START>", '')
        chatbot_response = chatbot_response.replace("<END>", '')
        return chatbot_response

    # Method to check for exit commands
    def make_exit(self, reply):
        for exit_command in self.exit_commands:
            if exit_command in reply:
                print("Всего хорошего!")
                return True
        return False

def getAnswer(msg):
    chatbot.generate_response(user_input=msg)

chatbot = ChatBot()
#chatbot.start_chat()
