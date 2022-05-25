import tensorflow
import numpy as np
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import joblib
import streamlit as st

st.title("Welcome To Gmail Smart Compose Webapp")
#st.write('AS we all know about the E-mail, the E-mail continues to be a ubiquitous and growing form of communication all over the world, with an estimated 3.8 billion users sending 281 billion e-mails daily. All you have experienced the Gmail smart compose, maybe even without knowing you all are actually using the Gmail smart compose in daily life. Maybe you notice when typing an email, sometimes Gmail suggests the relevant or relevant sentences, this is nothing but Gmail smart compose.')
st.write('Problem statement : So here is the task we have to make a model, which can predict the sentences or words based on the given sentence or some words.')
st.markdown("![Alt Text](https://www.androidguys.com/wp-content/uploads/2018/05/Gmail.gif)")

input_words = st.text_input('Enter The Words')

################################################### Loding the Model Structure #########################################################
ita_token = joblib.load('ita_token.pkl')
eng_token = joblib.load('eng_token.pkl')
y_token   = joblib.load('y_token.pkl')

input_vocab  = len(ita_token.word_index) + 1
output_vocab = len(y_token.word_index) + 1

class Encoder(tensorflow.keras.Model):
    '''
    Encoder model -- That takes a input sequence and returns output sequence
    '''

    def __init__(self,inp_vocab_size,embedding_size,lstm_size,input_length):

        #Initialize Embedding layer
        #Intialize Encoder LSTM layer
        super().__init__()
        self.inp_vocab_size = inp_vocab_size
        self.embedding_size = embedding_size
        self.input_length = input_length
        self.lstm_size= lstm_size
        self.embedding = tensorflow.keras.layers.Embedding(input_dim=self.inp_vocab_size, output_dim=self.embedding_size, name="embedding_layer_encoder")
        self.lstm = tensorflow.keras.layers.LSTM(self.lstm_size, return_state=True, return_sequences=True, name="Encoder_LSTM", recurrent_initializer='glorot_uniform')

    def call(self,input_sequence,initial_state):
      '''
          This function takes a sequence input and the initial states of the encoder.
          Pass the input_sequence input to the Embedding layer, Pass the embedding layer ouput to encoder_lstm
          returns -- All encoder_outputs, last time steps hidden and cell state
      '''
      input_embedd = self.embedding(input_sequence)
      lstm_output, lstm_state_h, lstm_state_c = self.lstm(input_embedd,initial_state = initial_state)
      return lstm_output, lstm_state_h, lstm_state_c
    
    def initialize_states(self,batch_size):
      '''
      Given a batch size it will return intial hidden state and intial cell state.
      If batch size is 32- Hidden state shape is [32,lstm_units], cell state shape is [32,lstm_units]
      '''
      return tensorflow.zeros((batch_size, self.lstm_size)) , tensorflow.zeros((batch_size, self.lstm_size))

class Decoder(tensorflow.keras.Model):
    '''
    Encoder model -- That takes a input sequence and returns output sequence
    '''

    def __init__(self,out_vocab_size,embedding_size,lstm_size,input_length):
      super().__init__()
      self.out_vocab_size = out_vocab_size
      self.embedding_size = embedding_size
      self.input_length = input_length
      self.lstm_size= lstm_size

        #Initialize Embedding layer
        #Intialize Decoder LSTM layer

      self.embedding = tensorflow.keras.layers.Embedding(input_dim = self.out_vocab_size, output_dim = self.embedding_size, name="embedding_decoder_layer")
      
      self.lstm = tensorflow.keras.layers.LSTM(self.lstm_size, return_state=True, return_sequences=True, name="Decoder_lstm_layer", recurrent_initializer='glorot_uniform')
        


    def call(self,input_sequence,initial_states):


      
      input_embedd = self.embedding(input_sequence)
      decoder_output, decoder_final_state_h, decoder_final_state_c = self.lstm(input_embedd,initial_state = initial_states)
      return decoder_output, decoder_final_state_h, decoder_final_state_c


class Encoder_decoder(tensorflow.keras.Model):
    
    def __init__(self,encoder_inputs_length,decoder_inputs_length, output_vocab_size):
        super().__init__() # https://stackoverflow.com/a/27134600/4084039
        #(self,inp_vocab_size,embedding_size,lstm_size,input_length):
        self.encoder = Encoder(inp_vocab_size = input_vocab, embedding_size =  256, lstm_size = 200, input_length= encoder_inputs_length)
        self.decoder = Decoder(out_vocab_size = output_vocab, embedding_size = 256, lstm_size = 200, input_length= decoder_inputs_length)
        self.dense   = Dense(output_vocab_size, activation='softmax')
        
        #Create encoder object
        #Create decoder object
        #Intialize Dense layer(out_vocab_size) with activation='softmax'
    
    
    def call(self,data):
        '''
        A. Pass the input sequence to Encoder layer -- Return encoder_output,encoder_final_state_h,encoder_final_state_c
        B. Pass the target sequence to Decoder layer with intial states as encoder_final_state_h,encoder_final_state_C
        C. Pass the decoder_outputs into Dense layer 
        
        Return decoder_outputs
        '''
        input_to_encoder,input_to_decoder = data[0], data[1]
        initial_state= self.encoder.initialize_states(tensorflow.shape(input_to_encoder)[0])
        encoder_output,state_h,state_c = self.encoder(input_to_encoder,initial_state)

        decoder_output, decoder_final_state_h, decoder_final_state_c = self.decoder(input_to_decoder, [state_h,state_c]) 
        decoder_output = self.dense(decoder_output)

        # return the decoder output
        return decoder_output


model  = Encoder_decoder(encoder_inputs_length=31 ,decoder_inputs_length=8 , output_vocab_size = output_vocab)

model.build(input_shape=[(None,31),(None,8)])

#Load weights
model.load_weights('best_model1.h5')

###################################################################### Predictions ########################################################################

#here we change the key value pair to value key pair
eng_word = y_token.word_index
new_dict = dict([(value, key) for key, value in eng_word.items()])
new_dict[0] = 'start'

def predict(input_sentence):
# A. Given input sentence, convert the sentence into integers using tokenizer used earlier
#here we also check the we have the start and end token in sentences
#then we convert out input into to token and do a padding
  input = input_sentence
  if input.split(' ')[0] != '<start>' and input.split(' ')[-1] != '<end>':
    input = '<start>'+ ' ' + input + ' ' + '<end>'
  else:
    input = str(input)
  input = ita_token.texts_to_sequences([str(input)])
  input = pad_sequences(input, padding="post",maxlen= 31)

#as we know we have a three layer 1 ==> encoder 2 ==> decoder 3 ==> dense
#so first is encoder we give the input as input sec and ini_state

  enc_ini_states = model.layers[0].initialize_states(1)
  enc_out, enc_h_state, enc_c_state = model.layers[0](input, enc_ini_states)

#output of the encoder is a input as decoder
# first word input of decoder is start token 
# then 2nd input of decoder is predecting word by model

  decoder_initial_state = [enc_h_state, enc_c_state]
  decoder_initial_input = np.zeros((1,1))
  decoder_initial_input[0,0] = eng_token.word_index['<start>']

  predict_word = []
  w = []
  for i in range(12):
    dec_out, dec_h_state, dec_c_state = model.layers[1](decoder_initial_input, decoder_initial_state)
    # we use 3rd layer and we get the max proba word as out put and the this word is next input of the decoder 
    english_predict = np.argmax(model.layers[2](dec_out).numpy().ravel())
    predict_word.append(english_predict)
    decoder_initial_input[0,0] = english_predict
    #replacing the next decoder initial states with current decoder output 
    decoder_initial_state = [dec_h_state, dec_c_state]
    w.append(new_dict[english_predict])

    if new_dict[english_predict] == '<end>':
      break
  return ' '.join(w)


########################################################## cleaning part##################################################################################
#here we apply all the cleaning things
def data_cleaning(text):

  if text is not None:


    #stopwords_list = stopwords.words('english')
    q = re.sub('Message-ID[^\n]+', ' ', text)
    q = re.sub('Date[^\n]+', ' ', q)
    q = re.sub('X[^\n]+', ' ', q)
    q = re.sub('Content[^\n]+', ' ', q)
    q = re.sub('From[^\n]+', ' ', q)
    q = re.sub('To[^\n]+', ' ', q)
    q = re.sub('Subject[^\n]+', ' ', q)
    q = re.sub('Mime-Version[^\n]+', ' ', q)
    q = re.sub('.*?\(.*?\)',' ',q)
    q = re.sub('--[^\n]+', ' ', q)
    q = re.sub('cc:[^\n]+|Cc:[^\n]+', ' ', q)
    q = re.sub('Sent:[^\n]+', '', q)
    q = re.sub('email address:[^\n]+', ' ', q)
    q = re.sub('[0-9]+', ' ', q)
    q = re.sub('[\/:]', ' ', q)
    q = re.sub('AM|PM|a\.m\.|p\.m\.', ' ', q)
    q = re.sub('\(|\)|,|;|\.|!', ' ', q)
    q = re.sub('-[^\n]+', '', q)
    q = re.sub('-', ' ', q)
    q = re.sub('\?', ' ', q)
    q = re.sub('\@', ' ', q)
    q = re.sub('[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+', ' ', q)
    q = re.sub('(Write to:|From:).*',' ',q)
    q = re.sub('re:', ' ', q)
    q = re.sub('\[[^]]*\]', ' ', q)
    q = re.sub(r'[^\w\s]',' ',q)
    #remove the < anykeyword> like this from the data
    q = re.sub('<.*>',' ',q)
    #remove the all word which is present in the bracket
    q = re.sub('\(.*\)',' ',q)
    #remove all the new lines
    q = re.sub(r'[\n\t-]*','',q)

    #remove all the word which is end with :
    q = re.sub(r'\w+:\s?','',q)

    #replace the short word with there full words code taken from donors choose assignment
            # specific
    q = re.sub(r"won't", " will not", q)
    q = re.sub(r"can\'t", " can not", q)

    # general
    q = re.sub(r"n\'t", " not", q)
    q = re.sub(r"\'re", " are", q)
    q = re.sub(r"\'s", " is", q)
    q = re.sub(r"\'d", " would", q)
    q = re.sub(r"\'ll", " will", q)
    q = re.sub(r"\'t", " not", q)
    q = re.sub(r"\'ve", " have", q)
    q = re.sub(r"\'m", " am", q)

    q = re.sub(r'\b_([a-zA-z]+)_\b',r'\1' , q)
    q = re.sub(r'\b_([a-zA-z]+)\b',r'\1' , q)
    q = re.sub(r'\b([a-zA-z]+)_\b',r'\1' , q)

    #remove _ sign from the word like di_shivam and we want only shivam
    q = re.sub(r'\b[a-zA-Z]{1}_([a-zA-Z]+)',r'\1'  , q)
    q = re.sub(r'\b[a-zA-Z]{2}_([a-zA-Z]+)',r'\1' , q)

    #convert all into a lower case
    q = q.lower()

    #remove the words which are greater than or equal to 15 or less than or equal to 2
    #https://stackoverflow.com/questions/24332025/remove-words-of-length-less-than-4-from-string
    #q = re.sub(r'\b\w{,2}\b' ,'', q)
    q = re.sub(r'\b\w{15,}\b','', q)

    #replace all the words except "A-Za-z_" with space
    q= re.sub(r'[^a-zA-Z_]',' ',q)
    q = q.replace("\n", " ")
#   f = ' '.join([word for word in q.split() if word not in stopwords.words("english")])\
    f = ' '.join([word for word in q.split()])
    return f
###########################################################################################################################################################

a = data_cleaning(input_words)
new = predict(a)
new = ' '.join([i for i in new.split() if i != '<end>'])

##########################################################################################################################################

if st.button("Generate"):
	st.write('The Predicted Words By Model Is')


  #a = data_cleaning(input_words)

	st.success(new)



