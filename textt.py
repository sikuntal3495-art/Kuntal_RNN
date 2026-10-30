import numpy as np
import tensorflow as tlf 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Input,Embedding,SimpleRNN

sentences = [
    # Positive sentences (15)
    "I love this movie",
    "This product is amazing",
    "I am very happy",
    "The food tastes great",
    "Today is a wonderful day",
    "I really enjoyed it",
    "The service is excellent",
    "This is a beautiful place",
    "I feel very excited",
    "The experience was fantastic",
    "I like this song",
    "Everyone is very helpful",
    "The movie was brilliant",
    "I had a great time",
    "This makes me smile",

    # Negative sentences (15)
    "I hate this movie",
    "This product is terrible",
    "I am very sad",
    "The food tastes awful",
    "Today is a horrible day",
    "I really disliked it",
    "The service is poor",
    "This is an ugly place",
    "I feel very disappointed",
    "The experience was terrible",
    "I dislike this song",
    "Everyone is very rude",
    "The movie was boring",
    "I had a bad time",
    "This makes me cry"
]

labels = [1]*15 + [0]*15

labels = np.array(labels)

labels=[1]*15+[0]*15
labels=np.array(labels)

vocabulary_size=2000
toki=Tokenizer(num_words=vocabulary_size,oov_token="<oov>")
toki.fit_on_texts(sentences)
seqs=toki.texts_to_sequences(sentences)
maxlen=max(len(s) for s in seqs)
input_data=pad_sequences(
    seqs,
    maxlen=maxlen,
    padding="post"
)

y=labels

print(maxlen)
print(input_data[29])
print(y)

embend_dim=16
rnn_utils=8
imp=Input(shape=(maxlen,),dtype="int32",name="input")

x=Embedding(input_dim=vocabulary_size,output_dim=embend_dim,mask_zero=True,name="embedding")(imp)

rnn=SimpleRNN(units=rnn_utils,return_sequences=False,return_state=False,name="rnn")
x_last=rnn(x)
out=Dense(1,activation="sigmoid",name="output")(x_last)
Mod=Model(inputs=imp,outputs=out)
Mod.compile(
    optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"]
)
Mod.summary()

Mod.fit(input_data,y,epochs=25,batch_size=8,verbose=1)

test_sentence=["i hate movie"]
test_seq=toki.texts_to_sequences(test_sentence)
text_padding=pad_sequences(
    test_seq,
    maxlen=maxlen,
    padding="post"
)
prediction=Mod.predict(text_padding)
print("prediction score:",prediction[0][0])
if(prediction[0][0]>=0.5):
    print("the sentence is positive")
else:
    print("the sentence is negative")