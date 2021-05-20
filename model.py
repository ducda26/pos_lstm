import numpy as np
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import LSTM, Embedding, Dense
from tensorflow.keras.layers import TimeDistributed, SpatialDropout1D, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from livelossplot import PlotLossesKerasTF

def model_lstm(x_train, y_train, max_len, n_words, npos, batch_size, epochs):

    input=Input(shape=(max_len))

    model=Embedding(input_dim=n_words+1,output_dim=153,input_length=max_len)(input)
    model=SpatialDropout1D(0.1)(model)
    model=Bidirectional(LSTM(units=153,return_sequences=True))(model)
    output = TimeDistributed(Dense(npos, activation="softmax"))(model)
    model=Model(input,output)

    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

    early_stop= EarlyStopping(monitor='val_accuracy',patience=1,verbose=0,mode='max',restore_best_weights=False)
    callbacks=[PlotLossesKerasTF(),early_stop]

    history=model.fit(x_train,np.array(y_train),validation_split=0.2,batch_size=batch_size,epochs=epochs,verbose=1,callbacks=callbacks)
    return history, model