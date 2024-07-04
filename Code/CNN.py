def cnn_model():
  model = Sequential()
  model.add(Conv2D(16, (3, 3),strides=(1,1),input_shape=(20,208,1), kernel_initializer='he_normal',padding='same'))#input_shape=(20,max div ,class no to classify)
  # model.add(Conv2D(16, (3, 3), strides=(1, 1), input_shape=(20, 208, 1), kernel_initializer='he_normal', padding='same', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))

  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPool2D(pool_size=(2, 2),strides=(2, 2)))
  model.add(Dropout(0.25))

  model.add(Conv2D(32, (3, 3),strides=(1,1),padding='same'))
  #model.add(Activation('relu'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPool2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))

  model.add(Conv2D(64, (3, 3),strides=(1,1),padding='same'))
  #model.add(Activation('relu'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPool2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))

  model.add(Conv2D(128, (3, 3),strides=(1,1),padding='same'))
  #model.add(Activation('relu'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPool2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))

  model.add(Conv2D(256, (3, 3),strides=(1,1),padding='same'))
  #model.add(Activation('relu'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  # model.add(MaxPool2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))

  model.add(Flatten())

  model.add(Dense(128,activation='relu'))
  model.add(Dense(64,activation='relu'))
  model.add(Dense(16,activation='relu'))
  model.add(Dense(1, activation='sigmoid'))


  opt = Adam(lr=0.0001)
  model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
  return model
