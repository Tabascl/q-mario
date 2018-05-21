from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten

class Brain:
    def __init__(self, state_cnt, action_cnt, img_stack, img_width, img_height):
        self.state_cnt = state_cnt
        self.action_cnt = action_cnt
        self.img_stack = img_stack
        self.img_width = img_width
        self.img_height = img_height

        self.model = self._create_model()
        self._model = self._create_model() # target network

    def _create_model(self):
        model = Sequential()

        model.add(Conv2D(32, (8,8), strides=(4,4), activation='relu', input_shape=(self.state_cnt), data_format='channels_first'))
        model.add(Conv2D(64, (4,4), strides=(2,2), activation='relu'))
        model.add(Conv2D(64, (3,3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_cnt, activation='linear'))

        model.compile('adam', 'mse')

        return model

    def train(self, x, y, epochs=1):
        self.model.fit(x, y, batch_size=32, epochs=epochs, verbose=0)

    def predict(self, s, target=False):
        if target:
            return self._model.predict(s)
        else:
            return self.model.predict(s)

    def predict_one(self, s, target=False):
        return self.predict(s.reshape(1, self.img_stack, self.img_width, self.img_height), target).flatten()

    def update_target_model(self):
        self._model.set_weights(self.model.get_weights())
