class FeedForwardNeuralNetwork:
  def __init__(self):
    self.W1 = None
    self.B1 = None
    self.W2 = None
    self.B2 = None
    self.W3 = None
    self.B3 = None
    self.__learning_rate = 0.08
    self.epochs = 3000

  def __sigmoid(self, z):
    return 1.0/ (1.0 + np.exp(-z))

  def __pre_activation(self, W, B, X):
    return np.dot(W, X) + B

  def __softmax(self, z):
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

  def __dsoftmax(self, Y, A3):
    return A3 - Y

  def __dsigmoid(self, z):
    return z * (1 - z)

  def __forward_prop(self, X, W1, B1, W2, B2, W3, B3):
    H1 = self.__pre_activation(W1.T, B1, X.T)
    A1 = self.__sigmoid(H1)
    H2 = self.__pre_activation(W2, B2, A1)
    A2 = self.__sigmoid(H2)
    H3 = self.__pre_activation(W3.T, B3, A2)
    A3 = self.__softmax(H3)

    return H1, A1, H2, A2, H3, A3

  def __back_prop(self, X, Y, W1, B1, H1, A1, W2, B2, H2, A2, W3, B3, H3, A3):
    m = X.shape[1]  # number of samples

    # Output layer
    dH3 = self.__dsoftmax(Y, A3)
    dW3 = np.dot(dH3, A2.T) / m
    dB3 = np.sum(dH3, axis=1, keepdims=True) / m

    # Hidden layer 2
    dA2 = np.dot(W3, dH3)
    dH2 = dA2 * self.__dsigmoid(A2)
    dW2 = np.dot(dH2, A1.T) / m
    dB2 = np.sum(dH2, axis=1, keepdims=True) / m

    # Hidden layer 1
    dA1 = np.dot(W2, dH2)
    dH1 = dA1 * self.__dsigmoid(A1)
    dW1 = np.dot(dH1, X.T) / m
    dB1 = np.sum(dH1, axis=1, keepdims=True) / m

    # Update weights
    W1 -= self.__learning_rate * dW1.T
    B1 -= self.__learning_rate * dB1
    W2 -= self.__learning_rate * dW2
    B2 -= self.__learning_rate * dB2
    W3 -= self.__learning_rate * dW3.T
    B3 -= self.__learning_rate * dB3

    return (W1, B1, W2, B2, W3, B3)

  def __cross_entropy(self, Y, A3):
    m = Y.shape[1]
    epsilon = 1e-8
    loss = -np.sum(Y * np.log(A3 + epsilon)) / m
    return loss

  def fit(self, x_train, y_train):
    # Random Initialization
    W1 = np.random.randn(784, 32) * 1 / np.sqrt(784)
    W2 = np.random.randn(32, 32) * 1 / np.sqrt(32)
    W3 = np.random.randn(32, 10) * 1 / np.sqrt(32)
    B1 = np.random.randn(32, 1)
    B2 = np.random.randn(32, 1)
    B3 = np.random.randn(10, 1)

    for i in range(self.epochs):
      H1, A1, H2, A2, H3, A3 = self.__forward_prop(x_train, W1, B1, W2, B2, W3, B3)

      if i % 100 == 0:
        print(f"Epoch {i}: Loss = {self.__cross_entropy(y_train, A3)}")

      W1, B1, W2, B2, W3, B3 = self.__back_prop(x_train.T, y_train, W1, B1, H1, A1, W2, B2, H2, A2, W3, B3, H3, A3)

    # Save weights and biases
    self.W1 = W1
    self.B1 = B1
    self.W2 = W2
    self.B2 = B2
    self.W3 = W3
    self.B3 = B3

  def predict(self, x_test):
    W1, W2, W3 = self.W1, self.W2, self.W3
    B1, B2, B3 = self.B1, self.B2, self.B3

    H1 = self.__pre_activation(W1.T, B1, x_test.T)
    A1 = self.__sigmoid(H1)

    H2 = self.__pre_activation(W2, B2, A1)
    A2 = self.__sigmoid(H2)

    H3 = self.__pre_activation(W3.T, B3, A2)
    A3 = self.__softmax(H3)

    y_pred = np.argmax(A3, axis=0)
    return y_pred

  def score(self, x_test, y_test):
    y_pred = self.predict(x_test)
    y_true = np.argmax(y_test, axis=1)

    accuracy = np.mean(y_pred == y_true)
    return (f"{accuracy * 100}%")