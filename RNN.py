import numpy as np
import random

class SimpleRNNNumpy:
    def __init__(self, hidden_size, learning_rate=0.01):
        self.vocab_size = 0
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        self.word_to_idx = {}
        self.idx_to_word = {}

        self.Wxh = None
        self.Whh = None
        self.Why = None
        self.bh = None
        self.by = None

    def build_vocab(self, text):
        words = text.split()
        unique_words = sorted(list(set(words)))
        self.word_to_idx = {w: i for i, w in enumerate(unique_words)}
        self.idx_to_word = {i: w for i, w in enumerate(unique_words)}
        self.vocab_size = len(unique_words)

        scale_xh = np.sqrt(1.0 / self.vocab_size)
        scale_hh = np.sqrt(1.0 / self.hidden_size)
        scale_hy = np.sqrt(1.0 / self.hidden_size)

        self.Wxh = np.random.randn(self.hidden_size, self.vocab_size) * scale_xh
        self.Whh = np.random.randn(self.hidden_size, self.hidden_size) * scale_hh
        self.Why = np.random.randn(self.vocab_size, self.hidden_size) * scale_hy
        self.bh = np.zeros((self.hidden_size, 1))
        self.by = np.zeros((self.vocab_size, 1))
        print(f"Vocabulary built. Size: {self.vocab_size}")
        print(f"Weights initialized: Wxh={self.Wxh.shape}, Whh={self.Whh.shape}, Why={self.Why.shape}")

    def one_hot_encode(self, word_idx):
        vec = np.zeros((self.vocab_size, 1))
        vec[word_idx] = 1.0
        return vec

    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return e_x / np.sum(e_x, axis=0, keepdims=True)

    def forward(self, inputs_indices):
        xs, hs, ps = {}, {}, {}
        hs[-1] = np.zeros((self.hidden_size, 1))

        for t in range(len(inputs_indices)):
            xs[t] = self.one_hot_encode(inputs_indices[t])
            hs[t] = np.tanh(self.Wxh @ xs[t] + self.Whh @ hs[t-1] + self.bh)
            ys_t = self.Why @ hs[t] + self.by
            ps[t] = self.softmax(ys_t)

        return xs, hs, ps, hs[len(inputs_indices)-1]

    def backward(self, xs, hs, ps, target_idx):
        dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
        dh_next = np.zeros_like(hs[0])
        target_one_hot = self.one_hot_encode(target_idx)

        for t in reversed(range(len(xs))):
            dy = ps[t].copy()
            dy[target_idx] -= 1

            dWhy += dy @ hs[t].T
            dby += dy

            dh = self.Why.T @ dy + dh_next
            dtanh = (1 - hs[t]**2) * dh

            dbh += dtanh
            dWxh += dtanh @ xs[t].T
            dWhh += dtanh @ hs[t-1].T
            dh_next = self.Whh.T @ dtanh

        grads = [dWxh, dWhh, dWhy, dbh, dby]
        max_norm = 5.0
        for grad in grads:
            np.clip(grad, -max_norm, max_norm, out=grad)

        return dWxh, dWhh, dWhy, dbh, dby

    def update_parameters(self, grads):
        dWxh, dWhh, dWhy, dbh, dby = grads
        self.Wxh -= self.learning_rate * dWxh
        self.Whh -= self.learning_rate * dWhh
        self.Why -= self.learning_rate * dWhy
        self.bh -= self.learning_rate * dbh
        self.by -= self.learning_rate * dby

    def train(self, input_sequence_words, target_word, epochs=100):
        if not self.word_to_idx:
             raise ValueError("Vocabulary not built. Call build_vocab first.")
        if target_word not in self.word_to_idx:
             raise ValueError(f"Target word '{target_word}' not in vocabulary.")
        if not all(word in self.word_to_idx for word in input_sequence_words):
             raise ValueError("One or more input words not in vocabulary.")

        input_indices = [self.word_to_idx[w] for w in input_sequence_words]
        target_idx = self.word_to_idx[target_word]

        for epoch in range(epochs):
            xs, hs, ps, _ = self.forward(input_indices)
            grads = self.backward(xs, hs, ps, target_idx)
            self.update_parameters(grads)

            loss = 0.0
            for t in ps:
                prob_target = ps[t][target_idx, 0]
                loss += -np.log(prob_target + 1e-9)
            loss /= len(ps)

            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, input_sequence_words):
        if not self.word_to_idx:
             raise ValueError("Vocabulary not built. Call build_vocab first.")
        if not all(word in self.word_to_idx for word in input_sequence_words):
             raise ValueError("One or more input words not in vocabulary.")

        input_indices = [self.word_to_idx[w] for w in input_sequence_words]
        _, _, ps, _ = self.forward(input_indices)
        last_ps = ps[len(input_indices) - 1]
        pred_idx = np.argmax(last_ps)
        return self.idx_to_word[pred_idx]

if __name__ == "__main__":
    text = "I love python programming and machine learning"
    words = text.split()

    rnn = SimpleRNNNumpy(hidden_size=10, learning_rate=0.05)
    rnn.build_vocab(text)

    input_sequence = words[:3]
    target_word = words[3]

    print("\n--- Before Training ---")
    print(f"Input: {' '.join(input_sequence)}")
    try:
        initial_pred = rnn.predict(input_sequence)
        print(f"Initial Prediction: {initial_pred}")
    except KeyError as e:
        print(f"Initial prediction failed (likely untrained): {e}")
    print(f"Target: {target_word}")

    print("\n--- Training ---")
    rnn.train(input_sequence, target_word, epochs=200)

    print("\n--- After Training ---")
    print(f"Input: {' '.join(input_sequence)}")
    final_pred = rnn.predict(input_sequence)
    print(f"Final Prediction: {final_pred}")
    print(f"Target: {target_word}")

    print("\n--- Another Prediction ---")
    test_sequence = ['python', 'programming', 'and']
    if all(w in rnn.word_to_idx for w in test_sequence):
        next_word = rnn.predict(test_sequence)
        print(f"Input: {' '.join(test_sequence)}")
        print(f"Predicted next word: {next_word}")
    else:
        print(f"Cannot predict for sequence '{' '.join(test_sequence)}' - words not in vocab.")
