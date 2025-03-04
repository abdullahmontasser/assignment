import random

# Input values
i1, i2 = 0.05, 0.10

# Random weights between -0.5 and 0.5
random.seed(42)
w1, w2, w3, w4, w5, w6, w7, w8 = [random.uniform(-0.5, 0.5) for _ in range(8)]

# Biases
b1, b2 = 0.5, 0.7


# Tanh activation function
def tanh(x: float) -> float:
    """Compute the hyperbolic tangent (tanh) of x."""
    e_pos = pow(2.718281828459045, x)
    e_neg = pow(2.718281828459045, -x)
    return (e_pos - e_neg) / (e_pos + e_neg)


# Derivative of tanh
def tanh_derivative(x: float) -> float:
    """Compute the derivative of tanh(x)."""
    return 1 - tanh(x) ** 2


# Forward pass for the hidden layer
net_h1 = w1 * i1 + w2 * i2 + b1
net_h2 = w3 * i1 + w4 * i2 + b1
out_h1 = tanh(net_h1)
out_h2 = tanh(net_h2)

# Forward pass for the output layer
net_o1 = w5 * out_h1 + w6 * out_h2 + b2
net_o2 = w7 * out_h1 + w8 * out_h2 + b2
out_o1 = tanh(net_o1)
out_o2 = tanh(net_o2)


# Print the outputs and weights before back propagation
print("Weights: random weights \n")
print(f"w1 = {w1:.2f}", f"w2 = {w2:.2f}", sep="\t" * 3)
print(f"w3 = {w3:.2f}", f"w4 = {w4:.2f}", sep="\t" * 3)
print(f"w5 = {w5:.2f}", f"w6 = {w6:.2f}", sep="\t" * 3)
print(f"w7 = {w7:.2f}", f"w8 = {w8:.2f}", sep="\t" * 3, end="\n" * 2)
print("outputs for the random weights \n")
print(f"net_h1 = {net_h1:.4f}", f"out_h1 = {out_h1:.4f}", sep="\t" * 3)
print(f"net_h2 = {net_h2:.4f}", f"out_h2 = {out_h2:.4f}", sep="\t" * 3, end="\n" * 2)

print(f"net_o1 = {net_o1:.4f}", f"-< out_o1 = {out_o1:.4f} >-", sep="\t" * 3)
print(
    f"net_o2 = {net_o2:.4f}", f"-< out_o2 = {out_o2:.4f} >-", sep="\t" * 3, end="\n" * 2
)

# Target values
target_o1, target_o2 = 0.01, 0.99

# Learning rate
learning_rate = 0.5

# Calculate errors
error_o1 = target_o1 - out_o1
error_o2 = target_o2 - out_o2

# Calculate output layer deltas
delta_o1 = error_o1 * tanh_derivative(net_o1)
delta_o2 = error_o2 * tanh_derivative(net_o2)

# Calculate hidden layer deltas
delta_h1 = (delta_o1 * w5 + delta_o2 * w7) * tanh_derivative(net_h1)
delta_h2 = (delta_o1 * w6 + delta_o2 * w8) * tanh_derivative(net_h2)

# Update output layer weights and bias
w5 += learning_rate * delta_o1 * out_h1
w6 += learning_rate * delta_o1 * out_h2
w7 += learning_rate * delta_o2 * out_h1
w8 += learning_rate * delta_o2 * out_h2
b2 += learning_rate * (delta_o1 + delta_o2)

# Update hidden layer weights and bias
w1 += learning_rate * delta_h1 * i1
w2 += learning_rate * delta_h1 * i2
w3 += learning_rate * delta_h2 * i1
w4 += learning_rate * delta_h2 * i2
b1 += learning_rate * (delta_h1 + delta_h2)


# Print updated weight values
print("Updated Weights: after back propagation \n")
print(f"w1 = {w1:.4f}", f"w2 = {w2:.4f}", sep="\t" * 3)
print(f"w3 = {w3:.4f}", f"w4 = {w4:.4f}", sep="\t" * 3)
print(f"w5 = {w5:.4f}", f"w6 = {w6:.4f}", sep="\t" * 3)
print(f"w7 = {w7:.4f}", f"w8 = {w8:.4f}", sep="\t" * 3)
print(f"b1 = {b1:.4f}", f"b2 = {b2:.4f}", sep="\t" * 3)
