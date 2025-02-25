import random

# Input values
i1, i2 = 0.05, 0.10

# Random weights between -0.5 and 0.5
random.seed(42)  # For reproducibility
w1, w2, w3, w4, w5, w6, w7, w8 = [random.uniform(-0.5, 0.5) for _ in range(8)]

# Biases
b1, b2 = 0.5, 0.7

# Tanh activation function
def tanh(x: float) -> float:
    """Compute the hyperbolic tangent (tanh) of x."""
    e_pos = pow(2.718281828459045, x)   # e^x
    e_neg = pow(2.718281828459045, -x)  # e^-x
    return (e_pos - e_neg) / (e_pos + e_neg)


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

print(f"Output o1: {out_o1:.4f}")
print(f"Output o2: {out_o2:.4f}")

# Print the outputs
print(f"w1 = {w1:.2f}", f"w2 = {w2:.2f}", sep="\t" * 3)
print(f"w3 = {w3:.2f}", f"w4 = {w4:.2f}", sep="\t" * 3)
print(f"w5 = {w5:.2f}", f"w6 = {w6:.2f}", sep="\t" * 3)
print(f"w7 = {w7:.2f}", f"w8 = {w8:.2f}", sep="\t" * 3, end="\n" * 2)

print(f"net_h1 = {net_h1:.4f}", f"out_h1 = {out_h1:.4f}", sep="\t" * 3)
print(f"net_h2 = {net_h2:.4f}", f"out_h2 = {out_h2:.4f}", sep="\t" * 3, end="\n" * 2)

print(f"net_o1 = {net_o1:.4f}", f"-< out_o1 = {out_o1:.4f} >-", sep="\t" * 3)
print(f"net_o2 = {net_o2:.4f}", f"-< out_o2 = {out_o2:.4f} >-", sep="\t" * 3, end="\n" * 2)
