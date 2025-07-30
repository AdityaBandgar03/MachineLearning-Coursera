
# Min-Max Scaling in Machine Learning

## What is Min-Max Scaling?
Min-Max Scaling (or Min-Max Normalization) is a technique used to scale data within a fixed range, typically **[0, 1]** or **[-1, 1]**. This method preserves the relationships between data points while ensuring that all values fit within a specified range.

## Formula
The Min-Max scaling formula is:

&\[
X' = \frac{X - X_{\text{min}}}{X_{\text{max}} - X_{\text{min}}}
\]

where:
- \( X \) = Original value
- \( X_{\text{min}} \) = Minimum value in the dataset
- \( X_{\text{max}} \) = Maximum value in the dataset
- \( X' \) = Scaled value in the range **[0,1]**

If scaling to a different range **[a, b]**, the formula is:

\[
X' = a + \left( \frac{X - X_{\text{min}}}{X_{\text{max}} - X_{\text{min}}} \right) \times (b - a)
\]

## Why Use Min-Max Scaling?
- Ensures all features have the same scale, preventing some features from dominating others.
- Preserves original distribution relationships.
- Useful for algorithms that rely on distance calculations (e.g., **KNN, K-Means, Neural Networks**).

# 📉 Gradient Descent in Machine Learning

## 📌 What is Gradient Descent?
Gradient Descent is an **optimization algorithm** used to minimize a function by iteratively moving in the direction of the steepest descent, as defined by the **negative gradient**. It is widely used for optimizing **machine learning models**, particularly in training **neural networks** and regression models.

## 🔢 Formula
The general formula for updating the model parameters using Gradient Descent is:
θ = θ - α * ∇J(θ)

where:
- `θ` = Model parameters (weights)
- `α` = Learning rate (step size)
- `∇J(θ)` = Gradient of the cost function `J(θ)` with respect to `θ`

For **Linear Regression**, the parameter update rule is:
θ_j = θ_j - α * (∂J/∂θ_j)

where:
- `θ_j` = Parameter for feature `j`
- `(∂J/∂θ_j)` = Partial derivative of the cost function with respect to `θ_j`


## Types of Gradient Descent
### Batch Gradient Descent
- Computes the gradient using the **entire dataset**.
- **Pros**: Stable updates, converges smoothly.
- **Cons**: Slow for large datasets.

###  Stochastic Gradient Descent (SGD)
- Computes the gradient **for a single data point** at each step.
- **Pros**: Faster updates, better for large datasets.
- **Cons**: More fluctuations, may not converge smoothly.

###  Mini-Batch Gradient Descent
- Computes the gradient using a **small batch** of data points.
- **Pros**: Balances efficiency and stability.
- **Cons**: Requires tuning the batch size.

# 📉 Batch Gradient Descent vs. Stochastic Gradient Descent

## What is Gradient Descent?
Gradient Descent is an **optimization algorithm** used in machine learning to minimize a cost function by updating model parameters iteratively in the direction of the **negative gradient**.

There are multiple variations of Gradient Descent, but the two most commonly used are:
- **Batch Gradient Descent**
- **Stochastic Gradient Descent (SGD)**

---

## 1️⃣ Batch Gradient Descent

### Definition
Batch Gradient Descent computes the gradient of the cost function **using the entire dataset** before updating the parameters.

### 🔢 Formula
For a cost function `J(θ)`, the parameter update rule is:
θ = θ - α * (1/m) * Σ (∇J(θ))
where:
- `θ` = Model parameters (weights)
- `α` = Learning rate
- `m` = Total number of training samples
- `∇J(θ)` = Gradient of the cost function
- `Σ` = Summation over all training examples

### Advantages
✔️ More **stable** updates, leading to smooth convergence.  
✔️ More **accurate** updates as it considers all data points.  
✔️ Suitable for **smaller datasets** where computation is not an issue.  

### Disadvantages
❌ **Slow convergence** for large datasets.  
❌ Requires storing the entire dataset in memory.  




# 📉 Stochastic Gradient Descent (SGD)

## 📌 What is Stochastic Gradient Descent?
Stochastic Gradient Descent (SGD) is an **optimization algorithm** used in machine learning to minimize a cost function by updating the model parameters iteratively. Unlike **Batch Gradient Descent**, which updates parameters after processing all training examples, **SGD updates the parameters after processing each individual training example**.

---

## 🔢 Formula
For each training example `(x_i, y_i)`, the parameter update rule is:
θ = θ - α * ∇J(θ; x_i, y_i)
where:
- `θ` = Model parameters (weights)
- `α` = Learning rate (step size)
- `∇J(θ; x_i, y_i)` = Gradient of the cost function computed using only one training example `(x_i, y_i)`

SGD updates the parameters **much more frequently** than Batch Gradient Descent, leading to faster but noisier convergence.

---

## Advantages of Stochastic Gradient Descent
✔️ **Faster updates**, making it suitable for large datasets.  
✔️ **Efficient for online learning** and real-time data processing.  
✔️ **Requires less memory**, as it does not store the entire dataset.  
✔️ **Can escape local minima** due to its inherent randomness.

---

## Disadvantages of Stochastic Gradient Descent
❌ Updates are **noisy**, leading to fluctuations in parameter updates.  
❌ **May not converge smoothly** like Batch Gradient Descent.  
❌ Requires careful **learning rate tuning** to avoid divergence.  

---
