# Discretization of Gradient Flow

## Overview

This repository explores and compares three gradient descent approaches—**Explicit GD**, **Implicit GD**, and **Newton GD**—using only **NumPy** for maximum transparency and control over the gradient flow. The goal is to study their convergence behaviors in depth.

### Models Explanation

- **Explicit GD (Standard Gradient Descent)**

  - **Formula:**

$$\theta_{k+1} = \theta_k - \eta \nabla L_k$$

- **Explanation:**  
  Uses a **first-order (linear) Taylor approximation** of the loss to update parameters. Each step moves along the gradient direction with step size $\eta$. Simple but may struggle with stiff or ill-conditioned loss landscapes.

- **Implicit GD (Backward Euler on Quadratic)**

  - **Formula:**

$$\theta_{k+1} = (I + \eta H_k)^{-1} \left(\theta_k + \eta (H_k \theta_k - \nabla L_k)\right)$$

- **Explanation:**  
  Uses a **second-order (quadratic) Taylor approximation**, effectively taking the curvature of the loss into account. This allows more stable updates, particularly in stiff regions, by solving for the next step implicitly.

- **Newton GD (Newton’s Method)**
  - **Formula:**

$$\theta_{k+1} = \theta_k - H_k^{-1} \nabla L_k$$

- **Explanation:**  
  Also a quadratic approximation, but it **directly jumps to the local optimum** of the quadratic approximation in a single step.
  - **Note:** Damping and Armijo line search are used to prevent instability, as Newton updates can diverge if the Hessian is ill-conditioned or the step overshoots.

### Experiment Objective

- Compare the **advantages and disadvantages** of each gradient descent method.
- Analyze convergence speed and stability using **validation loss** curves over training batches.

### Experiment Results

#### 1. Early Convergence (Initial Steps)

- Observed speed:

$$\text{Newton GD} > \text{Implicit GD} > \text{Explicit GD}$$

- **Reason:**
  - Newton GD can jump directly to the optimum of the local quadratic, achieving rapid initial convergence.
  - Implicit GD moves along a precise quadratic path, slower than Newton but faster than explicit GD.
  - Explicit GD follows a rough linear path, making its early convergence the slowest.

<p align="center" style="text-align: center;">
  <img src="docs-src/result/short-convergence-speed.png" alt="Early Convergence Comparison" width="450px">
  <br>
  <em>Figure 1: Validation loss comparison (batch 0–45, ~0–1 epochs)</em>
</p>

#### 2. Late Convergence (Long-Term Behavior)

- Observed trend:

$$\text{Implicit GD} > \text{Explicit GD}, \quad \text{Newton GD is unstable}$$

- **Insights:**
  - Implicit GD’s quadratic approximation ensures fast and **stable convergence**, often faster than explicit GD in the long term. Convergence resembles $\frac{1}{x^2}$ versus $\frac{1}{x}$ for explicit GD.
  - Explicit GD eventually converges but requires much longer training.
  - Newton GD becomes unstable when the loss drops below ~4%, as Hessian sensitivity amplifies noise, causing erratic updates.

<p align="center" style="text-align: center;">
  <img src="docs-src/result/long-convergence-speed.png" alt="Long-Term Convergence Comparison" width="450px">
  <br>
  <em>Figure 2: Validation loss comparison over extended training</em>
</p>

#### 3. Conclusion (Long-Term Behavior)

- **Explicit GD**: Slow but safe; eventually converges with enough training.
- **Implicit GD**: Stable and follows a quadratic path; converges faster than explicit GD in the long term.
- **Newton GD**: Extremely fast at the beginning but becomes unstable later due to Hessian sensitivity and noise amplification.

## Project Structure

```bash
Project/
├── custom_library/
│    │
│    ├── core/               # The absolute fundamentals of all models
│    │   ├── baseLrModels.py             # base model functions: getHistory(), predict(), save(), etc.
│    │   └── baseLrModelsImplementHessain.py         # compute tools: compute_loss(), compute_gradient(), compute_hessian()
│    │
│    ├── layers/             # Pure layer implementations
│    │   ├── denseLayer.py
│    │   └── activationLayer.py
│    │
│    ├── models/             # Training *strategies*
│    │   ├── explicitLrModel.py
│    │   ├── implicitLrModel.py
│    │   ├── newtonLrModel.py
│    │   └── optimizeNewtonLrModel.py
│    │
│    └── utils/              # Tools, not ML logic
│        ├── history.py
│        ├── metrics.py
│        └── dataUtils.py
│
├── main.ipynb      # Experiment lab, including all process
│
└── README.md       # Experiment report, including all description
```

## Mathematical Proof

### 1. The origin of each discretization types

#### 1.0.1 Taylor Series

The Taylor series gives an approximation of $L(\theta)$ around the current point $\theta$, with $\Delta \theta = \theta - \theta_0$:

$$
L(\theta + \Delta \theta) \approx L(\theta) + (\nabla L(\theta))^T (\Delta \theta) +
\frac{1}{2}(\Delta \theta)^T(\nabla^2 L(\theta)) (\Delta \theta) + \cdots
$$

$$
L(\theta + \Delta \theta) \approx L(\theta) + g_t^T (\Delta \theta) +
\frac{1}{2}(\Delta \theta)^T H_t (\Delta \theta) + \cdots
$$

#### 1.1 Explicit Gradient Descent (Normal Gradient Descent)

Explicit gradient descent uses only the first-order terms of the Taylor series.

We choose $\Delta \theta$ to maximize the decrease in the loss under the linear approximation:

$$
\Delta L \approx (\nabla L(\theta))^T \Delta \theta
= \lVert \nabla L(\theta) \rVert \lVert \Delta \theta \rVert \cos (\phi)
$$

The decrease is largest (most negative) when:

$$
\cos (\phi) = -1 , \quad \phi = 180 \degree
$$

Thus:

$$
\text{Explicit GD:} \quad \theta_{k+1} = \theta_k - \eta \nabla L(\theta)
$$

#### 1.0.2 Quadratic form from Taylor Series

We construct the quadratic approximation:

$$
L(\theta) = a \Delta \theta^2 - b \Delta \theta + c
$$

Recall:

- $\Delta \theta = \theta - \theta_0$
- $L(\theta + \Delta \theta) \approx L(\theta) + g_t^T (\Delta \theta) + (\Delta \theta)^T H_t (\Delta \theta) + \cdots$

Substitute the second-order term:

$$
\begin{align*}
  \frac{1}{2} (\Delta \theta)^T H (\Delta \theta) &\approx
  \frac{1}{2} (\theta - \theta_0)^T H (\theta - \theta_0) \\
  &\approx \frac{1}{2} \left[ (\theta^T H \theta) + (\theta_0^T H \theta) + (\theta^T H \theta_0) + (\theta_0^T H \theta_0) \right] \\
  &\approx \frac{1}{2} \left[ (\theta^T H \theta) + (\theta^T H \theta_0) + (\theta^T H \theta_0) + (\theta_0^T H \theta_0) \right] \\
  &\approx \frac{1}{2} (\theta^T H \theta) + (\theta^T H \theta_0)  + \frac{1}{2} (\theta_0^T H \theta_0)
\end{align*}
$$

Then the full approximation becomes:

$$
\begin{align*}
  L(\theta) &\approx
  L(\theta_0) + g^T (\Delta \theta) + \frac{1}{2} (\theta^T H \theta) + (\theta^T H \theta_0)  + \frac{1}{2} (\theta_0^T H \theta_0) \\
  &\approx \frac{1}{2} \left[ H \right] \theta^2 - \left[ H \theta_0 - g \right]^T \theta + \left[ L(\theta_0) + \frac{1}{2} (\theta_0^T H \theta_0) \right] \\
  &\approx \frac{1}{2} a \theta^2 - b \theta + c
\end{align*}
$$

#### 1.2 Implicit Gradient Descent (Backward Euler on Quadratic)

Implicit gradient descent uses the second-order Taylor approximation.

Unlike explicit GD, which uses:

$$
\theta_{k+1} = \theta_{k} - \eta \nabla L_k
$$

Implicit GD uses:

$$
\theta_{k+1} = \theta_{k} - \eta \nabla L_{k+1}
$$

This means we predict where the next point will be, compute the gradient there,
and move according to that slope — a two-step prediction process.

From the quadratic approximation:

$$
L(\theta) \approx \frac{1}{2} H \theta^2 - b \theta + c
$$

$$
\nabla L(\theta) \approx H \theta - b
$$

Substitute into the update rule:

$$
\begin{align*}
  \theta_{k+1} &= \theta_{k} - \eta \nabla L_{k+1} \\
  &= \theta_{k} - \eta (H \theta_{k+1} - b) \\
  &= \theta_{k} - \eta H \theta_{k+1} + \eta b
\end{align*}
$$

$$
\begin{align*}
  \theta_{k+1} + \eta H \theta_{k+1} = \theta_{k} + \eta b \\
  (I + \eta H)(\theta_{k+1}) = \theta_{k} + \eta b \\
\end{align*}
$$

Thus:

$$
\theta_{k+1} = (I + \eta H)^{-1} (\theta_{k} + \eta b)
$$

Substituting $b = H \theta_k - g$

$$
\text{Implicit GD:} \quad \theta_{k+1} = (I + \eta H)^{-1} (\theta_{k} + \eta H \theta_k - \eta g)
$$

#### 1.3 Newton Gradient Descent (Newton's Method)

Newton’s method also uses the quadratic approximation, but instead of taking a small step of size $\eta$, it directly jumps to the stationary point where the gradient is zero:

From the quadratic:

$$
L(\theta) \approx L(\theta_0) + g^T (\Delta \theta) + \frac{1}{2} ((\Delta \theta)^T H (\Delta\theta))
$$

Gradient:

$$
\nabla L(\theta) \approx  g +  H (\Delta \theta)
$$

Setting $\nabla L(\theta) = 0$:

$$
0 \approx  g +  H (\Delta \theta)
$$

$$
H (\Delta \theta) \approx  -g
$$

$$
\Delta \theta \approx -H^{-1} g
$$

Thus:

$$
\text{Newton GD:} \quad \theta_{k+1} = \theta_k - H^{-1} g
$$

---

### 2. Convergence of each discretization types

> **🚧 Planned for the next milestone.**
> This section will derive the convergence *rates* of each method — formalizing the
> empirical observation in [Experiment Results](#experiment-results) that Implicit GD
> behaves like $\tfrac{1}{x^2}$ while Explicit GD behaves like $\tfrac{1}{x}$, and that
> Newton GD destabilizes once the loss is small. The analysis below is not yet written.

#### 2.1 Explicit Gradient Descent (Normal Gradient Descent)

_Planned: linear/sublinear convergence under a fixed step size._

#### 2.2 Implicit Gradient Descent (Backward Euler on Quadratic)

_Planned: stability of the backward-Euler step and its faster long-term rate._

#### 2.3 Newton Gradient Descent (Newton's Method)

_Planned: local quadratic convergence and the source of late-stage instability._

---

## 2. Components and Fundamental Properties of the Hessian

### 2.1 Dimensions and Interpretation of the Hessian

- **Gradient (First-order derivative):** $\nabla L(\theta) \in \mathbb{R}^n$ — describes the slope of $L$

- **Hessian (Second-order derivative):** $\nabla^2 L(\theta) = H(\theta) \in \mathbb{R}^{n \times n}$ — describes the curvature of $L$

<p align="center"  style="text-align: center;">
  <img src="docs-src/latex/2.1.hessain-dimension.png" alt="Neural Network Overview" width="600px">
</p>

<!-- $$
\displaystyle
\boldsymbol{\theta}
=
\begin{bmatrix}
\theta_1 \\ \theta_2 \\ \vdots \\ \theta_n
\end{bmatrix}

, \quad

\nabla L(\boldsymbol{\theta})
=
\begin{bmatrix}
\displaystyle \frac{\partial L}{\partial \theta_1}
\\[10pt]

\displaystyle \frac{\partial L}{\partial \theta_2}
\\[10pt]

\vdots
\\[10pt]

\displaystyle \frac{\partial L}{\partial \theta_n}
\end{bmatrix}

, \quad

H(\boldsymbol{\theta})
=
\nabla^2 L(\boldsymbol{\theta})
=
\begin{bmatrix}
\displaystyle \frac{\partial^2 L}{\partial \theta_1^2}
&
\displaystyle \frac{\partial^2 L}{\partial \theta_1 \partial \theta_2}
&
\cdots
&
\displaystyle \frac{\partial^2 L}{\partial \theta_1 \partial \theta_d}
\\[10pt]

\displaystyle \frac{\partial^2 L}{\partial \theta_2 \partial \theta_1}
&
\displaystyle \frac{\partial^2 L}{\partial \theta_2^2}
&
\cdots
&
\displaystyle \frac{\partial^2 L}{\partial \theta_2 \partial \theta_d}
\\[10pt]

\vdots & \vdots & \ddots & \vdots
\\[10pt]

\displaystyle \frac{\partial^2 L}{\partial \theta_d \partial \theta_1}
&
\displaystyle \frac{\partial^2 L}{\partial \theta_d \partial \theta_2}
&
\cdots
&
\displaystyle \frac{\partial^2 L}{\partial \theta_d^2}
\end{bmatrix}
$$ -->

### 2.2 Symmetry Properties

From Clairaut's Theorem (Equality of Mixed Partials), if the mixed second partial derivatives are continuous over a region, then:

$$
\frac{\partial^2 f(x)} {\partial x_i \partial x_j}
= \frac{\partial^2 f(x)} {\partial x_j \partial x_i} \quad
\longrightarrow \quad
H_{ij}(\theta)
= \frac{\partial^2 f(x)} {\partial \theta_i \partial \theta_j}
= \frac{\partial^2 f(x)} {\partial \theta_j \partial \theta_i}
= H_{ji}(\theta)
$$

### 2.3 Quadratic Form

The quadratic form $Q(\mathbf{x}) = \mathbf{x}^T H \mathbf{x}$ is the second-order term of a function's Taylor expansion. It tells us about the local curvature (or shape) of the function $f(\mathbf{x})$ around a critical point.

$$
\begin{align*}
Q(\mathbf{x}) &= \mathbf{x}^T \mathbf{H} \mathbf{x} \in \mathbb{R}, \quad \forall \mathbf{x} \in \mathbb{R}^n \quad \text{(The Quadratic Form)} \\
&= \sum_{i=1}^n \sum_{j=1}^n h_{ij} x_i x_j \\
&= \sum_{i=1}^n h_{ii} x_i^2 + \sum_{i \ne j}^n h_{ij} x_i x_j \\
&= \sum_{i=1}^n h_{ii} x_i^2 + 2 \sum_{i \lt j}^n h_{ij} x_i x_j \in \mathbb{R} \quad \text{, where } \mathbf{H} \text{ is symmetric}
\end{align*}
$$

- If Q(x) > 0 (Positive Definite), the function is convex and the critical point is a local minimum.
- If Q(x) < 0 (Negative Definite), the function is concave and the critical point is a local maximum.
- If Q(x) takes on both signs (Indefinite), the critical point is a saddle point.

### 2.4 Eigen-decomposition

From the Spectral Theorem, if $H \in \mathbb{R}^{n \times n}$ and $H$ is symmetric ($H = H^T$), then:

$$
\lambda_i \in \mathbb{R}, \quad \text{for all } i = 1, \ldots, n ,\quad
\mathbf{v}_i \in \mathbb{R}^n, \quad \text{for all } i = 1, \ldots, n
$$

Proving:

1. $A \in \mathbb{R}^{n \times n}, \lambda \in \mathbb{R}, v \in \mathbb{R}^n$ that $Av = \lambda v \ (1)$
2. take (1) with $v^\dagger$ from the left side

$$
\begin{array}{l}
v^\dagger (Av) = v^\dagger (\lambda v) \quad
(v^\dagger A) v = \lambda (v^\dagger v) \quad (2)
\end{array}
$$

3. take (2) with conjugate transpose

$$
[(v^\dagger A) v]^\dagger = v^\dagger A^\dagger v^{\dagger\dagger} = v^\dagger A^\dagger v \quad (3)
$$

4. Because $A \in \mathbb{R}^{n \times n}$, making $\bar{A} = A$. And because $A$ is a symmetric, making $A^T = A$. So from (3):

$$
\begin{array}{l}
v^\dagger A^\dagger v = v^\dagger \bar{A}^T v = v^\dagger A v \quad (4) \\
v^\dagger A^\dagger v = [\lambda (v^\dagger v)]^\dagger = \bar{\lambda} (v^\dagger v) \quad (5)
\end{array}
$$

5. From (2) is equal (5):

$$
(v^\dagger A) v = \lambda (v^\dagger v) = \bar{\lambda} (v^\dagger v)
\\
(\lambda - \bar{\lambda})(v^\dagger v) = 0
$$

6. Because $v^\dagger v \ne 0$, So:

$$
(\lambda - \bar{\lambda}) = 0 \quad \rightarrow \quad
\lambda = \bar{\lambda} \quad \longrightarrow \quad
\forall \lambda \in \mathbb{R}
$$

7. Using the eigenvector formula:

$$
(A - \lambda I)v = 0
$$

We know $A \in \mathbb{R}^{n \times n}, \lambda \in \mathbb{R}$, Thus $\forall (A-\lambda I) \in \mathbb{R}$, Thus $\forall v \in \mathbb{R}$.

### 2.5 Orthogonality of Eigenvectors

From the Spectral Theorem, if $H \in \mathbb{R}^{n \times n}$ and $H$ is symmetric ($H = H^T$), then the eigenvectors of $H$ are orthogonal.

Proving:

1. Consider $Av_1 = \lambda_1 v_1 \ (1)$ and $Av_2 = \lambda_2 v_2 \ (2)$
2. Start with $v_1^T (A v_2)$ to substitute $A v_2$:

$$
\begin{array}{l}
v_1^T (A v_2) = v_1^T (\lambda_2 v_2) \quad \\
v_1^T (A v_2) = \lambda_2 (v_1^T v_2) \quad (3)
\end{array}
$$

3. Since $v^T (A v)$ is a scalar (1×1), $v^T (A v) = [v^T (A v)]^T$. From (3):

$$
\begin{align*}
v_1^T (A v_2) &= [v_1^T (A v_2)]^T \\
&= v_2^T A^T v_1 \quad, A \ \text{is symmetric} \\
&= v_2^T A v_1 \quad, \text{substitute (1)} \\
&= v_2^T (\lambda_1 v_1) \\
&= \lambda_1 (v_2^T v_1) \quad (4)
\end{align*}
$$

4. (3) is equal (4):

$$
\begin{array}{c}
\lambda_1 (v_2^T v_1) = \lambda_2 (v_1^T v_2) \\
(\lambda_1-\lambda_2)(v_1^T v_2) = 0
\end{array}
$$

5. By the Spectral Theorem, we know $\lambda_i \in \mathbb{R}, \quad \text{for all } i = 1, \ldots, n ,\quad \mathbf{v}_i \in \mathbb{R}^n, \quad \text{for all } i = 1, \ldots, n$. For distinct eigenvalues $\lambda_1 \ne \lambda_2$, the factor $(\lambda_1 - \lambda_2)$ is non-zero. Thus $(v_1^T v_2) = 0$, i.e. $v_1 \cdot v_2 = 0$. Therefore $v_1$ and $v_2$ are orthogonal.

### 2.6 Quadratic Form of a Symmetric Hessian via Eigen-decomposition

The Hessian matrix, $\mathbf{H}$, being symmetric ($\mathbf{H} = \mathbf{H}^T$), can be decomposed via the Spectral Theorem as $\mathbf{H} = \mathbf{Q} \mathbf{\Lambda} \mathbf{Q}^T$. In this decomposition, the eigenvalues ($\lambda_i$) are guaranteed to be real numbers, and the corresponding eigenvectors ($\mathbf{v}_i$), collected in the orthogonal matrix $\mathbf{Q}$, form a basis of principal directions.

Applying this decomposition to the quadratic form $Q(\mathbf{x}) = \mathbf{x}^T \mathbf{H} \mathbf{x}$ allows for a precise analysis of the local curvature. The eigenvalues $\lambda_i$ quantify the magnitude and sign of the curvature along the unique directions defined by the eigenvectors $\mathbf{v}_i$.

$$
\begin{align*}
Q(x) = x^T H x &= x^T Q \Lambda Q^T x \\
&= (Q^T x)^T \Lambda (Q^T x) \\
&= \sum_{i}^n \Lambda_{ii} (Q^T x)_i^2 + 2 \sum_{i<j}^n \Lambda_{ii} (Q^T x)_i (Q^T x)_j \\
&= \sum_{i}^n \lambda_{ii} (v_i^T x)^2
\end{align*}
$$

- $\lambda_i$ = Principal Magnitude.
- $v_i$ = Principal Direction.
- $(v_i^T x)^2$ = Contribution of $x$ in $v_i$ direction.

---

## License

Original code and documentation authored by Jakkarin Promsee are licensed under the [MIT License](LICENSE). Third-party code, assets, datasets, and other materials are not covered by this license; their original terms apply.
