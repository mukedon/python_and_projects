# Python Studies

A collection of Python notebooks and scripts exploring scientific computing, numerical methods, simulation, and quantum information. The projects span classical physics, probability theory, linear algebra, and quantum error correction — implemented primarily in Jupyter notebooks.

---

## Repository Structure

### Monte Carlo Simulations
**`monte_carlo_sims/`**

Probabilistic simulations using random sampling to approximate mathematical results. Covers classic applications such as estimating π, integration, and stochastic modeling. Demonstrates the power of repeated random trials for solving deterministic problems.

---

### Differential Equations
**`differential_equations/`**

Numerical solutions to ordinary and/or partial differential equations using Python. Likely covers methods such as Euler integration, Runge-Kutta solvers, and visualization of solution trajectories. Practical applications in physics and engineering modeling.

---

### Double Pendulum
**`double_pendulum/`**

Simulation of the canonical chaotic system in classical mechanics. Solves the equations of motion for a double pendulum and visualizes the sensitive dependence on initial conditions — a hallmark of deterministic chaos. Animated trajectories illustrate the divergence of nearby states over time.

---

### Galton Board
**`galton_board/`**

A simulation of Sir Francis Galton's physical device that demonstrates the emergence of a normal (Gaussian) distribution from repeated binary random events. Each ball falls through a lattice of pegs, and the aggregate histogram converges to a bell curve as the number of trials grows — a visual proof of the Central Limit Theorem.

---

### Feedback Trajectories
**`feedback_trajectories/`**

Exploration of dynamical systems with feedback. Simulates how systems evolve when outputs feed back into inputs — covering concepts like fixed points, limit cycles, and convergence behavior.

---

### Matrix Multiplication
**`matrix_mult/`**

Implementations and performance analysis of matrix multiplication. Covers naive approaches alongside optimized routines, with comparisons to NumPy's BLAS-backed operations. Useful for understanding computational complexity and numerical linear algebra.

---

### Stim Surface Code
**`stim_surface_code/`**

Quantum error correction simulations using [Stim](https://github.com/quantumlib/Stim), Google's high-performance stabilizer circuit simulator. Implements rotated and/or unrotated surface code circuits with configurable noise models, detector sampling, and logical error rate estimation. Relevant to fault-tolerant quantum computing research.

---

### Trotter–Clifford Circuits
**`trott_cliff/`**

Quantum circuit constructions combining Trotterization (used to simulate Hamiltonian evolution) with Clifford operations. Explores how Clifford circuits can be used to efficiently represent and simulate quantum dynamics within the stabilizer formalism.

---

### Tutorials
**`tutorials/`**

General-purpose Python tutorials covering core language features, data structures, and scientific computing libraries. A useful reference for Python patterns and best practices used throughout the rest of the repository.

---

## Requirements

Most notebooks rely on standard scientific Python libraries:

```
numpy
scipy
matplotlib
jupyter
```

The quantum error correction notebooks additionally require:

```
stim
pymatching   # (optional, for decoding)
```

Install all dependencies with:

```bash
pip install numpy scipy matplotlib jupyter stim pymatching
```

---

## Getting Started

Clone the repository and launch Jupyter:

```bash
git clone https://github.com/mukedon/python_studies.git
cd python_studies
jupyter notebook
```

Then navigate to any folder and open the `.ipynb` file of interest.

---

## License

This repository is intended for educational purposes. Feel free to use, adapt, or build on the code with attribution.
