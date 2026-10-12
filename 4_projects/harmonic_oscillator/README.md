# Harmonic Oscillator: Classical & Quantum Simulations

A comprehensive tutorial and simulation suite exploring the physics of harmonic oscillators, ranging from classical mechanics to quantum optomechanical systems. Developed with interactive Python scripts and Jupyter notebooks following British English conventions.

---

## 🚀 Quick Links & Interactive Execution

* **Full Source Code (`oscillator_app.py`)**: Classical SHO, Damped, and Quantum wavefunctions.
* **Full Source Code (`driven_oscillator_app.py`)**: Driven/Resonant frequency response modelling.
* **▶️ Run in your Browser (No Installation Required)**:
  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mukedon/python_and_projects/main?urlpath=lab/tree/4_projects/harmonic_oscillator)  
  *Click the badge above to launch an interactive cloud JupyterLab environment. (Note: The first launch may take 1–3 minutes while the container builds).*

---

## 📂 Repository Contents

* `harmonic_oscillator.ipynb` : Main tutorial notebook containing derivations, analytical solutions, and simulations.
* `oscillator_app.py` : Standalone Python script for classical trajectory analysis and quantum Fock state visualisations.
* `driven_oscillator_app.py` : Script simulating resonance curves and steady-state amplitude responses across multiple damping ratios.
* `requirements.txt` : Python package dependency list.

---

## 🔬 Core Physics & Theoretical Models

### 1. Simple Harmonic Oscillator (SHO)
* **Equation of Motion**: $\frac{d^2 x}{dt^2} + \Omega^2 x = 0$, where $\Omega = \sqrt{\frac{k}{m}}$ is the undamped natural angular frequency.
* **General Solution**: $x(t) = A \cos(\Omega t) + B \sin(\Omega t)$.
* **Potential Energy**: Small-amplitude Taylor expansion around equilibrium yields $U(x) \approx \frac{1}{2} U''(0) x^2$.

### 2. Damped Harmonic Oscillator (DHO)
* **Equation of Motion**: $\frac{d^2 x}{dt^2} + 2\zeta\Omega_0 \frac{dx}{dt} + \Omega_0^2 x = 0$, where $\zeta = \frac{c}{2\sqrt{mk}}$ is the dimensionless damping ratio.
* **Damping Regimes**:
  * **Overdamped ($\zeta > 1$)**: Exponential return to equilibrium without oscillation.
  * **Critically Damped ($\zeta = 1$)**: Fastest return to equilibrium without overshooting.
  * **Underdamped ($\zeta < 1$)**: Oscillations with exponentially decaying amplitude envelopes.

### 3. Driven/Resonated Harmonic Oscillator
* **Driven Equation**: $\frac{d^2 q}{dt^2} + 2\zeta\Omega_0 \frac{dq}{dt} + \Omega_0^2 q = \frac{F_0}{m} \cos(\Omega t)$.
* **Resonance**: Amplitude peaking near $\Omega_{\text{res}} = \Omega_0 \sqrt{1 - 2\zeta^2}$, inversely proportional to the damping coefficient.

### 4. Quantum Harmonic Oscillator
* **Hamiltonian**: $\hat{H} = \hbar\Omega\left(\hat{a}^\dagger\hat{a} + \frac{1}{2}\right)$.
* **Fock States & Wavefunctions**: Position-space eigenstates $\psi_n(q)$ expressed via Hermite polynomials with zero-point fluctuation amplitude $x_{\text{zpf}} = \sqrt{\frac{\hbar}{2m\Omega}}$.
* **Thermal Statistics**: Bose-Einstein distribution governing mean thermal occupation numbers $\bar{n}_{\text{th}} = \frac{1}{\exp\left(\frac{\hbar\Omega}{k_B T}\right) - 1}$.

---

## 💻 Running Locally

To run the simulations and notebooks on your local machine, execute the following commands in your terminal:

```bash
# 1. Clone the repository
git clone [https://github.com/mukedon/python_and_projects.git](https://github.com/mukedon/python_and_projects.git)

# 2. Navigate to the project directory
cd python_and_projects/4_projects/harmonic_oscillator

# 3. Install required dependencies
pip install -r requirements.txt

# 4. Run the applications
python oscillator_app.py
python driven_oscillator_app.py