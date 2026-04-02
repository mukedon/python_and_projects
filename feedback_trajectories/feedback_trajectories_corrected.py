"""
Corrected simulation of two-particle dynamical equilibrium trajectories.

Based on: arXiv:2505.10221 (Döner, 2025)
          "Entanglement generation & dynamical equilibrium within de Broglie-Bohm theory"

The original code (feedback_trajectories_scheme.py) contained the following issues,
corrected here:

  Bug 1 — F1_1, F2_1, F1_2, F2_2 : Gm² prefactor missing from reference (second) term.
           Original: G*m*m*(x-X2)/|x-X2|³  −  (X1-X2)/|X1-X2|³   ← no Gm² on rhs
           Corrected: G*m² * [ (x-X2)/|x-X2|³  −  (X1-X2)/|X1-X2|³ ]

  Bug 2 — F2_1, F2_2 : wrong sign on the x-dependent (first) term.
           For particle 2 the gradient of V is ∂V/∂x₁, not ∂V/∂x₂ (Eq. 3b).
           ∂V/∂x₁[X₁,x] = Gm²(X₁−x)/|X₁−x|³  (not (x−X₁)/|x−X₁|³)
           Original: F2_1 = G*m*m*(x-X1)/|x-X1|³ − ...
           Corrected: F2_1 = G*m² * [ (X1-x)/|X1-x|³  −  (X1-X2)/|X1-X2|³ ]

  Bug 3 — Wave-function evolution: x-space arrays (F1_1, F1_2) added to k-space array (k²).
           exp(-i(ħk²/2m + F1_1 + F1_2)dt) is dimensionally and physically invalid.
           Corrected via the standard split-operator method:
             (i)   half-step kinetic in k-space
             (ii)  full-step conditional potential in x-space
             (iii) feedback amplitude modulation (1st-order, Eq. B2) in x-space
             (iv)  feedback phase modulation   (2nd-order, Eq. B4) in x-space
             (v)   half-step kinetic in k-space

  Bug 4 — np.gradient missing dx argument.
           np.gradient(np.log(psi1)) uses index-spacing = 1 instead of dx,
           giving a Bohmian velocity off by factor 1/dx.
           Corrected: np.gradient(np.log(psi1 + eps), dx)

  Bug 5 — No softening: np.abs(x-X2) hits zero when x = X2 (grid point coincides
           with particle position), causing NaN / Inf in all feedback terms.
           Corrected: softened distance  sqrt((a-b)² + ε²)

All equation references are to arXiv:2505.10221 unless noted otherwise.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq

# ═══════════════════════════════════════════════════════════════════════
# 1. SIMULATION PARAMETERS  (all constants = 1 as in paper §III.A)
# ═══════════════════════════════════════════════════════════════════════

hbar  = 1.0          # ħ
m     = 1.0          # particle mass
G     = 1.0          # gravitational constant
a     = 1.0          # initial separation |X₂₀ − X₁₀|
sigma = 0.5          # Gaussian width σ  (Eqs. 11–12)
dt    = 0.01         # time step  Δt
T     = 1000.0       # total simulation time
N     = 1000         # number of spatial grid points
L     = 100.0        # spatial domain length  [−L/2, L/2]
SOFT  = 0.3          # softening length ε  (Bug 5 fix)
EPS   = 1e-30        # regularisation for log of near-zero ψ

dx = L / N
x  = np.linspace(-L/2, L/2, N)

# ═══════════════════════════════════════════════════════════════════════
# 2. k-SPACE GRID AND KINETIC HALF-STEP PROPAGATOR
# ═══════════════════════════════════════════════════════════════════════

k        = fftfreq(N, d=dx) * 2 * np.pi     # wavenumber grid
k2       = k**2
kin_half = np.exp(-1j * hbar * k2 * dt / (4.0 * m))   # exp(−iKΔt/2)

# ═══════════════════════════════════════════════════════════════════════
# 3. INITIAL WAVE FUNCTIONS  (Eqs. 11–12)
#    ψᵢ(t₀, x) = (1/2πσ²)^{1/4} exp[−(x−Xᵢ₀)²/2σ²] exp(iφᵢ)
#    Initial phases φ₁ = 0, φ₂ = 0.1 kept from original code.
# ═══════════════════════════════════════════════════════════════════════

X1_0 = -a / 2
X2_0 =  a / 2

def gauss_cwf(X0, phi):
    """Normalised Gaussian CWF centred at X0 with global phase φ."""
    psi = np.exp(-((x - X0)**2) / (2.0 * sigma**2) + 1j * phi)
    return psi / np.sqrt(np.sum(np.abs(psi)**2) * dx)

psi1 = gauss_cwf(X1_0, phi=0.0)
psi2 = gauss_cwf(X2_0, phi=0.1)

X1, X2 = X1_0, X2_0
V1, V2 = 0.0, 0.0     # Bohmian velocities at t₀

# ═══════════════════════════════════════════════════════════════════════
# 4. HELPER: SOFTENED DISTANCE
#    |a − b|_ε = sqrt((a−b)² + ε²)   (avoids 1/r singularity)
# ═══════════════════════════════════════════════════════════════════════

def sdist(a_val, b_val):
    """Element-wise softened |a − b| = sqrt((a−b)² + SOFT²)."""
    return np.sqrt((a_val - b_val)**2 + SOFT**2)

# ═══════════════════════════════════════════════════════════════════════
# 5. FEEDBACK FORCE FUNCTIONS  (Eqs. 7–8)
# ═══════════════════════════════════════════════════════════════════════

def F_1st_p1(Xi, Xj):
    """
    First-order feedback on particle 1 (Eq. 7, x-array form):
      F₁^(1)(x, Xⱼ) = Gm² [ (x−Xⱼ)/|x−Xⱼ|³ − (Xᵢ−Xⱼ)/|Xᵢ−Xⱼ|³ ]
    Both terms carry Gm² (Bug 1 fix).
    """
    d_x  = sdist(x, Xj)           # shape (N,)
    d_ij = sdist(Xi, Xj)          # scalar
    return G * m**2 * (
        (x - Xj) / d_x**3
        - (Xi - Xj) / d_ij**3
    )

def F_1st_p2(Xi, Xj):
    """
    First-order feedback on particle 2 (Eq. 3b, x-array form):
      F₂^(1)(x, Xᵢ) = Gm² [ (Xᵢ−x)/|Xᵢ−x|³ − (Xᵢ−Xⱼ)/|Xᵢ−Xⱼ|³ ]
    Uses (Xᵢ−x) not (x−Xᵢ) — this is Bug 2 of the original code.
    For particle 2: Xᵢ = X1 (the other particle), Xⱼ = X2 (self).
    """
    d_x  = sdist(Xi, x)           # shape (N,)  — note: Xi first
    d_ij = sdist(Xi, Xj)          # scalar
    return G * m**2 * (
        (Xi - x) / d_x**3
        - (Xi - Xj) / d_ij**3
    )

def F_2nd_p1(Xi, Xj):
    """
    Second-order feedback on particle 1 (Eq. 8, x-array form):
      F₁^(2)(x, Xⱼ) = Gm² [ (3(x−Xⱼ)³ − |x−Xⱼ|)/|x−Xⱼ|⁶
                              − (3(Xᵢ−Xⱼ)³ − |Xᵢ−Xⱼ|)/|Xᵢ−Xⱼ|⁶ ]
    Both terms carry Gm² (Bug 1 fix).
    """
    d_x  = sdist(x, Xj)
    d_ij = sdist(Xi, Xj)
    t1 = (3.0 * (x  - Xj)**3 - d_x ) / d_x**6
    t2 = (3.0 * (Xi - Xj)**3 - d_ij) / d_ij**6
    return G * m**2 * (t1 - t2)

def F_2nd_p2(Xi, Xj):
    """
    Second-order feedback on particle 2 (Eq. 8 analogue, x-array form):
      F₂^(2)(x, Xᵢ) = Gm² [ (3(Xᵢ−x)³ − |Xᵢ−x|)/|Xᵢ−x|⁶
                              − (3(Xᵢ−Xⱼ)³ − |Xᵢ−Xⱼ|)/|Xᵢ−Xⱼ|⁶ ]
    Uses (Xᵢ−x) not (x−Xᵢ) — this is Bug 2 of the original code.
    For particle 2: Xᵢ = X1, Xⱼ = X2.
    """
    d_x  = sdist(Xi, x)
    d_ij = sdist(Xi, Xj)
    t1 = (3.0 * (Xi - x )**3 - d_x ) / d_x**6
    t2 = (3.0 * (Xi - Xj)**3 - d_ij) / d_ij**6
    return G * m**2 * (t1 - t2)

# ═══════════════════════════════════════════════════════════════════════
# 6. WAVE-FUNCTION EVOLUTION  (split-operator + feedback, Eqs. B2, B4)
#
#    One full time step Δt for CWF ψᵢ of particle i:
#
#      (i)   Half kinetic step in k-space
#      (ii)  Full conditional-potential step in x-space  (Eq. B4, first term)
#      (iii) First-order feedback → amplitude modulation  (Eq. B2)
#              r₁(t+Δt) = r₁(t) · exp( Vⱼ · F^(1) · Δt / ħ )
#      (iv)  Second-order feedback → phase modulation  (Eq. B4)
#              s₁(t+Δt) = s₁(t) − (ħ/2m) · F^(2) · Δt
#      (v)   Half kinetic step in k-space
#      (vi)  Renormalise
#
#    Bug 3 of the original code is fixed here: feedback forces are applied
#    in x-space, separately from the kinetic term which lives in k-space.
# ═══════════════════════════════════════════════════════════════════════

def evolve_p1(psi, V2_cur):
    """Advance ψ₁ one step Δt; V2_cur = Ẋ₂(tₙ) from guidance equation."""
    # (i) half kinetic step in k-space
    psi = ifft(fft(psi) * kin_half)

    # (ii) full conditional-potential step in x-space
    V_cond = -G * m**2 / sdist(x, X2)
    psi *= np.exp(-1j * V_cond * dt / hbar)

    # (iii) first-order feedback → amplitude  (Eq. B2)
    F1 = F_1st_p1(X1, X2)
    psi *= np.exp(V2_cur / hbar * F1 * dt)          # real multiplicative

    # (iv) second-order feedback → phase  (Eq. B4)
    F2 = F_2nd_p1(X1, X2)
    psi *= np.exp(-1j * (hbar / (2.0 * m)) * F2 * dt)

    # (v) half kinetic step in k-space
    psi = ifft(fft(psi) * kin_half)

    # (vi) renormalise
    psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx)
    return psi

def evolve_p2(psi, V1_cur):
    """Advance ψ₂ one step Δt; V1_cur = Ẋ₁(tₙ) from guidance equation."""
    # (i) half kinetic step in k-space
    psi = ifft(fft(psi) * kin_half)

    # (ii) full conditional-potential step in x-space
    V_cond = -G * m**2 / sdist(x, X1)
    psi *= np.exp(-1j * V_cond * dt / hbar)

    # (iii) first-order feedback → amplitude  (Eq. B2, particle 2 analogue)
    F1 = F_1st_p2(X1, X2)
    psi *= np.exp(V1_cur / hbar * F1 * dt)

    # (iv) second-order feedback → phase  (Eq. B4, particle 2 analogue)
    F2 = F_2nd_p2(X1, X2)
    psi *= np.exp(-1j * (hbar / (2.0 * m)) * F2 * dt)

    # (v) half kinetic step in k-space
    psi = ifft(fft(psi) * kin_half)

    # (vi) renormalise
    psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx)
    return psi

# ═══════════════════════════════════════════════════════════════════════
# 7. BOHMIAN GUIDANCE EQUATION  (Eqs. 5–6)
#
#    Vᵢ(tₙ) = (ħ/m) Im(∂ₓψᵢ/ψᵢ)|_{x=Xᵢ}
#            = (ħ/m) Im(∂ₓ log ψᵢ)|_{x=Xᵢ}
#
#    Bug 4 fix: pass dx to np.gradient (original used index-spacing = 1).
# ═══════════════════════════════════════════════════════════════════════

def bohm_velocity(psi, Xi):
    """Bohmian velocity at particle position Xi."""
    grad_log = np.gradient(np.log(psi + EPS), dx)   # Bug 4 fix: dx supplied
    idx = np.argmin(np.abs(x - Xi))
    return (hbar / m) * np.imag(grad_log[idx])

# ═══════════════════════════════════════════════════════════════════════
# 8. MAIN TIME-EVOLUTION LOOP
# ═══════════════════════════════════════════════════════════════════════

n_steps  = int(T / dt)
traj1    = np.empty(n_steps + 1)
traj2    = np.empty(n_steps + 1)
traj1[0] = X1
traj2[0] = X2

for i in range(n_steps):

    # Evolve CWFs with feedback from the other particle (§III.A, Eq. 4)
    psi1 = evolve_p1(psi1, V2)
    psi2 = evolve_p2(psi2, V1)

    # Update Bohmian velocities from evolved CWFs (Eqs. 5–6)
    V1 = bohm_velocity(psi1, X1)
    V2 = bohm_velocity(psi2, X2)

    # Update particle positions (Euler step)
    X1 += V1 * dt
    X2 += V2 * dt

    traj1[i + 1] = X1
    traj2[i + 1] = X2

# ═══════════════════════════════════════════════════════════════════════
# 9. PLOT  (matches style of Fig. 2 in arXiv:2505.10221)
# ═══════════════════════════════════════════════════════════════════════

time_arr = np.linspace(0, T, n_steps + 1)

plt.rcParams["font.family"] = "Times New Roman"

fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
ax.plot(time_arr, traj1, label="Particle 1", linewidth=0.8)
ax.plot(time_arr, traj2, label="Particle 2", linewidth=0.8)

ax.set_xlabel("Time")
ax.set_ylabel("Position")
ax.set_title("Particle Trajectories with Feedback")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("trajectories_corrected.png", dpi=300, bbox_inches="tight")
print("Saved → trajectories_corrected.png")
plt.show()