import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation

# ---------------------------------------------------------------
# Driven damped oscillator: steady-state analytical solution
#   x(t) = A(Omega) * cos(Omega t - phi)
# ---------------------------------------------------------------
def amplitude(Omega, W0, zeta, F0, m):
    den = np.sqrt((W0**2 - Omega**2)**2 + (2 * zeta * W0 * Omega)**2)
    return (F0 / m) / den


def phase(Omega, W0, zeta):
    return np.arctan2(2 * zeta * W0 * Omega, W0**2 - Omega**2)


# ---------------------------------------------------------------
# Figure layout
# ---------------------------------------------------------------
fig = plt.figure(figsize=(14, 9))
fig.subplots_adjust(left=0.28, bottom=0.30, right=0.97, top=0.95,
                    hspace=0.55, wspace=0.30)
gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 1, 1])
ax_spring = fig.add_subplot(gs[0, :])
ax_time   = fig.add_subplot(gs[1, :])
ax_res    = fig.add_subplot(gs[2, 0])
ax_phase  = fig.add_subplot(gs[2, 1])

# --- Spring animation ---
ax_spring.set_xlim(-3, 3)
ax_spring.set_ylim(-0.6, 0.6)
ax_spring.set_aspect("equal")
ax_spring.set_title("Driven Mass–Spring Animation")
ax_spring.axhline(0, color="gray", lw=0.5)
ax_spring.axvline(0, color="gray", lw=0.3, ls=":")
ax_spring.axvline(-3, color="k", lw=3)
(spring_line,) = ax_spring.plot([], [], "k-", lw=1.5)
(mass_dot,) = ax_spring.plot([], [], "o", ms=25, color="tab:blue")
(force_arrow,) = ax_spring.plot([], [], "r-", lw=2)  # drive force indicator
time_text = ax_spring.text(0.02, 0.88, "", transform=ax_spring.transAxes)

# --- Time-domain trace: drive F(t) and response x(t) ---
ax_time.set_xlabel("t [s]")
ax_time.set_title("Steady-state response x(t) vs drive F(t)")
ax_time.grid(True, alpha=0.3)
(line_x,) = ax_time.plot([], [], "b-", lw=1.5, label="x(t) [m]")
(line_F,) = ax_time.plot([], [], "r--", lw=1.2, label="F(t)/k [m] (scaled)")
(xhead,) = ax_time.plot([], [], "bo", ms=6)
ax_time.legend(loc="upper right", fontsize=9)

# --- Resonance curve A(Omega) ---
ax_res.set_xlabel(r"Driving freq $\Omega$ [rad/s]")
ax_res.set_ylabel(r"Amplitude $A(\Omega)$ [m]")
ax_res.set_title("Resonance curve")
ax_res.grid(True, alpha=0.3)
(res_curve,) = ax_res.plot([], [], "b-", lw=1.5)
(res_marker,) = ax_res.plot([], [], "ro", ms=8)
res_W0_line = ax_res.axvline(0, color="k", ls="--", alpha=0.5)

# --- Phase lag curve phi(Omega) ---
ax_phase.set_xlabel(r"Driving freq $\Omega$ [rad/s]")
ax_phase.set_ylabel(r"Phase lag $\varphi$ [rad]")
ax_phase.set_title("Phase lag")
ax_phase.grid(True, alpha=0.3)
(phase_curve,) = ax_phase.plot([], [], "g-", lw=1.5)
(phase_marker,) = ax_phase.plot([], [], "ro", ms=8)
ax_phase.set_ylim(0, np.pi)
ax_phase.set_yticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
ax_phase.set_yticklabels(["0", "π/4", "π/2", "3π/4", "π"])

# ---------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------
axc = "#f0f0f0"
sl_m  = Slider(fig.add_axes([0.07, 0.22, 0.17, 0.025], facecolor=axc), "m [kg]",   0.1, 5.0,  valinit=1.0)
sl_k  = Slider(fig.add_axes([0.07, 0.18, 0.17, 0.025], facecolor=axc), "k [N/m]",  1.0, 50.0, valinit=16.0)
sl_z  = Slider(fig.add_axes([0.07, 0.14, 0.17, 0.025], facecolor=axc), "zeta",     0.01, 1.5, valinit=0.15)
sl_F0 = Slider(fig.add_axes([0.07, 0.10, 0.17, 0.025], facecolor=axc), "F0 [N]",   0.1, 10.0, valinit=2.0)
sl_W  = Slider(fig.add_axes([0.07, 0.06, 0.17, 0.025], facecolor=axc), "Ω [rad/s]",0.1, 15.0, valinit=4.0)

btn_reset = Button(fig.add_axes([0.07, 0.01, 0.17, 0.035]), "Reset / Restart")

# ---------------------------------------------------------------
# State
# ---------------------------------------------------------------
state = {"t": None, "x": None, "F": None, "frame": 0, "dt": 0.02}


def recompute():
    m, k = sl_m.val, sl_k.val
    zeta, F0, W = sl_z.val, sl_F0.val, sl_W.val
    W0 = np.sqrt(k / m)

    # Resonance & phase curves
    W_arr = np.linspace(0.01, 3.0 * W0, 600)
    A_arr = amplitude(W_arr, W0, zeta, F0, m)
    phi_arr = phase(W_arr, W0, zeta)

    res_curve.set_data(W_arr, A_arr)
    ax_res.set_xlim(0, W_arr.max())
    ax_res.set_ylim(0, 1.15 * A_arr.max())
    res_W0_line.set_xdata([W0, W0])

    phase_curve.set_data(W_arr, phi_arr)
    ax_phase.set_xlim(0, W_arr.max())

    # Current operating point
    A_now = amplitude(W, W0, zeta, F0, m)
    phi_now = phase(W, W0, zeta)
    res_marker.set_data([W], [A_now])
    phase_marker.set_data([W], [phi_now])

    # Time traces: show ~6 drive periods
    T = 2 * np.pi / W
    tmax = max(6 * T, 4.0)
    t = np.arange(0, tmax, state["dt"])
    x = A_now * np.cos(W * t - phi_now)
    F = (F0 / k) * np.cos(W * t)  # scaled for comparable visual

    state.update(t=t, x=x, F=F, frame=0,
                 A_now=A_now, W=W, F0=F0, k=k)

    ax_time.set_xlim(0, tmax)
    ymax = max(1.15 * max(abs(x).max(), abs(F).max()), 0.1)
    ax_time.set_ylim(-ymax, ymax)
    line_x.set_data(t, x)
    line_F.set_data(t, F)


def draw_spring(x_mass):
    N = 25
    xs = np.linspace(-3, x_mass, N)
    ys = 0.15 * np.sin(np.linspace(0, 7 * np.pi, N))
    ys[0] = ys[-1] = 0
    spring_line.set_data(xs, ys)


def update_anim(_):
    i = state["frame"] % len(state["t"])
    x = state["x"][i]
    ti = state["t"][i]
    draw_spring(x)
    mass_dot.set_data([x], [0])
    xhead.set_data([ti], [x])

    # Red horizontal arrow at mass, length ∝ instantaneous drive force
    F_inst = state["F0"] * np.cos(state["W"] * ti)
    L = 0.4 * F_inst / max(state["F0"], 1e-9)
    force_arrow.set_data([x, x + L], [0.25, 0.25])

    time_text.set_text(
        f"t = {ti:5.2f} s   Ω = {state['W']:.2f} rad/s   "
        f"A = {state['A_now']:.3f} m"
    )
    state["frame"] += 1
    return spring_line, mass_dot, xhead, force_arrow, time_text


def on_change(_=None): recompute()
for s in (sl_m, sl_k, sl_z, sl_F0, sl_W):
    s.on_changed(on_change)
btn_reset.on_clicked(lambda _: state.update(frame=0))

recompute()
ani = FuncAnimation(fig, update_anim, interval=30, blit=False,
                    cache_frame_data=False)
plt.show()