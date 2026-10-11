import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, Button
from matplotlib.animation import FuncAnimation

# ---------------------------------------------------------------
# Analytical solutions
# ---------------------------------------------------------------
def sho(t, x0, v0, m, k):
    W = np.sqrt(k / m)
    x = x0 * np.cos(W * t) + (v0 / W) * np.sin(W * t)
    v = -x0 * W * np.sin(W * t) + v0 * np.cos(W * t)
    return x, v


def dho(t, x0, v0, m, k, zeta):
    W = np.sqrt(k / m)
    if zeta < 1.0:
        wd = W * np.sqrt(1 - zeta**2)
        d = zeta * W
        A = x0
        B = (v0 + d * x0) / wd
        e = np.exp(-d * t)
        x = e * (A * np.cos(wd * t) + B * np.sin(wd * t))
        v = -d * x + e * (-A * wd * np.sin(wd * t) + B * wd * np.cos(wd * t))
    elif np.isclose(zeta, 1.0):
        C1, C2 = x0, v0 + W * x0
        e = np.exp(-W * t)
        x = (C1 + C2 * t) * e
        v = (C2 - W * (C1 + C2 * t)) * e
    else:
        s = np.sqrt(zeta**2 - 1)
        g1, g2 = -W * (zeta - s), -W * (zeta + s)
        C2 = (v0 - g1 * x0) / (g2 - g1)
        C1 = x0 - C2
        x = C1 * np.exp(g1 * t) + C2 * np.exp(g2 * t)
        v = C1 * g1 * np.exp(g1 * t) + C2 * g2 * np.exp(g2 * t)
    return x, v


# ---------------------------------------------------------------
# Figure layout (2x2 grid of plots + widgets on the left)
# ---------------------------------------------------------------
fig = plt.figure(figsize=(14, 9))
fig.subplots_adjust(left=0.28, bottom=0.28, right=0.97, top=0.95,
                    hspace=0.45, wspace=0.30)

gs = fig.add_gridspec(2, 2)
ax_spring = fig.add_subplot(gs[0, :])       # top: full width spring
ax_traj   = fig.add_subplot(gs[1, 0])       # bottom-left: x(t)
ax_energy = fig.add_subplot(gs[1, 1])       # bottom-right: energies
# phase space will be an inset-like extra window via gridspec; use another figure area:
# To keep it visible, place it in the bottom portion by splitting ax_energy region.
# Simpler: add a 3rd row.
fig.clf()
fig.subplots_adjust(left=0.28, bottom=0.28, right=0.97, top=0.95,
                    hspace=0.55, wspace=0.30)
gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1, 1])
ax_spring = fig.add_subplot(gs[0, :])
ax_traj   = fig.add_subplot(gs[1, 0])
ax_energy = fig.add_subplot(gs[1, 1])
ax_phase  = fig.add_subplot(gs[2, :])

# --- Spring animation axis ---
ax_spring.set_xlim(-2.5, 2.5)
ax_spring.set_ylim(-0.6, 0.6)
ax_spring.set_aspect("equal")
ax_spring.set_title("Mass–Spring Animation")
ax_spring.axhline(0, color="gray", lw=0.5)
ax_spring.axvline(0, color="gray", lw=0.3, ls=":")
ax_spring.axvline(-2.5, color="k", lw=3)
(spring_line,) = ax_spring.plot([], [], "k-", lw=1.5)
(mass_dot,) = ax_spring.plot([], [], "o", ms=25, color="tab:blue")
time_text = ax_spring.text(0.02, 0.88, "", transform=ax_spring.transAxes)

# --- Trajectory x(t) ---
ax_traj.set_xlabel("t [s]")
ax_traj.set_ylabel("x(t) [m]")
ax_traj.set_title("Displacement")
ax_traj.grid(True, alpha=0.3)
(line_full,) = ax_traj.plot([], [], "b-", lw=1, alpha=0.5)
(line_head,) = ax_traj.plot([], [], "ro", ms=6)
(line_env_p,) = ax_traj.plot([], [], "r--", lw=1, alpha=0.7)
(line_env_m,) = ax_traj.plot([], [], "r--", lw=1, alpha=0.7)

# --- Energy plot ---
ax_energy.set_xlabel("t [s]")
ax_energy.set_ylabel("Energy [J]")
ax_energy.set_title("Energies  K, U, E")
ax_energy.grid(True, alpha=0.3)
(line_K,) = ax_energy.plot([], [], "r-", lw=1.2, label="K(t)")
(line_U,) = ax_energy.plot([], [], "b-", lw=1.2, label="U(t)")
(line_E,) = ax_energy.plot([], [], "k--", lw=1.5, label="E(t)")
(e_head,) = ax_energy.plot([], [], "ko", ms=5)
ax_energy.legend(loc="upper right", fontsize=8)

# --- Phase space (x, v) ---
ax_phase.set_xlabel("x [m]")
ax_phase.set_ylabel("v [m/s]")
ax_phase.set_title("Phase space (x, v)")
ax_phase.grid(True, alpha=0.3)
ax_phase.axhline(0, color="gray", lw=0.4)
ax_phase.axvline(0, color="gray", lw=0.4)
(phase_full,) = ax_phase.plot([], [], "g-", lw=1, alpha=0.5)
(phase_head,) = ax_phase.plot([], [], "ro", ms=6)

# ---------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------
axc = "#f0f0f0"
sl_m  = Slider(fig.add_axes([0.07, 0.20, 0.17, 0.025], facecolor=axc), "m [kg]",   0.1, 5.0, valinit=1.0)
sl_k  = Slider(fig.add_axes([0.07, 0.16, 0.17, 0.025], facecolor=axc), "k [N/m]",  1.0, 50.0, valinit=16.0)
sl_x0 = Slider(fig.add_axes([0.07, 0.12, 0.17, 0.025], facecolor=axc), "x0 [m]",  -2.0, 2.0, valinit=1.0)
sl_v0 = Slider(fig.add_axes([0.07, 0.08, 0.17, 0.025], facecolor=axc), "v0 [m/s]",-5.0, 5.0, valinit=0.0)
sl_z  = Slider(fig.add_axes([0.07, 0.04, 0.17, 0.025], facecolor=axc), "zeta",     0.0, 3.0, valinit=0.15)

ax_radio = fig.add_axes([0.02, 0.70, 0.20, 0.14], facecolor=axc)
radio = RadioButtons(ax_radio, ("SHO (ζ=0)", "Damped (use ζ)"), active=0)

btn_reset = Button(fig.add_axes([0.04, 0.62, 0.16, 0.05]), "Reset / Restart")

# ---------------------------------------------------------------
# State
# ---------------------------------------------------------------
state = {"t": None, "x": None, "v": None, "K": None, "U": None, "E": None,
         "frame": 0, "tmax": 10.0, "dt": 0.02}


def recompute():
    m, k = sl_m.val, sl_k.val
    x0, v0 = sl_x0.val, sl_v0.val
    zeta = 0.0 if radio.value_selected.startswith("SHO") else sl_z.val
    W = np.sqrt(k / m)
    state["tmax"] = max(6 * 2 * np.pi / W, 10.0)
    t = np.arange(0, state["tmax"], state["dt"])

    if zeta == 0.0:
        x, v = sho(t, x0, v0, m, k)
        line_env_p.set_data([], []); line_env_m.set_data([], [])
    else:
        x, v = dho(t, x0, v0, m, k, zeta)
        if zeta < 1.0:
            env = np.abs(x0) * np.exp(-zeta * W * t) + 1e-9
            line_env_p.set_data(t, env); line_env_m.set_data(t, -env)
        else:
            line_env_p.set_data([], []); line_env_m.set_data([], [])

    K = 0.5 * m * v**2
    U = 0.5 * k * x**2
    E = K + U

    state.update(t=t, x=x, v=v, K=K, U=U, E=E, frame=0)

    # axes limits
    ax_traj.set_xlim(0, state["tmax"])
    ymax = max(1.1 * np.max(np.abs(x)), 0.1)
    ax_traj.set_ylim(-ymax, ymax)
    line_full.set_data(t, x)

    ax_energy.set_xlim(0, state["tmax"])
    ax_energy.set_ylim(0, 1.1 * max(E.max(), 1e-6))
    line_K.set_data(t, K); line_U.set_data(t, U); line_E.set_data(t, E)

    vmax = max(1.1 * np.max(np.abs(v)), 0.1)
    ax_phase.set_xlim(-ymax, ymax)
    ax_phase.set_ylim(-vmax, vmax)
    phase_full.set_data(x, v)


def draw_spring(x_mass):
    N = 20
    xs = np.linspace(-2.5, x_mass, N)
    ys = 0.15 * np.sin(np.linspace(0, 6 * np.pi, N))
    ys[0] = ys[-1] = 0
    spring_line.set_data(xs, ys)


def update_anim(_):
    i = state["frame"] % len(state["t"])
    x = state["x"][i]; v = state["v"][i]; ti = state["t"][i]
    draw_spring(x)
    mass_dot.set_data([x], [0])
    line_head.set_data([ti], [x])
    e_head.set_data([ti], [state["E"][i]])
    phase_head.set_data([x], [v])
    time_text.set_text(f"t = {ti:5.2f} s   x = {x:+.3f} m   v = {v:+.3f} m/s")
    state["frame"] += 1
    return (spring_line, mass_dot, line_head, e_head, phase_head, time_text)


def on_change(_=None): recompute()
for s in (sl_m, sl_k, sl_x0, sl_v0, sl_z): s.on_changed(on_change)
radio.on_clicked(on_change)
btn_reset.on_clicked(lambda _: state.update(frame=0))

recompute()
ani = FuncAnimation(fig, update_anim, interval=30, blit=False, cache_frame_data=False)
plt.show()