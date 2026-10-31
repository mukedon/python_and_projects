import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button


# ==============================================================
# 1. CONFIGURATION
# ==============================================================

plt.rcParams["font.family"] = "DejaVu Sans"

BG = "#f3f6fa"
PANEL = "#ffffff"
TEXT = "#263445"
MUTED = "#66788a"
BLUE = "#2878b5"
ORANGE = "#e58a36"
GREEN = "#299b75"
RED = "#d9534f"
GRID = "#dce4ed"

# Initial parameters
M0 = 1.0
K0 = 16.0
ZETA0 = 0.15
F00 = 2.0
OMEGA0_DRIVE = 4.0

DT = 0.02
ANIMATION_INTERVAL = 25


# ==============================================================
# 2. ANALYTICAL STEADY-STATE SOLUTION
# ==============================================================

def amplitude(Omega, W0, zeta, F0, m):
    """
    Steady-state displacement amplitude of a driven damped
    harmonic oscillator.

    x(t) = A(Omega) * cos(Omega*t - phi)

    Omega : driving angular frequency
    W0    : natural angular frequency
    zeta  : damping ratio
    F0    : driving-force amplitude
    m     : mass
    """
    denominator = np.sqrt(
        (W0**2 - Omega**2)**2
        + (2.0 * zeta * W0 * Omega)**2
    )

    return (F0 / m) / denominator


def phase(Omega, W0, zeta):
    """
    Phase lag of the steady-state displacement relative
    to the driving force.
    """
    return np.arctan2(
        2.0 * zeta * W0 * Omega,
        W0**2 - Omega**2
    )


def oscillator_state(t, W, A, phi):
    """Return steady-state displacement and velocity."""
    x = A * np.cos(W * t - phi)
    v = -A * W * np.sin(W * t - phi)

    return x, v


# ==============================================================
# 3. FIGURE AND LAYOUT
# ==============================================================

fig = plt.figure(figsize=(15, 9))
fig.patch.set_facecolor(BG)

fig.suptitle(
    "DRIVEN DAMPED HARMONIC OSCILLATOR",
    fontsize=18,
    fontweight="bold",
    color=TEXT,
    y=0.975,
)

fig.text(
    0.50,
    0.945,
    "Explore resonance, phase lag, and steady-state motion",
    ha="center",
    fontsize=10,
    color=MUTED,
)

# Reserve the right-hand side for the control panel.
gs = fig.add_gridspec(
    3,
    2,
    left=0.07,
    right=0.69,
    bottom=0.09,
    top=0.90,
    height_ratios=[1.05, 1.0, 1.0],
    hspace=0.52,
    wspace=0.34,
)

ax_spring = fig.add_subplot(gs[0, :])
ax_time = fig.add_subplot(gs[1, :])
ax_res = fig.add_subplot(gs[2, 0])
ax_phase = fig.add_subplot(gs[2, 1])


def style_axes(ax, title, xlabel=None, ylabel=None):
    """Apply consistent styling to an axes."""
    ax.set_facecolor(PANEL)

    ax.set_title(
        title,
        fontsize=11,
        fontweight="bold",
        color=TEXT,
        pad=9,
    )

    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, color=TEXT)

    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=TEXT)

    ax.tick_params(labelsize=8, colors=MUTED)
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.9)

    for spine in ax.spines.values():
        spine.set_color(GRID)


style_axes(ax_spring, "MASS–SPRING SYSTEM")
style_axes(
    ax_time,
    "STEADY-STATE RESPONSE",
    xlabel="Time (s)",
    ylabel="Displacement (m)",
)
style_axes(
    ax_res,
    "RESONANCE CURVE",
    xlabel=r"Driving frequency $\Omega$ (rad/s)",
    ylabel=r"Amplitude $A(\Omega)$ (m)",
)
style_axes(
    ax_phase,
    "PHASE LAG",
    xlabel=r"Driving frequency $\Omega$ (rad/s)",
    ylabel=r"Phase lag $\phi$ (rad)",
)


# ==============================================================
# 4. SPRING ANIMATION
# ==============================================================

ax_spring.set_xlim(-3.3, 3.3)
ax_spring.set_ylim(-0.65, 0.70)
ax_spring.set_xticks([])
ax_spring.set_yticks([])
ax_spring.grid(False)

# Fixed wall
ax_spring.plot(
    [-3.0, -3.0],
    [-0.25, 0.25],
    color=TEXT,
    linewidth=4,
)

for y in np.linspace(-0.23, 0.23, 9):
    ax_spring.plot(
        [-3.20, -3.0],
        [y - 0.06, y],
        color=MUTED,
        linewidth=1,
    )

# Equilibrium position
ax_spring.axvline(
    0,
    color=GRID,
    linestyle="--",
    linewidth=1.2,
)

ax_spring.text(
    0,
    -0.39,
    "Equilibrium",
    ha="center",
    fontsize=8,
    color=MUTED,
)

# Floor
ax_spring.plot(
    [-3.1, 3.1],
    [-0.30, -0.30],
    color=GRID,
    linewidth=2,
)

# Animated spring
spring_line, = ax_spring.plot(
    [],
    [],
    color=BLUE,
    linewidth=2.3,
)

# Animated mass
mass_dot, = ax_spring.plot(
    [],
    [],
    marker="o",
    markersize=23,
    color=ORANGE,
    markeredgecolor="white",
    markeredgewidth=1.5,
    linestyle="None",
)

# Driving-force arrow
force_arrow, = ax_spring.plot(
    [],
    [],
    color=RED,
    linewidth=2.5,
)

# Animated time display
time_text = ax_spring.text(
    0.02,
    0.92,
    "",
    transform=ax_spring.transAxes,
    fontsize=9,
    color=TEXT,
    va="top",
)

ax_spring.text(
    0.98,
    0.92,
    "Red arrow: driving force",
    transform=ax_spring.transAxes,
    fontsize=8,
    color=RED,
    ha="right",
    va="top",
)


def draw_spring(x_mass):
    """Draw a zigzag spring between the wall and the mass."""
    x_start = -3.0
    x_end = x_mass - 0.20

    n_coils = 12
    n_points = 2 * n_coils + 2

    xs = np.linspace(x_start, x_end, n_points)
    ys = np.zeros(n_points)

    ys[1:-1:2] = 0.13
    ys[2:-1:2] = -0.13

    # Keep the endpoints attached to the wall and mass.
    ys[0] = 0.0
    ys[-1] = 0.0

    spring_line.set_data(xs, ys)


# ==============================================================
# 5. TIME-DOMAIN PLOT
# ==============================================================

ax_time.axhline(
    0,
    color=MUTED,
    linewidth=0.8,
    alpha=0.7,
)

line_x, = ax_time.plot(
    [],
    [],
    color=BLUE,
    linewidth=2,
    label=r"Response $x(t)$",
)

line_F, = ax_time.plot(
    [],
    [],
    color=RED,
    linestyle="--",
    linewidth=1.5,
    label=r"Driving force $F(t)/k$",
)

xhead, = ax_time.plot(
    [],
    [],
    marker="o",
    color=ORANGE,
    markersize=7,
    linestyle="None",
)

ax_time.legend(
    loc="upper right",
    fontsize=8,
    frameon=False,
)


# ==============================================================
# 6. RESONANCE AND PHASE PLOTS
# ==============================================================

res_curve, = ax_res.plot(
    [],
    [],
    color=BLUE,
    linewidth=2,
)

res_marker, = ax_res.plot(
    [],
    [],
    marker="o",
    color=RED,
    markersize=7,
    linestyle="None",
)

res_W0_line = ax_res.axvline(
    0,
    color=MUTED,
    linestyle="--",
    linewidth=1.2,
    alpha=0.8,
    label=r"$\omega_0$",
)

ax_res.legend(
    loc="upper right",
    fontsize=8,
    frameon=False,
)

phase_curve, = ax_phase.plot(
    [],
    [],
    color=GREEN,
    linewidth=2,
)

phase_marker, = ax_phase.plot(
    [],
    [],
    marker="o",
    color=RED,
    markersize=7,
    linestyle="None",
)

ax_phase.set_ylim(0, np.pi)
ax_phase.set_yticks(
    [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]
)
ax_phase.set_yticklabels(
    ["0", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"]
)


# ==============================================================
# 7. CONTROL PANEL
# ==============================================================

# Background panel
ax_panel = fig.add_axes([0.725, 0.06, 0.25, 0.85])
ax_panel.set_facecolor(PANEL)
ax_panel.set_xticks([])
ax_panel.set_yticks([])

for spine in ax_panel.spines.values():
    spine.set_color(GRID)

ax_panel.set_zorder(-1)

fig.text(
    0.85,
    0.875,
    "CONTROL PANEL",
    ha="center",
    fontsize=13,
    fontweight="bold",
    color=TEXT,
)


# --------------------------------------------------------------
# Parameter sliders
# --------------------------------------------------------------

slider_color = "#e8eef5"

sl_m = Slider(
    fig.add_axes([0.790, 0.805, 0.17, 0.025], facecolor=slider_color),
    "Mass m (kg)",
    0.1,
    5.0,
    valinit=M0,
)

sl_k = Slider(
    fig.add_axes([0.790, 0.745, 0.17, 0.025], facecolor=slider_color),
    "Spring k (N/m)",
    1.0,
    50.0,
    valinit=K0,
)

sl_z = Slider(
    fig.add_axes([0.790, 0.685, 0.17, 0.025], facecolor=slider_color),
    "Damping ratio ζ",
    0.01,
    1.5,
    valinit=ZETA0,
)

sl_F0 = Slider(
    fig.add_axes([0.790, 0.625, 0.17, 0.025], facecolor=slider_color),
    "Force F₀ (N)",
    0.1,
    10.0,
    valinit=F00,
)

sl_W = Slider(
    fig.add_axes([0.790, 0.565, 0.17, 0.025], facecolor=slider_color),
    "Frequency Ω (rad/s)",
    0.1,
    15.0,
    valinit=OMEGA0_DRIVE,
)

for slider in (sl_m, sl_k, sl_z, sl_F0, sl_W):
    slider.label.set_color(TEXT)
    slider.label.set_fontsize(8)
    slider.valtext.set_color(BLUE)
    slider.valtext.set_fontsize(8)


# --------------------------------------------------------------
# Playback buttons
# --------------------------------------------------------------

btn_play = Button(
    fig.add_axes([0.755, 0.485, 0.085, 0.045]),
    "Pause",
    color="#e5effa",
    hovercolor="#cfe2f7",
)

btn_reset = Button(
    fig.add_axes([0.855, 0.485, 0.09, 0.045]),
    "Reset",
    color="#fbe8e6",
    hovercolor="#f5d3d0",
)

for btn in (btn_play, btn_reset):
    btn.label.set_color(TEXT)
    btn.label.set_fontsize(9)


# --------------------------------------------------------------
# Animation speed slider
# --------------------------------------------------------------

sl_speed = Slider(
    fig.add_axes([0.785, 0.415, 0.17, 0.025], facecolor=slider_color),
    "Animation speed",
    0.25,
    2.5,
    valinit=1.0,
)

sl_speed.label.set_color(TEXT)
sl_speed.label.set_fontsize(8)
sl_speed.valtext.set_color(GREEN)
sl_speed.valtext.set_fontsize(8)


# --------------------------------------------------------------
# Live numerical readout
# --------------------------------------------------------------

fig.text(
    0.755,
    0.365,
    "LIVE READOUT",
    fontsize=10,
    fontweight="bold",
    color=TEXT,
)

# Use a dedicated axes for the readout.
# This avoids the Matplotlib blitting problem associated with
# figure-level text artists.
ax_readout = fig.add_axes([0.755, 0.105, 0.20, 0.245])
ax_readout.set_axis_off()
ax_readout.set_facecolor(PANEL)

readout = ax_readout.text(
    0.0,
    1.0,
    "",
    transform=ax_readout.transAxes,
    fontsize=9,
    family="monospace",
    color=TEXT,
    va="top",
    linespacing=1.6,
)


# ==============================================================
# 8. STATE
# ==============================================================

state = {
    "t": np.array([0.0]),
    "x": np.array([0.0]),
    "F": np.array([0.0]),
    "frame": 0,
    "dt": DT,
    "A_now": 0.0,
    "phi_now": 0.0,
    "W": OMEGA0_DRIVE,
    "F0": F00,
    "k": K0,
    "m": M0,
    "W0": np.sqrt(K0 / M0),
    "playing": True,
    "speed": 1.0,
}


# ==============================================================
# 9. RECOMPUTE CURVES
# ==============================================================

def recompute(_value=None):
    """Recalculate all plots after a slider change."""

    m = float(sl_m.val)
    k = float(sl_k.val)
    zeta = float(sl_z.val)
    F0 = float(sl_F0.val)
    W = float(sl_W.val)

    W0 = np.sqrt(k / m)

    # ----------------------------------------------------------
    # Resonance and phase curves
    # ----------------------------------------------------------

    W_arr = np.linspace(0.01, 3.0 * W0, 700)

    A_arr = amplitude(W_arr, W0, zeta, F0, m)
    phi_arr = phase(W_arr, W0, zeta)

    res_curve.set_data(W_arr, A_arr)

    ax_res.set_xlim(0, W_arr[-1])

    max_amplitude = max(float(np.max(A_arr)), 0.01)
    ax_res.set_ylim(0, 1.15 * max_amplitude)

    res_W0_line.set_xdata([W0, W0])

    phase_curve.set_data(W_arr, phi_arr)
    ax_phase.set_xlim(0, W_arr[-1])

    # ----------------------------------------------------------
    # Current driving frequency
    # ----------------------------------------------------------

    A_now = float(amplitude(W, W0, zeta, F0, m))
    phi_now = float(phase(W, W0, zeta))

    res_marker.set_data([W], [A_now])
    phase_marker.set_data([W], [phi_now])

    # ----------------------------------------------------------
    # Time-domain curves
    # ----------------------------------------------------------

    period = 2.0 * np.pi / W
    t_max = max(6.0 * period, 4.0)

    t = np.arange(0.0, t_max + DT, DT)

    x = A_now * np.cos(W * t - phi_now)

    # Driving force divided by k, for comparison with displacement.
    F_scaled = (F0 / k) * np.cos(W * t)

    # ----------------------------------------------------------
    # Save the new data
    # ----------------------------------------------------------

    state.update(
        t=t,
        x=x,
        F=F_scaled,
        frame=0,
        A_now=A_now,
        phi_now=phi_now,
        W=W,
        F0=F0,
        k=k,
        m=m,
        W0=W0,
    )

    line_x.set_data(t, x)
    line_F.set_data(t, F_scaled)

    ax_time.set_xlim(0, t_max)

    ymax = max(
        1.15 * float(np.max(np.abs(x))),
        1.15 * float(np.max(np.abs(F_scaled))),
        0.1,
    )

    ax_time.set_ylim(-ymax, ymax)

    # Update the animation immediately at the current time.
    update_current_frame()

    fig.canvas.draw_idle()


# ==============================================================
# 10. UPDATE ANIMATED OBJECTS AND READOUT
# ==============================================================

def update_current_frame():
    """Update the animation objects using the current state."""

    t = state["t"]
    x_data = state["x"]

    if len(t) == 0:
        return

    # Keep frame index within the current time array.
    state["frame"] %= len(t)

    i = state["frame"]

    ti = float(t[i])
    x_now = float(x_data[i])

    W = state["W"]
    F0 = state["F0"]
    A_now = state["A_now"]
    phi_now = state["phi_now"]
    k = state["k"]

    # Instantaneous driving force
    F_inst = F0 * np.cos(W * ti)

    # Spring and mass
    draw_spring(x_now)
    mass_dot.set_data([x_now], [0.0])

    # ----------------------------------------------------------
    # Force indicator
    # ----------------------------------------------------------
    #
    # The arrow points right for positive force and left for
    # negative force. Its length is proportional to |F(t)|.
    #
    arrow_length = 0.65 * F_inst / max(F0, 1e-12)

    arrow_y = 0.30

    force_arrow.set_data(
        [x_now, x_now + arrow_length],
        [arrow_y, arrow_y],
    )

    # Add a visible arrowhead using the line's marker.
    force_arrow.set_marker(">")
    force_arrow.set_markevery([1])
    force_arrow.set_markersize(7)

    # Flip the arrowhead for negative force.
    if F_inst < 0:
        force_arrow.set_marker("<")

    # ----------------------------------------------------------
    # Time-domain marker and time label
    # ----------------------------------------------------------

    xhead.set_data([ti], [x_now])

    time_text.set_text(
        f"t = {ti:5.2f} s    "
        f"Ω = {W:.2f} rad/s    "
        f"A = {A_now:.3f} m"
    )

    # ----------------------------------------------------------
    # Live numerical readout
    # ----------------------------------------------------------

    x_analytical, v_analytical = oscillator_state(
        ti,
        W,
        A_now,
        phi_now,
    )

    F_scaled = F_inst / k

    readout.set_text(
        f"Time       : {ti:7.3f} s\n"
        f"x(t)       : {x_analytical:7.3f} m\n"
        f"v(t)       : {v_analytical:7.3f} m/s\n"
        f"F(t)       : {F_inst:7.3f} N\n"
        f"F(t)/k     : {F_scaled:7.3f} m\n"
        f"Amplitude  : {A_now:7.3f} m\n"
        f"Phase lag  : {phi_now:7.3f} rad\n"
        f"Natural ω₀ : {state['W0']:7.3f} rad/s"
    )


# ==============================================================
# 11. ANIMATION CALLBACK
# ==============================================================

def update_anim(_frame):
    """Advance the animation by one frame."""

    if state["playing"]:
        # Advance according to the speed control.
        dt_frame = state["dt"] * state["speed"]

        # Convert elapsed time into a frame increment.
        frame_increment = max(
            1,
            int(round(dt_frame / state["dt"])),
        )

        state["frame"] += frame_increment
        state["frame"] %= len(state["t"])

    update_current_frame()

    # blit=False is used deliberately for compatibility.
    # Returning the artists is still useful if blitting is
    # enabled later.
    return (
        spring_line,
        mass_dot,
        force_arrow,
        time_text,
        line_x,
        line_F,
        xhead,
        res_curve,
        res_marker,
        phase_curve,
        phase_marker,
        readout,
    )


# ==============================================================
# 12. CONTROL CALLBACKS
# ==============================================================

def on_parameter_change(_value):
    recompute()


def on_speed_change(value):
    state["speed"] = float(value)


def on_play_pause(_event):
    state["playing"] = not state["playing"]

    if state["playing"]:
        btn_play.label.set_text("Pause")
        animation.event_source.start()
    else:
        btn_play.label.set_text("Play")
        animation.event_source.stop()

    fig.canvas.draw_idle()


def on_reset(_event):
    # Restore original parameter values.
    sl_m.reset()
    sl_k.reset()
    sl_z.reset()
    sl_F0.reset()
    sl_W.reset()
    sl_speed.reset()

    state["frame"] = 0
    state["playing"] = True

    btn_play.label.set_text("Pause")

    recompute()

    animation.event_source.start()

    fig.canvas.draw_idle()


# Connect slider callbacks
for slider in (sl_m, sl_k, sl_z, sl_F0, sl_W):
    slider.on_changed(on_parameter_change)

sl_speed.on_changed(on_speed_change)

# Connect buttons
btn_play.on_clicked(on_play_pause)
btn_reset.on_clicked(on_reset)


# ==============================================================
# 13. INITIALIZE AND RUN
# ==============================================================

recompute()

animation = FuncAnimation(
    fig,
    update_anim,
    interval=ANIMATION_INTERVAL,
    blit=False,
    cache_frame_data=False,
)

plt.show()