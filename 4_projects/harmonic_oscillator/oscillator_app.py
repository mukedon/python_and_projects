import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button, RadioButtons


# ============================================================
# 1. CONFIGURATION
# ============================================================

plt.rcParams["font.family"] = "DejaVu Sans"

BG = "#f4f7fb"
PANEL = "#ffffff"
TEXT = "#243447"
MUTED = "#64748b"
BLUE = "#2878b5"
ORANGE = "#e58a36"
GREEN = "#279b75"
RED = "#d9534f"
GRID = "#dce4ed"

m_initial = 1.0
k_initial = 4.0
x0_initial = 1.0
v0_initial = 0.0
zeta_initial = 0.12

T_MAX = 12.0
N_POINTS = 800
t_grid = np.linspace(0.0, T_MAX, N_POINTS)

# Animation state
t_current = 0.0
playing = True
speed_factor = 1.0
mode = "SHO"


# ============================================================
# 2. OSCILLATOR MODEL
# ============================================================

def oscillator(t, m, k, x0, v0, zeta):
    """
    Return displacement x(t) and velocity v(t) for

        x'' + 2*zeta*omega0*x' + omega0**2*x = 0

    using analytical solutions for underdamped, critically
    damped, and overdamped motion.
    """
    t = np.asarray(t, dtype=float)

    omega0 = np.sqrt(k / m)

    # Simple harmonic motion
    if zeta == 0:
        omega = omega0
        x = (
            x0 * np.cos(omega * t)
            + (v0 / omega) * np.sin(omega * t)
        )
        v = (
            -x0 * omega * np.sin(omega * t)
            + v0 * np.cos(omega * t)
        )
        return x, v

    # Underdamped motion
    if zeta < 1.0:
        omega_d = omega0 * np.sqrt(1.0 - zeta**2)
        A = x0
        B = (v0 + zeta * omega0 * x0) / omega_d

        decay = np.exp(-zeta * omega0 * t)
        cosine = np.cos(omega_d * t)
        sine = np.sin(omega_d * t)

        x = decay * (A * cosine + B * sine)

        v = decay * (
            -zeta * omega0 * (A * cosine + B * sine)
            - A * omega_d * sine
            + B * omega_d * cosine
        )

        return x, v

    # Critically damped motion
    if np.isclose(zeta, 1.0):
        B = v0 + omega0 * x0
        decay = np.exp(-omega0 * t)

        x = (x0 + B * t) * decay
        v = (B - omega0 * (x0 + B * t)) * decay

        return x, v

    # Overdamped motion
    discriminant = np.sqrt(zeta**2 - 1.0)

    r1 = -omega0 * (zeta - discriminant)
    r2 = -omega0 * (zeta + discriminant)

    A = (v0 - r2 * x0) / (r1 - r2)
    B = (r1 * x0 - v0) / (r1 - r2)

    exp1 = np.exp(r1 * t)
    exp2 = np.exp(r2 * t)

    x = A * exp1 + B * exp2
    v = A * r1 * exp1 + B * r2 * exp2

    return x, v


def current_parameters():
    """Read the current values from the sliders."""
    m = slider_m.val
    k = slider_k.val
    x0 = slider_x0.val
    v0 = slider_v0.val

    # SHO has no damping; DHO uses the damping slider.
    zeta = 0.0 if mode == "SHO" else slider_zeta.val

    return m, k, x0, v0, zeta


def calculate_energy(m, k, x, v):
    """Calculate kinetic, potential, and total mechanical energy."""
    kinetic = 0.5 * m * v**2
    potential = 0.5 * k * x**2
    total = kinetic + potential

    return kinetic, potential, total


# ============================================================
# 3. FIGURE AND LAYOUT
# ============================================================

fig = plt.figure(figsize=(15, 8.5))
fig.patch.set_facecolor(BG)

fig.suptitle(
    "HARMONIC OSCILLATOR LAB",
    fontsize=19,
    fontweight="bold",
    color=TEXT,
    x=0.39,
    y=0.97,
)

fig.text(
    0.39,
    0.935,
    "Explore displacement, energy, and phase space",
    fontsize=10,
    color=MUTED,
    ha="center",
)


def style_axes(ax, title, xlabel=None, ylabel=None):
    """Apply consistent styling to the plotting axes."""
    ax.set_facecolor(PANEL)
    ax.set_title(title, fontsize=11, fontweight="bold", color=TEXT, pad=9)

    if xlabel:
        ax.set_xlabel(xlabel, color=TEXT)

    if ylabel:
        ax.set_ylabel(ylabel, color=TEXT)

    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.8)

    for spine in ax.spines.values():
        spine.set_color(GRID)


# Main plots
ax_spring = fig.add_axes([0.055, 0.565, 0.30, 0.29])
ax_displacement = fig.add_axes([0.055, 0.165, 0.30, 0.29])
ax_energy = fig.add_axes([0.395, 0.565, 0.30, 0.29])
ax_phase = fig.add_axes([0.395, 0.165, 0.30, 0.29])

style_axes(ax_spring, "MASS–SPRING SYSTEM")
style_axes(
    ax_displacement,
    "DISPLACEMENT",
    xlabel="Time (s)",
    ylabel="Position x (m)",
)
style_axes(
    ax_energy,
    "MECHANICAL ENERGY",
    xlabel="Time (s)",
    ylabel="Energy (J)",
)
style_axes(
    ax_phase,
    "PHASE SPACE",
    xlabel="Position x (m)",
    ylabel="Velocity v (m/s)",
)

# Sidebar background
ax_sidebar_bg = fig.add_axes([0.725, 0.055, 0.255, 0.86])
ax_sidebar_bg.set_facecolor(PANEL)
ax_sidebar_bg.set_xticks([])
ax_sidebar_bg.set_yticks([])

for spine in ax_sidebar_bg.spines.values():
    spine.set_color(GRID)

ax_sidebar_bg.set_zorder(-1)

fig.text(
    0.852,
    0.875,
    "CONTROL PANEL",
    fontsize=13,
    fontweight="bold",
    color=TEXT,
    ha="center",
)


# ============================================================
# 4. MASS–SPRING ANIMATION
# ============================================================

ax_spring.set_xlim(0.0, 1.0)
ax_spring.set_ylim(-0.48, 0.48)
ax_spring.set_xticks([])
ax_spring.set_yticks([])
ax_spring.grid(False)

# Fixed wall and floor
ax_spring.plot(
    [0.12, 0.12],
    [-0.22, 0.22],
    color=MUTED,
    linewidth=5,
    solid_capstyle="round",
)

for y in np.linspace(-0.22, 0.22, 9):
    ax_spring.plot(
        [0.055, 0.12],
        [y - 0.045, y],
        color=MUTED,
        linewidth=1,
    )

ax_spring.plot(
    [0.08, 0.94],
    [-0.28, -0.28],
    color=GRID,
    linewidth=2,
)

# Equilibrium position
equilibrium_x = 0.72

ax_spring.plot(
    [equilibrium_x, equilibrium_x],
    [-0.22, 0.22],
    linestyle="--",
    color=GRID,
    linewidth=1.3,
)

ax_spring.text(
    equilibrium_x,
    -0.36,
    "Equilibrium",
    ha="center",
    fontsize=8,
    color=MUTED,
)

# Animated spring
spring_line, = ax_spring.plot(
    [],
    [],
    color=BLUE,
    linewidth=2.5,
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

# Animated time display; attached to ax_spring
time_text = ax_spring.text(
    0.04,
    0.92,
    "",
    transform=ax_spring.transAxes,
    fontsize=9,
    color=TEXT,
    va="top",
)

ax_spring.text(
    0.04,
    0.80,
    "Mass position is proportional to x(t)",
    transform=ax_spring.transAxes,
    fontsize=8,
    color=MUTED,
    va="top",
)


def make_spring(x_start, x_end, y=0.0, coils=12, amplitude=0.09):
    """Generate coordinates for a zigzag spring."""
    if x_end <= x_start:
        x_end = x_start + 0.01

    n = coils * 2 + 2
    x = np.linspace(x_start, x_end, n)
    y_values = np.zeros_like(x) + y

    y_values[1:-1:2] += amplitude
    y_values[2:-1:2] -= amplitude

    return x, y_values


# ============================================================
# 5. DISPLACEMENT, ENERGY, AND PHASE-SPACE PLOTS
# ============================================================

x_values, v_values = oscillator(
    t_grid,
    m_initial,
    k_initial,
    x0_initial,
    v0_initial,
    zeta_initial,
)

kinetic_values, potential_values, total_values = calculate_energy(
    m_initial,
    k_initial,
    x_values,
    v_values,
)

# Displacement
displacement_line, = ax_displacement.plot(
    t_grid,
    x_values,
    color=BLUE,
    linewidth=2,
    label="x(t)",
)

displacement_point, = ax_displacement.plot(
    [],
    [],
    marker="o",
    color=ORANGE,
    markersize=7,
    linestyle="None",
)

ax_displacement.axhline(0, color=MUTED, linewidth=0.8, alpha=0.7)
ax_displacement.set_xlim(0, T_MAX)
ax_displacement.set_ylim(-1.5, 1.5)
ax_displacement.legend(loc="upper right", fontsize=8, frameon=False)

# Energy
kinetic_line, = ax_energy.plot(
    t_grid,
    kinetic_values,
    color=BLUE,
    linewidth=1.7,
    label="Kinetic",
)

potential_line, = ax_energy.plot(
    t_grid,
    potential_values,
    color=ORANGE,
    linewidth=1.7,
    label="Potential",
)

total_line, = ax_energy.plot(
    t_grid,
    total_values,
    color=GREEN,
    linewidth=2,
    label="Total",
)

energy_point, = ax_energy.plot(
    [],
    [],
    marker="o",
    color=RED,
    markersize=6,
    linestyle="None",
)

ax_energy.set_xlim(0, T_MAX)
ax_energy.set_ylim(0, 5)
ax_energy.legend(loc="upper right", fontsize=8, frameon=False)

# Phase space
phase_line, = ax_phase.plot(
    x_values,
    v_values,
    color=BLUE,
    linewidth=2,
)

phase_point, = ax_phase.plot(
    [],
    [],
    marker="o",
    color=ORANGE,
    markersize=7,
    linestyle="None",
)

ax_phase.axhline(0, color=MUTED, linewidth=0.8, alpha=0.7)
ax_phase.axvline(0, color=MUTED, linewidth=0.8, alpha=0.7)


# ============================================================
# 6. SIDEBAR SLIDERS
# ============================================================

def add_slider_axes(position):
    ax = fig.add_axes(position)
    ax.set_facecolor(PANEL)
    return ax


slider_m = Slider(
    add_slider_axes([0.780, 0.805, 0.180, 0.025]),
    "Mass m (kg)",
    0.2,
    5.0,
    valinit=m_initial,
    color=BLUE,
)

slider_k = Slider(
    add_slider_axes([0.780, 0.745, 0.180, 0.025]),
    "Spring k (N/m)",
    0.5,
    15.0,
    valinit=k_initial,
    color=BLUE,
)

slider_x0 = Slider(
    add_slider_axes([0.780, 0.685, 0.180, 0.025]),
    "Initial x₀ (m)",
    -2.0,
    2.0,
    valinit=x0_initial,
    color=BLUE,
)

slider_v0 = Slider(
    add_slider_axes([0.780, 0.625, 0.180, 0.025]),
    "Initial v₀ (m/s)",
    -3.0,
    3.0,
    valinit=v0_initial,
    color=BLUE,
)

slider_zeta = Slider(
    add_slider_axes([0.780, 0.565, 0.180, 0.025]),
    "Damping ratio ζ",
    0.0,
    2.0,
    valinit=zeta_initial,
    color=BLUE,
)

# Slider value formatting
for slider in (slider_m, slider_k, slider_x0, slider_v0, slider_zeta):
    slider.label.set_color(TEXT)
    slider.label.set_fontsize(8)
    slider.valtext.set_color(BLUE)
    slider.valtext.set_fontsize(8)


# ============================================================
# 7. MOTION MODE SELECTOR
# ============================================================

fig.text(
    0.755,
    0.515,
    "MOTION MODEL",
    fontsize=9,
    fontweight="bold",
    color=TEXT,
)

ax_mode = fig.add_axes([0.755, 0.445, 0.195, 0.065])
ax_mode.set_facecolor(PANEL)

radio_mode = RadioButtons(
    ax_mode,
    ("SHO", "Damped"),
    active=0,
    activecolor=BLUE,
)

for label in radio_mode.labels:
    label.set_color(TEXT)
    label.set_fontsize(9)


# ============================================================
# 8. PLAY, PAUSE, RESET, AND SPEED CONTROLS
# ============================================================

ax_play = fig.add_axes([0.755, 0.375, 0.085, 0.045])
ax_reset = fig.add_axes([0.855, 0.375, 0.095, 0.045])

button_play = Button(
    ax_play,
    "Pause",
    color="#e7f0fa",
    hovercolor="#d3e5f7",
)

button_reset = Button(
    ax_reset,
    "Reset",
    color="#fce9e7",
    hovercolor="#f7d6d2",
)

for button in (button_play, button_reset):
    button.label.set_color(TEXT)
    button.label.set_fontsize(9)

fig.text(
    0.755,
    0.335,
    "ANIMATION SPEED",
    fontsize=9,
    fontweight="bold",
    color=TEXT,
)

slider_speed = Slider(
    add_slider_axes([0.755, 0.295, 0.195, 0.025]),
    "Speed",
    0.25,
    2.5,
    valinit=1.0,
    color=GREEN,
)

slider_speed.label.set_color(TEXT)
slider_speed.label.set_fontsize(8)
slider_speed.valtext.set_color(GREEN)
slider_speed.valtext.set_fontsize(8)


# ============================================================
# 9. LIVE NUMERICAL READOUT
# ============================================================

fig.text(
    0.755,
    0.255,
    "LIVE READOUT",
    fontsize=10,
    fontweight="bold",
    color=TEXT,
)

# IMPORTANT FIX:
# The readout text belongs to a real Axes, not directly to the Figure.
# This prevents the '_get_view' error when FuncAnimation uses blitting.
ax_readout = fig.add_axes([0.755, 0.075, 0.205, 0.16])
ax_readout.set_facecolor(PANEL)
ax_readout.set_axis_off()

readout = ax_readout.text(
    0.0,
    1.0,
    "",
    transform=ax_readout.transAxes,
    fontsize=9,
    family="monospace",
    color=TEXT,
    linespacing=1.7,
    va="top",
)


# ============================================================
# 10. UPDATE THE MODEL WHEN PARAMETERS CHANGE
# ============================================================

def update_model(_=None):
    """Recalculate all curves after a parameter or mode change."""
    m, k, x0, v0, zeta = current_parameters()

    x, v = oscillator(t_grid, m, k, x0, v0, zeta)
    kinetic, potential, total = calculate_energy(m, k, x, v)

    # Update displacement curve
    displacement_line.set_data(t_grid, x)

    # Update energy curves
    kinetic_line.set_data(t_grid, kinetic)
    potential_line.set_data(t_grid, potential)
    total_line.set_data(t_grid, total)

    # Update phase-space curve
    phase_line.set_data(x, v)

    # Give the plots appropriate limits
    max_x = max(1.0, np.max(np.abs(x)) * 1.15)
    max_v = max(1.0, np.max(np.abs(v)) * 1.15)
    max_energy = max(1.0, np.max(total) * 1.15)

    ax_displacement.set_ylim(-max_x, max_x)
    ax_phase.set_xlim(-max_x, max_x)
    ax_phase.set_ylim(-max_v, max_v)
    ax_energy.set_ylim(0, max_energy)

    # Reset current markers at the current animation time
    x_now, v_now = oscillator(t_current, m, k, x0, v0, zeta)
    kinetic_now, potential_now, total_now = calculate_energy(
        m, k, x_now, v_now
    )

    displacement_point.set_data([t_current], [float(x_now)])
    energy_point.set_data([t_current], [float(total_now)])
    phase_point.set_data([float(x_now)], [float(v_now)])

    # Update mass and spring position
    mass_x = equilibrium_x + 0.12 * float(x_now)
    mass_x = np.clip(mass_x, 0.43, 0.88)

    sx, sy = make_spring(0.15, mass_x - 0.045)
    spring_line.set_data(sx, sy)
    mass_dot.set_data([mass_x], [0.0])

    # Update live readout
    readout.set_text(
        f"t       : {t_current:6.2f} s\n"
        f"x(t)    : {float(x_now):6.3f} m\n"
        f"v(t)    : {float(v_now):6.3f} m/s\n"
        f"Eₖ      : {float(kinetic_now):6.3f} J\n"
        f"Eₚ      : {float(potential_now):6.3f} J\n"
        f"E total : {float(total_now):6.3f} J"
    )

    fig.canvas.draw_idle()


# ============================================================
# 11. ANIMATION UPDATE FUNCTION
# ============================================================

def update_anim(_frame):
    global t_current

    # Advance simulation time.
    # Speed changes the simulated time increment per frame.
    dt = 0.025 * speed_factor
    t_current += dt

    # Loop back to the start after the plotted time interval.
    if t_current > T_MAX:
        t_current = 0.0

    m, k, x0, v0, zeta = current_parameters()

    x_now, v_now = oscillator(t_current, m, k, x0, v0, zeta)

    kinetic_now, potential_now, total_now = calculate_energy(
        m, k, x_now, v_now
    )

    # Update the mass and spring
    mass_x = equilibrium_x + 0.12 * float(x_now)
    mass_x = np.clip(mass_x, 0.43, 0.88)

    sx, sy = make_spring(0.15, mass_x - 0.045)
    spring_line.set_data(sx, sy)
    mass_dot.set_data([mass_x], [0.0])

    # Update the time display
    time_text.set_text(
        f"t = {t_current:.2f} s   |   Model: {mode}"
    )

    # Update the markers on the plots
    displacement_point.set_data([t_current], [float(x_now)])
    energy_point.set_data([t_current], [float(total_now)])
    phase_point.set_data([float(x_now)], [float(v_now)])

    # Update the numerical readout
    readout.set_text(
        f"t       : {t_current:6.2f} s\n"
        f"x(t)    : {float(x_now):6.3f} m\n"
        f"v(t)    : {float(v_now):6.3f} m/s\n"
        f"Eₖ      : {float(kinetic_now):6.3f} J\n"
        f"Eₚ      : {float(potential_now):6.3f} J\n"
        f"E total : {float(total_now):6.3f} J"
    )

    # Return only artists attached to axes.
    # This is important when blit=True.
    return (
        spring_line,
        mass_dot,
        time_text,
        displacement_point,
        energy_point,
        phase_point,
        readout,
    )


# ============================================================
# 12. CONTROL CALLBACKS
# ============================================================

def on_parameter_change(_value):
    update_model()


def on_mode_change(label):
    global mode

    mode = "SHO" if label == "SHO" else "DHO"

    if mode == "SHO":
        slider_zeta.label.set_color(MUTED)
        slider_zeta.valtext.set_color(MUTED)
    else:
        slider_zeta.label.set_color(TEXT)
        slider_zeta.valtext.set_color(BLUE)

    update_model()


def on_speed_change(value):
    global speed_factor

    speed_factor = float(value)

    # Keep the animation timer at a comfortable refresh rate.
    # The simulation-time increment is adjusted in update_anim().
    animation.event_source.interval = 25


def on_play_pause(_event):
    global playing

    if playing:
        animation.event_source.stop()
        button_play.label.set_text("Play")
        playing = False
    else:
        animation.event_source.start()
        button_play.label.set_text("Pause")
        playing = True

    fig.canvas.draw_idle()


def on_reset(_event):
    global t_current

    t_current = 0.0

    # Restore initial parameter values.
    slider_m.reset()
    slider_k.reset()
    slider_x0.reset()
    slider_v0.reset()
    slider_zeta.reset()
    slider_speed.reset()

    # Return to SHO mode.
    radio_mode.set_active(0)

    update_model()

    # Restart the animation if it was paused.
    global playing
    if not playing:
        animation.event_source.start()
        button_play.label.set_text("Pause")
        playing = True

    fig.canvas.draw_idle()


# Connect parameter sliders
slider_m.on_changed(on_parameter_change)
slider_k.on_changed(on_parameter_change)
slider_x0.on_changed(on_parameter_change)
slider_v0.on_changed(on_parameter_change)
slider_zeta.on_changed(on_parameter_change)
slider_speed.on_changed(on_speed_change)

# Connect buttons and radio selector
radio_mode.on_clicked(on_mode_change)
button_play.on_clicked(on_play_pause)
button_reset.on_clicked(on_reset)


# ============================================================
# 13. INITIALIZE AND START ANIMATION
# ============================================================

update_model()

animation = FuncAnimation(
    fig,
    update_anim,
    interval=25,
    blit=True,
    cache_frame_data=False,
)

# Keep a reference to the animation object (important).
# Display the figure.
plt.show()