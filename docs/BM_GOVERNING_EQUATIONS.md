# Beam-Membrane Model — Governing Equations

Extracted directly from Sean's MATLAB at `VocalFoldRegression/Beam+Membrane_ForSean/`. File:line citations throughout.

The model is a 6-stage pipeline. Stages 2, 3, 5 are differentiable; stage 1 calls an external library (`MuscleControlModel`); stage 4 is embedded inside stage 3; stage 6 uses non-differentiable post-hoc statistics.

```
[a_LCA, a_IA, a_PCA, a_CT, a_TA]
       │
       ▼  Stage 1: Posturing (Posturing_Simulation.m)
[eps0, Theta_G]
       │  (longitudinal strain, glottic angle)
       │
       ▼  Stage 2: Constitutive algebra (Membrane_Beam_Parameters.m, TA/LIG/MUC.m)
[sigma_MUC, F_beam, Mu_c_beam, M0_beam, d_MUC]
       │  (cover tension, beam axial tension, flexural rigidity, end moment, cover depth)
       │
       │       Stage 4: Aerodynamic pressure (embedded)
       │            ┌─────────────────────────┐
       ▼            ▼                         │
       Stage 3: Coupled PDE in (x, y, t)  ───>│  p(x, y, t) = -(p_aero + p_contact)
       (Membrane_Beam_Solver.m)               │
       Outputs: w(x,y,t), w_b(x,t), Am(t)     │
       │            ▲                         │
       │            └─────────────────────────┘
       ▼  Stage 5: WRA acoustic propagation (WRA_Solver.m)
       p_out(t)
       │
       ▼  Stage 6: Post-hoc measurements (Measures_Estimates.m)
       [F0, SPL]
```

---

## Stage 1 — Posturing (quasi-static, external library)

`Posturing_Simulation.m:1-61`. Uses `MuscleControlModel` (parameter set `Alzamendi2020`).

**Inputs:** `ActVec = [a_LCA, a_IA, a_PCA, a_CT, a_TA]` ∈ [0, 1]^5

**Outputs:**
- `eps0` — longitudinal strain of the vocal fold (dimensionless)
- `Theta_G` — glottic angle (radians); halved before being passed to stage 3 (`Randomly_Generating_Data_Membrane_Beam_Model.m:110`)

**Treatment for PINN:** This is a black box. For the PINN, we'll either (a) call MATLAB to obtain `(eps0, Theta_G)` per training sample, or (b) treat `(eps0, Theta_G)` as additional features alongside `(a_CT, a_TA, PS)` and side-step the posturing stage. (b) is simpler for v1.

---

## Stage 2 — Constitutive algebra (closed-form, differentiable)

### Tissue-layer stresses

Each layer has an exponential stress-strain law with optional muscle activation contribution:

`TA.m:1-11` — thyroarytenoid (with active component):
$$\sigma_{TA}(\varepsilon, a_{TA}) = \mathrm{sign}(\varepsilon) \cdot A_{TA} \cdot \left(e^{B_{TA} |\varepsilon|} - 1\right) + a_{TA} \cdot \Sigma_m$$
$$E_{TA}(\varepsilon) = A_{TA} \cdot B_{TA} \cdot e^{B_{TA} |\varepsilon|}$$

with $A_{TA} = 10^3$ Pa, $B_{TA} = 8$, $\Sigma_m = 1.05 \times 10^5$ Pa.

`LIG.m:1-11` — vocal ligament (passive):
$$\sigma_{LIG}(\varepsilon) = \mathrm{sign}(\varepsilon) \cdot A_{LIG} \cdot \left(e^{B_{LIG} |\varepsilon|} - 1\right), \quad E_{LIG} = A_{LIG} B_{LIG} e^{B_{LIG} |\varepsilon|}$$

with $A_{LIG} = 2 \times 10^3$ Pa, $B_{LIG} = 10$.

`MUC.m:1-11` — mucosa / cover (passive):
$$\sigma_{MUC}(\varepsilon) = \mathrm{sign}(\varepsilon) \cdot A_{MUC} \cdot \left(e^{B_{MUC} |\varepsilon|} - 1\right), \quad E_{MUC} = A_{MUC} B_{MUC} e^{B_{MUC} |\varepsilon|}$$

with $A_{MUC} = 1.5 \times 10^3$ Pa, $B_{MUC} = 7$.

### Beam parameters

`Membrane_Beam_Parameters.m:1-49`. Given strain `eps0`, activation `a_TA`, and geometry `(b, A_MUC, A_LIG, A_TA)`:

- **Layer depths:** $d_X = A_X / b$ for $X \in \{MUC, LIG, TA\}$
- **Centerline positions** (from base of TA): $r_c = d_{TA} + 0.5 d_{LIG}$, $r_{MUC} = d_{TA} + d_{LIG} + 0.5 d_{MUC}$, etc.
- **Axial forces:** $F_X = A_X \sigma_X$
- **Beam axial tension:** $F_{\text{beam}} = F_{LIG} + F_{TA}$
- **End moment:** $M_0^{\text{beam}} = (r_c - r_{LIG}) F_{LIG} + (r_c - r_{TA}) F_{TA}$
- **Effective flexural rigidity:**
$$\Theta = E_{LIG} I_{LIG} + E_{TA} I_{TA} + (r_c - r_{LIG}) A_{LIG} E_{LIG} \alpha_{LIG} + (r_c - r_{TA}) A_{TA} E_{TA} \alpha_{TA}$$
$$\mu_c^{\text{beam}} = \Theta (1 + \varepsilon_0)$$

with shift parameters
$$\alpha_{LIG} = -\frac{d_{TA} + d_{LIG}}{2 \frac{E_{LIG} A_{LIG}}{E_{TA} A_{TA}} + 1}, \quad \alpha_{TA} = \frac{d_{TA} + d_{LIG}}{2 \frac{E_{TA} A_{TA}}{E_{LIG} A_{LIG}} + 1}.$$

**Validity:** the simulation only runs if $F_{\text{beam}} > 0$ AND $\sigma_{MUC} > 0$ — beam in compression or compressed cover ⇒ NaN F0/SPL row. (`Randomly_Generating_Data_Membrane_Beam_Model.m:124`.)

**Stretched VF length:** $L = (1 + \varepsilon_0) L_0$, with $L_0 = 1.5 \times 10^{-2}$ m.

---

## Stage 3 — Coupled PDE system (the heart)

Two coupled PDEs on $(x, y) \in [0, L] \times [-h/2, h/2]$ with $h = b = 5 \times 10^{-3}$ m. Solved by central-difference time stepping in `Membrane_Beam_Solver.m:75-214`.

### Membrane (cover) — 2D second-order wave with body-cover coupling

`Membrane_Beam_Solver.m:126-130`. Let $w(x, y, t)$ denote membrane deflection.

$$\boxed{\rho_m \frac{\partial^2 w}{\partial t^2} = T_x \frac{\partial^2 w}{\partial x^2} + K_s \frac{\partial^2 w}{\partial y^2} + K_c (w_b - w) + C_c \left(\frac{\partial w_b}{\partial t} - \frac{\partial w}{\partial t}\right) + p(x, y, t)}$$

**Coefficients** (`Membrane_Beam_Solver.m:14-23`):
- $\rho_m = \rho_{MUC} \cdot d_{MUC}$ — areal mass density [kg/m²]
- $T_x = \sigma_{MUC} \cdot d_{MUC}$ — longitudinal cover tension per unit width [N/m]
- $K_s = S_m = 2$ — transverse stiffness [N/m²] (constant; `Stiffness_Damping_Membrane_Beam.m:11`)
- $K_c = 10^5$ — body-cover coupling stiffness [N/m³]
- $C_c = 400$ — body-cover coupling damping [N·s/m²]
- $p$ — aerodynamic + contact pressure (Stage 4 below)
- $w_b(x, t)$ — beam deflection (broadcast in $y$ via `wb_mat = wb · ones(1, Ny)`)

### Beam (body) — Euler-Bernoulli with axial tension and body-cover coupling

`Membrane_Beam_Solver.m:178-184`. Let $w_b(x, t)$ denote beam deflection.

$$\boxed{\rho_b \frac{\partial^2 w_b}{\partial t^2} + C_f \frac{\partial w_b}{\partial t} + K_f w_b + \mu_c^{\text{beam}} \frac{\partial^4 w_b}{\partial x^4} - T_b \frac{\partial^2 w_b}{\partial x^2} = -\int_{-h/2}^{h/2} \left[K_c(w_b - w) + C_c\left(\frac{\partial w_b}{\partial t} - \frac{\partial w}{\partial t}\right)\right] dy}$$

**Coefficients:**
- $\rho_b = \rho_{TA} A_{TA} + \rho_{LIG} A_{LIG}$ — mass per unit length [kg/m]
- $K_f = 5 \times 10^4$ — body stiffness [N/m²]
- $C_f = 100$ — body damping [N·s/m²]
- $\mu_c^{\text{beam}}$ — flexural rigidity (from Stage 2) [N·m²]
- $T_b = F_{\text{beam}}$ — beam axial tension (from Stage 2) [N]

The right-hand side integrates the cover-coupling reaction over the cover thickness $y \in [-h/2, h/2]$.

### Boundary conditions

**Membrane** (`Membrane_Beam_Solver.m:81-84, 193-196`):
- $w(0, y, t) = w(L, y, t) = 0$ — clamped at $x$ ends
- $\partial w / \partial y = 0$ at $y = \pm h/2$ — Neumann (free) at $y$ ends, implemented via mirror-image ghost nodes

**Beam** (`Membrane_Beam_Solver.m:87-88, 145-147, 199-200`):
- $w_b(0, t) = w_b(L, t) = 0$ — clamped displacement at $x$ ends
- Rotational spring at left end:
$$\left.\frac{\partial^2 w_b}{\partial x^2}\right|_{x=0} = -\frac{M_0^{\text{beam}} + K_r \left(\theta_G - \frac{\partial w_b}{\partial x}\big|_{x=0} - \Theta_0\right)}{\mu_c^{\text{beam}}}$$
- Free moment at right end:
$$\left.\frac{\partial^2 w_b}{\partial x^2}\right|_{x=L} = -\frac{M_0^{\text{beam}}}{\mu_c^{\text{beam}}}$$

with $K_r = 150$ N·m, $\Theta_0 = 0.2540$ rad (`Stiffness_Damping_Membrane_Beam.m:7, 34`).

### Initial conditions

(`Randomly_Generating_Data_Membrane_Beam_Model.m:77-80`)
- $w(x, y, 0) = 0$, $w_b(x, 0) = 0$
- $\partial w / \partial t (x, y, 0) = 10^{-6} \cdot \mathcal{U}(0, 1)$ — small noise to seed mucosal wave propagation
- $\partial w_b / \partial t (x, 0) = 0$

---

## Stage 4 — Aerodynamic pressure (Bernoulli triangular)

`Membrane_Beam_Solver.m:228-282`. Computed at every time step inside the dynamic loop. Inputs: current $w(x, y)$, subglottal pressure $P_s$, separation ratio $sr = 1.2$, glottal half-angle $\theta_G$.

### Local glottal gap

For each $(i, j)$ on the mesh:
$$g(x_i, y_j) = \max\left(x_i \tan(\theta_G) - w(x_i, y_j), \; 0\right)$$

The triangular gap shape is a fixed geometric assumption — the resting glottis is a wedge with half-angle $\theta_G$, narrowed by membrane deflection.

### Per-$x$ minimum and separation point

For each $x_i$:
- $g_{\min}(x_i) = \min_j g(x_i, y_j)$, with $j_{\min}(x_i) = \arg\min_j g(x_i, y_j)$
- Separation gap: $g_{\text{sep}}(x_i) = sr \cdot g_{\min}(x_i)$
- Separation index $j_{\text{sep}}(x_i)$ = first $j$ where $g(x_i, y_j)$ falls below $g_{\text{sep}}(x_i)$ (sweeping outward from $j_{\min}$)

### Minimum glottal area

$$A_m(t) = 2 \int_0^L g_{\min}(x, t) \, dx$$

(factor of 2 for left+right symmetry; `Membrane_Beam_Solver.m:259`).

### Pressure components

**Aerodynamic** (Bernoulli, with separation):
$$p_{\text{aero}}(x, y, t) = \begin{cases} P_s \left[1 - \left(\frac{g_{\min}(x)}{g(x, y)}\right)^2\right] & \text{if } j \le j_{\text{sep}}(x) \text{ and } g_{\min}(x) > 0 \\ 0 & \text{if } j > j_{\text{sep}}(x) \\ P_s & \text{if } j \le j_{\text{sep}}(x) \text{ and } g_{\min}(x) = 0 \end{cases}$$

**Contact** (penalty for tissue interpenetration):
$$p_{\text{contact}}(x, y, t) = K_{\text{contact}} \cdot \max\left(w(x, y, t) - x \tan(\theta_G), \; 0\right)$$

with $K_{\text{contact}} = 10^6$ N/m².

**Net load on membrane** (sign: downward = negative):
$$p(x, y, t) = -(p_{\text{aero}} + p_{\text{contact}})$$

This pressure feeds back into the membrane PDE in Stage 3.

---

## Stage 5 — WRA acoustic propagation

`WRA_Solver.m`. Wave Reflection Analog: the vocal tract is a sequence of cylindrical tubes. Forward and backward 1D pressure waves propagate with reflection coefficients at junctions.

### Source

Volume velocity at glottis:
$$q(t) = \sqrt{\frac{2 P_s}{\rho_{\text{air}}}} \cdot A_{\text{sep}}(t), \quad A_{\text{sep}}(t) = sr \cdot A_m(t)$$

with $\rho_{\text{air}} = 1.14$ kg/m³ and sound speed $c = 350$ m/s.

### Tract geometry

Sub-glottal: $A_{\text{sub}} = 10^{-4} \cdot [1, 1]$ m² (two tubes; `WRA_Solver.m:41`).

Supra-glottal: vowel-dependent area arrays from `DifferentVowels.m`. The script default uses `A_supi` (vowel /i/, 70 tubes). Each tube has length $\delta_{\text{sup}} = (10^{-2} \cdot 0.25) / \text{factor}$ with `factor=7` (`Randomly_Generating_Data_Membrane_Beam_Model.m:19`), so $\delta_{\text{sup}} \approx 3.57 \times 10^{-4}$ m. Tube count is then expanded by `expand_vector` factor of 7 → 490 tubes.

### Junction reflection equations

For a junction between tubes $i$ and $i+1$ with areas $A_i, A_{i+1}$ and attenuations $\alpha_i, \alpha_{i+1}$:

$$O_b^{(J,I)} = \alpha_i \frac{A_i - A_{i+1}}{A_i + A_{i+1}} O_f^{(J-1, I-1)} + \alpha_{i+1} \frac{2 A_{i+1}}{A_i + A_{i+1}} O_b^{(J-1, I+1)}$$

$$O_f^{(J,I)} = \alpha_i \frac{2 A_i}{A_i + A_{i+1}} O_f^{(J-1, I-1)} + \alpha_{i+1} \frac{A_{i+1} - A_i}{A_i + A_{i+1}} O_b^{(J-1, I+1)}$$

with $\alpha_i = 1 - 3.8 \times 10^{-3} A_i^{-0.5} \delta_{\text{sup}}$ — viscothermal attenuation per tube.

### Glottal source injection

At the glottis junction (between sub and supra):
$$O_b^{(J,I)} = \alpha_i O_f^{(J-1,I-1)} - \frac{\rho c}{A_i} q(j)$$
$$O_f^{(J,I)} = \alpha_{i+1} O_b^{(J-1,I+1)} + \frac{\rho c}{A_{i+1}} q(j)$$

### Mouth radiation

Radiation impedance via Levine-Schwinger:
$$Z = \frac{\rho c}{A_{\text{end}}}, \quad R_{\text{rad}} = \frac{128 Z}{9 \pi^2}, \quad L_{\text{rad}} = \frac{8 a Z}{3 \pi c}, \quad a = \sqrt{\frac{A_{\text{end}}}{\pi}}$$

with full reflection equation in `WRA_Solver.m:118-126`. Output: $p_{\text{out}}(t)$ at the mouth.

---

## Stage 6 — Post-hoc measurements (non-differentiable)

`Measures_Estimates.m`. Operates on steady-state window $t \in [t_1, t_2] = [0.5, 1.0]$ s.

**F0 estimation** (`:27-28`): MATLAB `pitch()` function on $A_m(t)$ in steady state, then averaged.

**SPL** (`:107-110`):
$$\text{SPL} = 20 \log_{10}\left(\frac{\text{rms}(p_{\text{out}}(t \in [t_1, t_2]))}{p_0}\right) \quad \text{dB SPL}$$

with $p_0 = 20 \times 10^{-6}$ Pa (reference pressure).

---

## Constants summary

From `Stiffness_Damping_Membrane_Beam.m`:

| Symbol | Value | Units | Purpose |
|---|---|---|---|
| $K_c$ | $10^5$ | N/m³ | body-cover coupling stiffness |
| $C_c$ | $400$ | N·s/m² | body-cover coupling damping |
| $K_{\text{contact}}$ | $10^6$ | N/m² | contact penalty stiffness |
| $K_r$ | $150$ | N·m | rotational spring at left end |
| $K_f$ | $5 \times 10^4$ | N/m² | beam (body) stiffness |
| $C_f$ | $100$ | N·s/m² | beam (body) damping |
| $S_m$ | $2$ | N/m² | membrane transverse stiffness $K_s$ |
| $\rho_{MUC}$ | $1000$ | kg/m³ | mucosa density |
| $\rho_{TA}$ | $1050$ | kg/m³ | TA density |
| $\rho_{LIG}$ | $1030$ | kg/m³ | ligament density |
| $L_0$ | $1.5 \times 10^{-2}$ | m | resting VF length |
| $b$ | $5 \times 10^{-3}$ | m | thickness |
| $A_{MUC}$ | $5 \times 10^{-6}$ | m² | mucosa area |
| $A_{LIG}$ | $6.1 \times 10^{-6}$ | m² | ligament area |
| $A_{TA}$ | $40.9 \times 10^{-6}$ | m² | TA area |
| $\Theta_0$ | $0.2540$ | rad | resting VF angle |

From `Randomly_Generating_Data_Membrane_Beam_Model.m`:

| Symbol | Value | Units | Purpose |
|---|---|---|---|
| $T$ | $1.0$ | s | simulation duration |
| $sr$ | $1.2$ | — | separation ratio |
| $N_x$ | $15$ | — | grid points along $x$ |
| $N_y$ | $6$ | — | grid points along $y$ |
| $\text{factor}$ | $7$ | — | WRA refinement |
| $dt$ | $\delta / (c \cdot \text{factor})$ ≈ $1.02 \times 10^{-6}$ | s | time step (CFL-driven) |
| $f_s$ | $\lfloor 1/dt \rfloor$ ≈ 980 kHz | Hz | sample rate |
| $[t_1, t_2]$ | $[0.5, 1.0]$ | s | steady-state window |

Aerodynamic input ranges:
- $P_s \in [300, 1000]$ Pa
- Activations $\in [0, 1]^5$ (only $a_{CT}, a_{TA}$ vary; others held at midpoints during data gen — see `Randomly_Generating_Data_Membrane_Beam_Model.m:41-58`).

---

## What this means for the PINN

A "real" PINN over Stage 3 would predict the fields $w(x, y, t)$ and $w_b(x, t)$ as neural networks, with PDE residuals computed at collocation points via autograd. This is a **two-coupled-PDE problem in 2+1 dimensions** (membrane) and **one PDE in 1+1 dimensions** (beam), with non-trivial boundary conditions including a rotational-spring BC that depends on $\partial w_b / \partial x$ at the left end.

Stages 1, 5 are external (MuscleControlModel, WRA): we either call MATLAB or build a PyTorch surrogate.

Stage 6 (F0/SPL extraction) is non-differentiable as written. For training, we'll either approximate with a smoothed pitch detector + RMS, or evaluate it post-hoc and only train Stage 3.

The full plan is in `docs/superpowers/specs/2026-05-06-real-pinn-design.md`.
