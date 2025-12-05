# Part 1 – Lorenz System Modeling and Visualization
This README provides an overview of the purpose, structure, mathematical foundation, and usage of the Part 1 Lorenz Attractor project.

---

## 📌 Project Overview
This project implements a fully interactive visualization of the **Lorenz Attractor**, a classic model used to demonstrate chaotic behavior and sensitivity to initial conditions.  
The program numerically solves the Lorenz system of differential equations and displays a dynamic 3D visualization that updates in real time as the user changes system parameters.

This model helps demonstrate how small changes in initial conditions or configuration values can dramatically influence long-term system behavior — an important concept in system modeling, performance analysis, and chaos theory.

---

## 📂 Files Included
- `Proj7Pt1.py` — Main Python program implementing:
  - Lorenz differential equations  
  - RK4 numerical solver  
  - Interactive sliders for σ, ρ, and β  
  - Real‑time 3D visualization with Matplotlib  
- `README.md` — This documentation file  
- (Optional) Screenshots or images inserted by the user

---

## ⚙️ Features
### **Interactive Parameter Sliders**
The program allows real-time adjustment of:
- **σ (sigma)** – Controls divergence in the x–y plane  
- **ρ (rho)** – Controls onset of chaotic behavior  
- **β (beta)** – Controls vertical contraction of the attractor  

### **Dynamic Lorenz Attractor Visualization**
- Computed using a **4th‑order Runge–Kutta (RK4)** numerical integrator  
- Updates automatically as sliders move  
- Includes an animated trailing effect to emphasize motion  

### **Reset Button**
Resets all parameters back to default values:
```
σ = 10  
ρ = 28  
β = 8/3
```

---

## 🧠 Mathematical Model
The Lorenz system is given by:

```
dx/dt = σ(y − x)
dy/dt = x(ρ − z) − y
dz/dt = xy − βz
```

These equations exhibit **chaos**, meaning the system is extremely sensitive to initial conditions.  
A tiny difference in any parameter can lead to dramatically different long‑term behavior — this is the basis of the famous **"butterfly effect."**

The project numerically solves the system using the **RK4 integrator**, producing the trajectory sampled at small time intervals.

---

## 🚀 How to Run the Program

### **1. Install dependencies**
Make sure Python is installed, then run:

```bash
pip install numpy matplotlib
```

### **2. Run the script**
```bash
python Proj7Pt1.py
```

A visualization window will open with:
- A 3D Lorenz attractor
- Sliders for σ, ρ, β
- A Reset button  
- Real‑time animation

---

## 🧩 How It Works (Step‑By‑Step)
1. Initialize system parameters and time settings  
2. Define Lorenz ODE function  
3. Implement RK4 numerical solver  
4. Compute initial trajectory  
5. Render 3D attractor using Matplotlib  
6. Create slider widgets for parameter control  
7. Listen for slider events → recompute trajectory  
8. Update 3D plot dynamically  
9. Animate system with a moving trail effect  

---

## 📚 References
Lorenz, E. N. (1963). *Deterministic nonperiodic flow*. Journal of the Atmospheric Sciences, 20(2), 130–141.

Strogatz, S. H. (2015). *Nonlinear dynamics and chaos* (2nd ed.). Westview Press.

Burden, R. L., & Faires, J. D. (2011). *Numerical analysis* (9th ed.). Brooks/Cole.

Harris, C. R., et al. (2020). *Array programming with NumPy*. Nature, 585, 357–362.

Hunter, J. D. (2007). *Matplotlib: A 2D graphics environment*. Computing in Science & Engineering, 9(3), 90–95.


## ✅ End of README
