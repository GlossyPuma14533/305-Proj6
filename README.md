# 📘 **README – CST-305 Benchmark Project 6: Numeric Computations with Taylor Polynomials**

## **Project Overview**
This project analyzes multiple differential equations using **Taylor series**, **recurrence relations**, and **numerical ODE methods** (RK4). It is divided into three major parts:

- **Part 1:** Taylor expansion of a second-order ODE, approximation of \( y(3.5) \), convergence analysis, and visualization.  
- **Part 2:** Ordinary point classification, recurrence relation derivation, general series solution, and comparison with a numerical RK4 solution.  
- **Part 3:** A holistic computer-performance model using a nonlinear differential equation, Taylor expansion, and numerical simulation of system behavior.

All code executes correctly and includes detailed graphs, error analysis, and printed values.

---

## 📂 **Repository Contents**

### **🔹 Part A Files**
- `Part A.docx` – Full derivation, calculations, and explanations.
- `part_a_code.py` – Python implementation:
  - RK4 numerical solver  
  - Taylor approximation up to 4th order  
  - Graphs of Taylor vs RK4  
  - Convergence visualization  
  - Evaluation of \( y(3.5) \)

---

### **🔹 Part B Files**
- `Part B.docx` – Second-order Taylor polynomial derivation around \( x = 3 \).
- `part_b_code.py` – Python implementation:
  - RK4 numerical solver  
  - Second-order Taylor polynomial  
  - Error graph and convergence discussion  

---

### **🔹 Part 2 Files**
- `Part2.docx` – Ordinary point analysis, recurrence relation, symbolic solution.
- `Part2.py` – Python implementation:
  - Recurrence relation for coefficients  
  - Taylor approximation up to order 10  
  - RK4 solver for differential equation  
  - Comparison graphs and error analysis  

---

### **🔹 Part 3 Files**
- `Part3_Merged.docx` – Full written documentation including:
  - Holistic performance model  
  - Differential equation  
  - Taylor expansion up to 5th order  
  - Ethical, professional, and legal considerations
- `Part3.py` – Python code for:
  - Defining performance model ODE  
  - RK4 numerical simulation  
  - Taylor-series convergence visualization  
  - Error graph  
  - Printed performance values at selected times  

---

## 🚀 **How to Run the Programs**

### **Prerequisites**
Ensure you have the following installed:

```bash
python 3.9+
numpy
matplotlib
```

Install missing packages via:

```bash
pip install numpy matplotlib
```

---

## ▶️ **Running Each Part**

### **Part A**
```bash
python part_a_code.py
```

### **Part B**
```bash
python part_b_code.py
```

### **Part 2**
```bash
python Part2.py
```

### **Part 3**
```bash
python Part3.py
```

Each script will automatically:
- Run the numerical RK4 solver  
- Compute Taylor approximations  
- Generate comparison graphs  
- Display convergence behavior  
- Print selected function values where required  

---

## 📈 **Features and Implementation Notes**

### **Numerical Method**
The **fourth-order Runge–Kutta (RK4)** method is used throughout:
- Stable for stiff and nonlinear ODEs  
- Excellent local accuracy  
- Ideal for comparing with Taylor approximations  

### **Taylor Expansions**
- Higher-order derivatives are computed analytically or by recurrence.
- Evaluations use Horner-optimized polynomial evaluation for stability.
- Convergence behavior is visualized at increasing orders.

### **Error Analysis**
Each part includes:
- RK4 “ground truth”
- Visual error plots:  
  - Absolute error vs x (or t)  
  - Taylor order comparison  
- Discussion of truncation and radius of convergence  

---

## 🖼️ **Screenshots**
Screenshots of:
- RK4 numerical solution  
- Taylor series overlays  
- Error plots  
- Convergence behavior  

are included in the Word documentation files for Parts 1–3.

---

## 📚 **References**
- **CST-305 Course Materials** (GCU)  
- Class lectures on Taylor polynomials, power series solutions, and RK4  
- Grand Canyon University. *Benchmark Project 6: Numeric Computations with Taylor Polynomials* (Assignment brief)  
- Research sources cited in Part 3 documentation  

---

## 👤 **Team Responsibilities**
- **Christoffer Hansen**  
  - Mathematical derivations for all parts  
  - Full Python implementation (Parts 1–3)  
  - Taylor series, recurrence relations, and RK4 computations  
  - Visualizations and analysis  
  - Documentation assembly, formatting, and editing  

---

## ✅ **Status: Complete & Ready for Submission**
All required deliverables (documentation, code, outputs, analysis) meet the expectations of the Benchmark Project 6 rubric.
