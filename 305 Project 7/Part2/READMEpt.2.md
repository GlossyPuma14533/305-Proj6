# README – Part 2: Butterfly Effect of a Small Code Error

## Overview
This project demonstrates how a tiny mistake in a software system can propagate and create large downstream effects—mirroring the butterfly effect in chaotic systems. The program simulates a three-stage data pipeline:

1. Load raw data  
2. Normalize the data (correct vs. buggy)  
3. Analyze downstream metrics  

A small code error in the buggy version (accidentally using `std**2` instead of `std`) produces drastic changes in variance and risk scores, despite the input data being identical.

---

## How to Run the Program

### Requirements
- Python 3.8+
- NumPy  
- Matplotlib  

Install dependencies:

```bash
pip install numpy matplotlib
```

### Run the script
```bash
python Proj7Pt2.py
```

---

## Pipeline Description

### Stage 0 — Load Raw Data
Loads 500 samples from a normal distribution (μ = 50, σ = 10).

### Stage 1 — Normalization
**Correct:** `(data - mean) / std`  
**Buggy:** `(data - mean) / (std ** 2)` ← small bug with big impact

### Stage 2 — Statistical Analysis
Computes:
- Mean  
- Variance  
- Logistic risk score  

---

## Visualization

The script produces a 2×2 figure:
- Histogram (Correct normalization)
- Histogram (Buggy normalization)
- Variance comparison bar chart
- Risk score comparison bar chart

---

## Code Butterfly Severity (CBS)

CBS compares 3 values:
1. Normalized mean  
2. Normalized variance  
3. Risk score  

It outputs:
- Absolute severity  
- Relative severity  

This represents how far the buggy system diverges from the correct system.

---

## Example Terminal Output

```
=== NUMERIC COMPARISON OF OUTPUTS ===
Correct normalized mean:   0.000000
Buggy normalized mean:     0.000000

Correct normalized var:    1.000000
Buggy normalized var:      0.010000

Correct risk score:        0.731059
Buggy risk score:          0.502500
```

---

## References
- Proj7Pt2.py (source code)
- NumPy documentation
- Matplotlib documentation
- CST-305 course materials

