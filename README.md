# Causal Modeling

This repository provides a framework for simulating and visualizing causal spreading processes across networks for Complex Contagions.  
It is organized into two main components:
- **`Calculation.py`**: Simulates contagion processes and generates processed data files.
- **`Visualization.ipynb`**: Loads and visualizes the generated results.

## System Requirements

### Software Dependencies

**For running `Calculation.py`:**
- Python 3.8 or higher
- numpy
- pandas
- matplotlib
- statsmodels
- scipy
- networkx
- tqdm
- joblib

**For running `Visualization.ipynb`:**
- matplotlib
- pandas
- seaborn
- joblib

### Operating System
- Tested on macOS 15.4.

### Hardware Requirements
- Standard desktop or laptop computer.
- Adequate RAM is recommended to handle the simulation outputs.

## Installation Guide

1. Clone the repository:
   ```bash
   git clone https://github.com/ftschofenig/causal-modeling.git
   cd causal-modeling
   ```

2. Install the required Python packages for both calculation and visualization:
   ```bash
   pip install numpy pandas matplotlib statsmodels scipy networkx tqdm joblib seaborn
   ```

3. Installation Time:
   - Typically less than 5 minutes on a standard machine.

## Usage Instructions

### Step 1: Perform Calculations
Run the `Calculation.py` script to generate simulation data:
   ```bash
   python Calculation.py
   ```
- This will create several `.joblib` files containing the processed results.
- **Note:** Depending on hardware, this step can take a **few hours to several days**.

### Step 2: Visualize Results
Open the Jupyter Notebook `Visualization.ipynb` to explore the results:
   ```bash
   jupyter notebook Visualization.ipynb
   ```
- Make sure the `.joblib` files generated in Step 1 are present in the same directory.
- The notebook will load these files and generate visualizations.

### Expected Outputs
- Graphs illustrating causal activation pathways.
- Histograms of causal activation types categorized by tie range.
- Comparative visualizations for different contagion models:
  - Deterministic activation
  - Constant exposure activation
  - First exposure activation

## Performance Notes
- **Simulation run time (`Calculation.py`)**:  
  Depending on the hardware, the full simulation may take **many hours to several days**.  
  For large-scale simulations, it is **highly recommended to run on a high-performance computing (HPC) cluster** or a machine with many CPU cores.
- **Visualization run time (`Visualization.ipynb`)**:  
  Once the simulation data is available, the visualizations can be generated in **less than 5 minutes**.

## Repository Link
- [GitHub Repository](https://github.com/ftschofenig/causal-modeling)
