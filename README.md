# Scaffold Design Tool for Bone Tissue Engineering

## Overview
This repository contains the implementation of **Scaffold Design Tool**, a machine learning (ML) and heuristic optimization-based framework developed to facilitate scaffold design for bone tissue engineering. The platform integrates two customizable tools:

- **Prediction Tool:** Forecasts scaffold properties (e.g., Bone Porosity, Connectivity Density) based on user-defined inputs (scaffold design parameters).

- **Optimization Tool:** Utilizes a heuristic genetic algorithm to refine scaffold       parameters, enabling users to achieve desired design outcomes.

The tools are packaged in a user-friendly web interface (available at: https://diagnostics.insybio.com/), providing researchers with an accessible platform to design scaffolds that replicate the structural characteristics of healthy and pathological bone tissue.

## Key Features

- **Prediction Tool:**
  - Developed using Elastic Net Regression models for robust prediction of scaffold properties.
  - Predicts key parameters such as:
    -   Bone Porosity
    -   Area-to-Volume Ratio
    -   Connectivity Density
    -   Trabecular Spacing
    -   Trabecular Thickness
  - Designed for seamless integration with 3D design tools like Meshmixer.

- **Optimization Tool:**
  - Employs a multi-objective evolutionary algorithm to identify optimal scaffold configurations.
  - Generates a Pareto front of solutions, allowing users to explore trade-offs between competing design objectives.

- **Web Interface:**
  - Hosted on IBM Cloud with a scalable backend and interactive frontend.
  - Features an intuitive user interface for defining optimization parameters, visualizing the optimization process, and analyzing final configurations.

## System Architecture

- Frontend: Interacts with users to gather input parameters and display results.
- Backend: Processes user requests, runs Python-based algorithms, and communicates with the database.
- Database: Manages structured data storage for scaffold parameters, results, and logs.

## Overall Flowchart of the proposed methodology.
**A. Flowchart of the proposed Scaffold-Based Prediction and Optimization Workflow.**
![image](https://github.com/user-attachments/assets/9f536137-fce6-4f49-bc6b-96fe6264c83a)

The scaffold parameters are used as input into 3D modeling tools e.g. Meshmixer (blue orthogonal box) for scaffold design. Structural characteristics such as trabecular thickness and connectivity density are calculated (gray orthogonal box) and used to train Elastic Net Regression models with 5-fold cross-validation (red hexagonal box). The trained models underpin:

Prediction Tool: Estimates scaffold performance based on user-provided design parameters (pink orthogonal box).
Optimization Tool: Uses a Multi-objective Evolutionary Algorithm to identify non-dominated solutions (pink orthogonal box).

**B. Flowchart of the proposed Multi-objective Optimization Algorithm.**
![image](https://github.com/user-attachments/assets/e31f0847-8881-465f-b5d3-3a9228ae160d)

Blue orthogonal shapes denote the algorithm’s steps, while the pink one denotes the output (Pareto Front). The light blue rhomboid denotes the termination criterion (number of generations reached). White orthogonal shapes denote a simplified example of an EA’s chromosome (scaffold configurations). The chromosome consists of the Sphere Diameter (SDm), Sphere Distance (SD), Delaunay Mesh Dimension (DMD), and Delaunay Point Spacing (DPS) variables (presented from left to right). Their range of values is indicated in the brackets above them. 

## Tutorial for implementing the tools

**a. Model Training in R:**

The Elastic Net Regression models for scaffold properties (Bone Porosity, Area-to-Volume Ratio, Connectivity Density, Trabecular Spacing, Trabecular Thickness) can be trained using the R script provided in the `scripts/` directory.

To train the models:

- Navigate to the `scripts/` directory.
- Execute the R script (train_models.R) with the required input data.

**b. Prediction Tool:**

The Prediction Tool forecasts scaffold properties based on user-provided scaffold design parameters.

To use the Prediction Tool, run the following command (bash):

    python3 scaffold_optimization.py --sphere_diameter 0.4 --sphere_distance 0.4 --delaunay_mesh 0.21 --delaunay_spacing 0.1 --output_folder /PATH/folder/ --model_dir /PATH/model_folder --scaling_dir /PATH/scaling_folder

| Argument           | Description                                                | Default Value / Range of Values |
|--------------------|------------------------------------------------------------|---------------|
| `--output_folder`  | Directory to save optimization results                     | Required      |
| `--model_dir`      | Path to trained R models                                   | Required      |
| `--scaling_dir`    | Path to scaling parameters                                 | Required      |
| `--sphere_diameter`| Input scaffold design parameter      | 0.1 - 1          |
| `--sphere_distance`     | Input scaffold design parameter       | 0.1 - 1            |
| `--delaunay_mesh`    | Input scaffold design parameter | 0.12 - 0.9            |
| `--delaunay_spacing`    | Input scaffold design parameter | 0.1 - 0.9            |

A CSV file (`predictions.csv`) containing the predicted values for scaffold properties based on user-defined input parameters will be stored in the specified `output_folder`.

**c. Optimization Tool:**

The Optimization Tool refines scaffold parameters to meet user-defined property targets using a heuristic genetic algorithm.

To use the Optimization Tool, run the following command (bash):

    python3 scaffold_optimization.py --output_folder /PATH/folder/ --model_dir /PATH/model_folder --scaling_dir /PATH/scaling_folder --desired_outputs  12 70 8 0.05 0.4 --population 10 --generations 20

| Argument           | Description                                                | Default Value / Range of Values |
|--------------------|------------------------------------------------------------|---------------|
| `--output_folder`  | Directory to save optimization results                     | Required      |
| `--model_dir`      | Path to trained R models                                   | Required      |
| `--scaling_dir`    | Path to scaling parameters                                 | Required      |
| `--desired_outputs`| Space-separated target values for scaffold properties: Area-to-Volume Ratio, Bone Porosity, Connectivity Density, Trabecular Thickness and Trabecular Spacing      | 4.5 - 28.2 / 39.1 - 100 / 0.3 - 22.9 / 0.02 - 0.2 / 0.09 - 0.93          |
| `--population`     | Number of individuals in the optimization population       | 50            |
| `--generations`    | Maximum number of generations for the optimization process | 300           |

**!!!!!** A CSV file (`Optimization_Pareto_Front.csv`) containing non-dominated scaffold configurations that meet user-defined targets.
- Visualization files (`Optimization_Visualization/`) illustrating optimization progress and results. **!!!!!**

## Acknowledgements
The present work has been developed with the funding support from the European Union’s Horizon Europe research and innovation programme OSTEONET (In vitro 3d cells models of healthy and OSTEOpathological ageing bone tissue for implantation and drug testing in a multidisciplinary NETwork, https://osteonethorizon.com/), under the Marie Sklodowska-Curie Grant Agreement Action (No. 101086329). 
  ![image](https://github.com/user-attachments/assets/6ae3e457-66b5-4086-a513-7b50a9c24356) 
