# Scaffold Design Tool for Bone Tissue Engineering

## Overview
This repository contains the implementation of **Scaffold Design Tool**, a machine learning (ML) and heuristic optimization-based framework developed to facilitate scaffold design for bone tissue engineering. The platform integrates two customizable tools:

- **Prediction Tool:** Forecasts scaffold properties (e.g., Bone Porosity, Connectivity Density) based on user-defined inputs (scaffold design parameters).

- **Optimization Tool:** Utilizes a heuristic genetic algorithm to refine scaffold parameters, enabling users to achieve desired design outcomes.

The tools are packaged in a user-friendly web interface (available at: https://diagnostics.insybio.com/), providing researchers with an accessible platform to design scaffolds that replicate the structural characteristics of healthy and pathological bone tissue.

---

## Key Features

### Prediction Tool
  - Developed using Elastic Net Regression models for robust prediction of scaffold properties.
  - Predicts key parameters such as:
    -   Bone Porosity
    -   Area-to-Volume Ratio
    -   Connectivity Density
    -   Trabecular Spacing
    -   Trabecular Thickness
  - Designed for seamless integration with 3D design tools like Meshmixer.

### Optimization Tool
  - Employs a multi-objective evolutionary algorithm to identify optimal scaffold configurations.
  - Generates a Pareto front of solutions, allowing users to explore trade-offs between competing design objectives.

### Web Interface
  - Hosted on IBM Cloud with a scalable backend and interactive frontend.
  - Features an intuitive user interface for defining optimization parameters, visualizing the optimization process, and analyzing final configurations.

---

## System Architecture

- Frontend: Interacts with users to gather input parameters and display results.
- Backend: Processes user requests, runs Python-based algorithms, and communicates with the database.
- Database: Manages structured data storage for scaffold parameters, results, and logs.
  

## Tutorial for Using the Tools

### a. Model Training in R
[This step is optional. You can download the already trained models from the `models/` directory and use the tools directly.]

The Elastic Net Regression models for scaffold properties (Bone Porosity, Area-to-Volume Ratio, Connectivity Density, Trabecular Spacing, Trabecular Thickness) can be trained using the R script provided in the `scripts/` directory.

To train the models:

- Navigate to the `scripts/elastic_net_models` directory.
- Execute the R scripts (area volume.R , bone porosity.R) with the required input data.

### b. Prediction Tool

The Prediction Tool forecasts scaffold properties based on user-provided scaffold design parameters.

**Command (bash):**

    python3 scaffold_design.py --sphere_diameter 0.4 --sphere_distance 0.4 --delaunay_mesh 0.21 --delaunay_spacing 0.1 --output_folder /PATH/folder/ --model_dir /PATH/model_folder --scaling_dir /PATH/scaling_folder

**Arguments:**

| Argument           | Description                                                | Default Value / Range of Values |
|--------------------|------------------------------------------------------------|---------------|
| `--output_folder`  | Directory to save optimization results                     | Required      |
| `--model_dir`      | Path to trained R models                                   | Required      |
| `--scaling_dir`    | Path to scaling parameters                                 | Required      |
| `--sphere_diameter`| Input scaffold design parameter      | 0.1 - 1          |
| `--sphere_distance`     | Input scaffold design parameter       | 0.1 - 1            |
| `--delaunay_mesh`    | Input scaffold design parameter | 0.12 - 0.9            |
| `--delaunay_spacing`    | Input scaffold design parameter | 0.1 - 0.9            |

**Output:**
A CSV file (`predictions.csv`) containing the predicted values for scaffold properties based on user-defined input parameters will be stored in the specified `output_folder`.

### c. Optimization Tool

The Optimization Tool refines scaffold parameters to meet user-defined property targets using a heuristic genetic algorithm.

**Command (bash):**

    python3 scaffold_design.py --output_folder /PATH/folder/ --model_dir /PATH/model_folder --scaling_dir /PATH/scaling_folder --desired_outputs  12 70 8 0.05 0.4 --population 10 --generations 20
    
**Arguments:**

| Argument           | Description                                                | Default Value / Range of Values |
|--------------------|------------------------------------------------------------|---------------|
| `--output_folder`  | Directory to save optimization results                     | Required      |
| `--model_dir`      | Path to trained R models                                   | Required      |
| `--scaling_dir`    | Path to scaling parameters                                 | Required      |
| `--desired_outputs`| Space-separated target values for scaffold properties: Area-to-Volume Ratio, Bone Porosity, Connectivity Density, Trabecular Thickness, and Trabecular Spacing. Example: `12 70 8 0.05 0.4` | Ranges: 4.5-28.2 / 39.1-100 / 0.3-22.9 / 0.02-0.2 / 0.09-0.93 |
| `--population`     | Number of individuals in the optimization population       | 50            |
| `--generations`    | Maximum number of generations for the optimization process | 300           |

**Outputs:**
- **CSV Outputs:**
  - `unique_solutions.csv`: Contains unique scaffold configurations, with each row representing a solution evaluated during optimization.
  - `final_predictions_vs_desired.csv`: Provides predictions for scaffold properties alongside user-defined targets, including absolute errors for each property.
- **Graph Outputs:**
  - `AvgOfAvgGoalsPerGen_plot.png`: Visualizes the average performance of the population across generations. Stabilization indicates convergence toward optimal solutions.
  - `BestPerformancePerGen_plot.png`: Tracks the best-performing individual per generation, showcasing iterative improvement.
---
## Installation 

### Prerequisites
- **Python 3.8+**
- **R 4.0+**

### Steps
1. Set up Python dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2. Install R dependencies:
    ```R
    install.packages(c("glmnet", "caret", "e1071"))    

---
## Access to the Web Interface
The web interface is available at: https://diagnostics.insybio.com

If you would like to use the platform, please contact us to request a free account. 

---
## How to create the scaffold library (optional)

### 1. Algorithm for creating the scaffolds remotely

**Overview**

The algorithm was implemented based on the existing code available at this link (https://github.com/meshmixer/mm-api/tree/master), which enables automated and remote access to Meshmixer. Building upon this foundation, the following algorithm was developed to automate the creation of scaffolds using key parameters chosen from the users **(Sphere Distance, Sphere Diameter, Delaunay Mesh Dimension and Delaunay Point Spacing)**. The scaffold creation process was implemented based on the article Eleonora Zenobi et al., ‘Tailoring the Microarchitectures of 3D Printed Bone-like Scaffolds for Tissue Engineering Applications’, Bioengineering 10, no. 5 (May 2023), which details the generation process using Meshmixer to create random scaffolds with a specific porosity and defined characteristics. This approach ensures that the methodology follows established scientific principles for scaffold design. These parameters can be specified within a CSV file, where each row represents a scaffold to be created, allowing users to define the exact number of scaffolds needed. The user is free to modify the code according to their preferred method for scaffold generation, allowing for flexibility in customization and adaptation to specific needs, such as the shape of the scaffold base and the way to create the desired porosity.

**How are scaffolds created using this tool**

1. The base scaffold from which the structure is generated is imported. In this case, it is a parallelepiped, but the user can choose any shape and size according to their preferences.
   
   <div align="center">
       <img src="https://github.com/user-attachments/assets/99741639-d70d-419c-a62a-79bf77d76ecd" width="40%">
   </div>
   
2. By opening **Make Pattern** and selecting **Random Primitives**, the process to achieve the desired porosity starts. The user selects a sphere diameter and a distance between spheres equal to the sphere diameter. These spheres are then randomly distributed within the parallelepiped by the software, ensuring a controlled yet stochastic scaffold arrangement. After selecting the **Clip to Surface** option, these spheres are subtracted from the parallelepiped, creating the porous structure of the scaffold.
   
   <div align="center">
       <img src="https://github.com/user-attachments/assets/6de5e6ae-c258-47a4-affd-87ac51817714" width="40%">
       <img src="https://github.com/user-attachments/assets/43cd843a-1877-4a4b-be82-61050e59efc4" width="40%">
   </div>

3. Next, by selecting **Make Pattern** again, the **Mesh and Delaunay Edges** options are chosen, allowing the user to define the Mesh Dimension and Point Spacing parameters.
   
   <div align="center">
       <img src="https://github.com/user-attachments/assets/f1db5718-00b5-41e2-a49e-a93a1a420a39" width="40%">
       <img src="https://github.com/user-attachments/assets/787c1018-e83d-4d51-8361-9ba423fc59c6" width="40%">
   </div>

4. Finally, the last two steps involve converting the mesh into a **solid**, allowing it to be imported into software like Autodesk and making it suitable for 3D printing. The final step is an **inspection** to remove any artifacts generated during the process. A **threshold** is set to determine the size of small objects to be eliminated, ensuring a clean and optimized scaffold structure.

**How to use this tool**

    • Python version: 2.7.18 (32-bit)
    • Meshmixer version: 3.5

1. Set up Python dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2. Inside the **random_parameters** file, enter the values for **Sphere Diameter, Sphere Distance, Mesh Delaunay Dimension and Point Spacing Dimension** for all the scaffolds you wish to generate.
   <div align="center">
       <img src="https://github.com/user-attachments/assets/1028acd8-e738-4439-a1a9-85ea958f0a94" width="40%">
   </div>


3. Create a **scaffold_base.stl** as the starting structure. In this case, a **10x10x3 mm parallelepiped** was chosen.

To proceed with scaffold creation, it is necessary to open Meshmixer and run the testScaffold.py script. Once the tool is running, the user will be prompted to select the **CSV file** containing the parameters, the **STL file** serving as the base for scaffold construction, and the **destination folder** to save the generated scaffolds.

**Outputs:**
As output, the tool generates as many STL scaffolds as the number of rows in the previously compiled CSV file. Each scaffold is named according to the parameters specified in the CSV file.

![image](https://github.com/user-attachments/assets/68a25c3b-70a3-4d99-90c7-b86a4ff61bbf)


### 2. Algorithm for calculating scaffold parameters

**Overview**

This algorithm was developed to calculate specific scaffold parameters that can be correlated with bone-related properties. This allows for the identification of the type of bone being reproduced—whether healthy or pathological—and its corresponding shape. Once the script is run, the user selects the folder containing all the previously generated **STL scaffolds** created with the other algorithm. The script then calculates specific parameters that can be used to compare the scaffold with human bone and to distinguish between healthy and pathological bones. The calculated parameters include **Bone Surface/Bone Volume, Porosity, Connectivity Density, Trabecular Thickness, and Trabecular Spacing**. The parameters have been calculated based on the definitions provided on the **BoneJ** website. BoneJ is an open-source software plugin for **ImageJ**, a popular image processing program. BoneJ provides a set of tools designed to analyze and quantify bone microstructure from 2D and 3D images, such as micro-CT scans or MRI data.

    • Python version: 3.11+

Set up Python dependencies:
    ```bash
    pip install -r requirements.txt
    ```

**Outputs:**
The parameters are then saved in an **Excel file** named **"results"**, which will be used to build the database for training the model presented previously.

---

## Acknowledgements

The present work has been developed with the funding support from the European Union’s Horizon Europe research and innovation programme **OSTEONET (In vitro 3d cells models of healthy and OSTEOpathological ageing bone tissue for implantation and drug testing in a multidisciplinary NETwork, https://osteonethorizon.com/)**, under the Marie Sklodowska-Curie Grant Agreement Action **(No. 101086329)**. 

![image](https://github.com/user-attachments/assets/4dde35fc-0256-4d87-a7ff-8801807e681e)

