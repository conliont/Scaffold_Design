# Scaffold Design Tool for Bone Tissue Engineering

## Overview
This repository contains the implementation of **Scaffold Design Tool**, a machine learning (ML) and heuristic optimization-based framework developed to facilitate scaffold design for bone tissue engineering. The platform integrates two customizable tools:

- **Prediction Tool:** Forecasts scaffold properties (e.g., Bone Porosity, Connectivity Density) based on user-defined inputs (scaffold design parameters).

- **Optimization Tool:** Utilizes a heuristic genetic algorithm to refine scaffold       parameters, enabling users to achieve desired design outcomes.

The tools are packaged in a user-friendly web interface (https://diagnostics.insybio.com/), providing researchers with an accessible platform to design scaffolds that replicate the structural characteristics of healthy and pathological bone tissue.

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
 
## Technical Details or System Architecture
r 
- **Backend**: PHP 8.1 using Yii 1.1 Framework
- **Optimization Algorithms**: Python-based, integrated with PHP through APIs
- **Frontend**: JavaScript with HTML/CSS
- **Database**: PostgreSQL
- **Hosting**: IBM Cloud

## Overall Flowchart of the proposed methodology.
**A. Flowchart of the proposed Scaffold-Based Prediction and Optimization Workflow.**
![image](https://github.com/user-attachments/assets/9f536137-fce6-4f49-bc6b-96fe6264c83a)

The scaffold parameters are used as input into MeshMixer (blue orthogonal box) for scaffold design. Structural scaffold characteristics, such as trabecular thickness and connectivity density, are subsequently calculated (gray orthogonal box). These parameters, along with scaffold characteristics, are used to train ElasticNet Regression models with 5-fold cross-validation (red hexagonal box). The trained models form the foundation for two tools: the Prediction Tool (pink orthogonal box), which estimates scaffold performance based on input design parameters, and the Optimization Tool (pink orthogonal box), which uses a Multi-objective Evolutionary Algorithm to identify non-dominated solutions for scaffold design.

**B. Flowchart of the proposed Multi-objective Optimization Algorithm.**
![image](https://github.com/user-attachments/assets/e31f0847-8881-465f-b5d3-3a9228ae160d)

Blue orthogonal shapes denote the algorithm’s steps, while the pink one denotes the output (Pareto Front). The light blue rhomboid denotes the termination criterion (number of generations reached). White orthogonal shapes denote a simplified example of an EA’s chromosome (scaffold configurations). The chromosome consists of the Sphere Diameter (SDm), Sphere Distance (SD), Delaunay Mesh Dimension (DMD), and Delaunay Point Spacing (DPS) variables (presented from left to right). Their range of values is indicated in the brackets above them. 

## Tutorial for implementing the tools

**Model Training in R**
The code for creating the 5 models, one for each parameter (Bone Porosity, Area-to-Volume Ratio, Connectivity Density, Trabecular Spacing, Trabecular Thickness) can be found in the directory 
**Prediction Tool**

**Optimization Tool**

## Acknowledgements
The present work has been developed with the funding support from the European Union’s Horizon Europe research and innovation programme OSTEONET (In vitro 3d cells models of healthy and OSTEOpathological ageing bone tissue for implantation and drug testing in a multidisciplinary NETwork, https://osteonethorizon.com/), under the Marie Sklodowska-Curie Grant Agreement Action (No. 101086329). 
  ![image](https://github.com/user-attachments/assets/6ae3e457-66b5-4086-a513-7b50a9c24356) 
