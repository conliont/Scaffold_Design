## Scaffold Design Tool for Bone Tissue Engineering

#Overview
This repository contains the implementation of Scaffold Design Tool, a machine learning (ML) and heuristic optimization-based framework developed to facilitate scaffold design for bone tissue engineering. The platform integrates two customizable tools:

Prediction Tool: Forecasts scaffold properties (e.g., Bone Porosity, Connectivity Density) based on user-defined inputs.
Optimization Tool: Utilizes a heuristic genetic algorithm to refine scaffold parameters, enabling users to achieve desired design outcomes.
The tools are packaged in a user-friendly web interface, providing researchers with an accessible platform to design scaffolds that replicate the structural characteristics of healthy and pathological bone tissue.

Features
Prediction Tool:

Developed using Elastic Net Regression models for robust prediction of scaffold properties.
Predicts key parameters such as:
Bone Porosity
Area-to-Volume Ratio
Connectivity Density
Trabecular Spacing
Trabecular Thickness
Designed for seamless integration with 3D design tools like Meshmixer.
Optimization Tool:

Employs a multi-objective evolutionary algorithm to identify optimal scaffold configurations.
Generates a Pareto front of solutions, allowing users to explore trade-offs between competing design objectives.
Web Interface:

Hosted on IBM Cloud with a scalable backend and interactive frontend.
Features an intuitive user interface for defining optimization parameters, visualizing the optimization process, and analyzing final configurations.
