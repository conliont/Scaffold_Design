# Scaffold Design Optimization Tool

## Overview
This repository contains a cloud-based application for optimizing bone scaffold designs using predictive modeling and heuristic optimization. The platform integrates **Elastic Net Regression models** for prediction and a **heuristic evolutionary algorithm** for optimization.

## Key Features
- Predict key scaffold properties: **Bone Porosity**, **Area/Volume Ratio**, **Connectivity Density**, etc.
- Optimize scaffold designs to meet **user-defined objectives**.
- Interactive **web-based interface** for researchers and clinicians.
- Highly customizable parameters for advanced users.

## Technical Details
- **Backend**: PHP 8.1 using Yii 1.1 Framework
- **Optimization Algorithms**: Python-based, integrated with PHP through APIs
- **Frontend**: JavaScript with HTML/CSS
- **Database**: PostgreSQL
- **Hosting**: IBM Cloud



# Scaffold Design Optimization Tool

## Overview
This repository contains a cloud-based application designed to optimize bone scaffold designs using advanced predictive modeling and heuristic optimization. The platform integrates **Elastic Net Regression models** for scaffold property prediction and a **heuristic evolutionary algorithm** for optimization, providing an intuitive and efficient tool for researchers and clinicians in tissue engineering.

## Key Features
- **Predictive Models**:
  - Predict key scaffold properties such as **Bone Porosity**, **Area/Volume Ratio**, **Connectivity Density**, **Trabecular Spacing**, and **Trabecular Thickness**.
  - Models are based on **Elastic Net Regression**, combining L1 and L2 regularization for robustness and interpretability.
  
- **Heuristic Optimization**:
  - Implements an evolutionary algorithm to explore the multidimensional scaffold design space.
  - Generates a **Pareto front** of optimal solutions, offering trade-offs between conflicting objectives.

- **Interactive Web Interface**:
  - User-friendly platform accessible through a web browser.
  - Allows users to set target scaffold properties, adjust algorithm parameters, and visualize results.

- **Scalability**:
  - Cloud-hosted solution ensures robust performance and scalability.
  - Supports a broad range of input parameters and objectives.

## Technical Details
- **Backend**: 
  - PHP 8.1 with the Yii 1.1 framework for compatibility and ease of use.
  - Python for implementing the heuristic optimization algorithms.
  
- **Frontend**:
  - Developed using JavaScript, HTML, and CSS for interactivity and styling.

- **Database**:
  - PostgreSQL, providing reliable and scalable data storage.

- **Hosting**:
  - IBM Cloud, offering robust and secure hosting.

- **Architecture**:
  - Follows a **three-tier architecture**:
    1. **Frontend** interacts with the backend through APIs.
    2. **Backend** processes requests and manages Python computations.
    3. **Database** stores data and supports real-time querying.

## Installation Instructions
1. Clone this repository:
   ```bash
   git clone https://github.com/conliont/Scaffold_Design.git
