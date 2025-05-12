# Assignment 1 Overview

This repository contains the files and code for Homework Assignment 1 in AML. The key parts include:

1. **Assignment Files**  
   - PDF documents outlining the tasks and requirements (e.g., Assignment1.pdf, Programming_Assignment_1_Report.pdf).
   - LaTeX source (main.tex) for generating a detailed report of the assignment.

2. **Solution Code**  
   - `solution_final.py` implements the required functions for performing inference on a probabilistic graphical model. It includes a `Factor` class, an `Inference` class, and methods for tasks like triangulation, message passing, marginal computations, and top-k assignments.
   - `TestCases.json` contains input data for testing the implementation. It includes multiple cases for verifying correctness.

3. **Purpose**  
   - The goal was to write a program that constructs and solves a graphical model problem using standard Python libraries only.  
   - We examined cliques, potentials, and used an inference process (triangulation, constructing a junction tree, summing over assignments) to compute probabilities, marginals, top-k assignments, and a partition function (Z value).

4. **How To Use**  
   - Review the LaTeX report in `main.tex` for an explanation of the approach and the results.
   - Run `solution_final.py` to generate the outputs for the given test cases in `TestCases.json`.

Feel free to update this README with additional details or clarifications as needed.
