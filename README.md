Hawkes Process Modeling for High-Frequency Limit Order Book Data
This repository explores the application of Hawkes processes to model high-frequency limit order book (LOB) data. It includes implementations for both real-world data and simulated data, along with a detailed PDF document summarizing the study. The focus is on power-law kernels, but the framework can be adapted for exponential kernels with minimal changes.

Repository Structure
Hawkes_Processes_in_High_Frequency_Trading___Neal_Batra.pdf/: A comprehensive PDF document detailing the study, including the methodology, results, and insights.

real_data_power_kernel.py/: Code for analyzing real-world high-frequency trading data.

simulated_data_power_kernel.py/: Code for generating and analyzing simulated data using Hawkes processes.

How to Use
1. PDF Document
The paper.pdf file provides a detailed overview of the study, including:

The theoretical framework of Hawkes processes.

Implementation details for both real and simulated cases.

Results, insights, and a derived trading strategy.

Simply open the paper.pdf file to explore the study.

2. Real Case Implementation
The code in code/real_data_power_kernel.py/ is designed to analyze real-world high-frequency trading data using Hawkes processes with power-law kernels.

Steps to run:

Navigate to code/real_data_power_kernel.py/.

Place your high-frequency trading data in the data/ directory (or update the path in the script). This must be a csv file

Run the main script:

3. Simulated Case Implementation
The code in code/simulated_data_power_kernel.py/ generates synthetic data using Hawkes processes with power-law kernels and analyzes its properties.

Steps to run:

Navigate to code/simulated_data_power_kernel.py/.

Run the simulation script:

The script will generate synthetic data using the brownian motion model, analyze it, and create a graph of the intensity function against time.

Adapting for Exponential Kernels
The provided code uses power-law kernels, but it can be easily adapted for exponential kernels. To do this:

Locate the kernel formula in the code (e.g., in real_case_analysis.py or simulated_case_analysis.py).

Replace the power-law kernel formula with the exponential kernel formula as seen in the pdf

Key Features
Real Case Analysis: Models real-world high-frequency trading data using Hawkes processes.

Simulated Case Analysis: Generates and analyzes synthetic data to validate the model.

Power-Law Kernels: Focused on capturing long-memory effects and heavy-tailed behavior.

Flexibility: Easily adaptable for exponential kernels by modifying the kernel formula.

Dependencies
Python 3.8+

NumPy

SciPy

Matplotlib

Pandas

Contributing
Contributions are welcome! If you have suggestions, improvements, or additional features, please open an issue or submit a pull request.

License
This project is licensed under the MIT License. See the LICENSE file for details.
