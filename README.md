# Adaptive-Causal-Directed-Graph

## 📝 Description

This project generates an **Adaptive Causal Directed Graph (ACDG)** from an AML (Automation Markup Language) file. It's designed to assist in monitoring alarm floods and analyzing process graphs. The code is tailored for the dataset format of the well-known **Tennessee Eastman** process and includes several test cases to demonstrate functionality and validate results. By adapting the dataset format, the code ACDG can also be applied to other processes, provided that a valid **Automation Markup Language AML** file exists. 

---

## 🛠️ Installation

1. **Download the Repository**  
   Clone or download all project files into a single folder on your local machine.

2. **Open with Visual Studio Code**  
   Launch [Visual Studio Code](https://code.visualstudio.com/) and open the folder containing the downloaded files.

---

## 📦 Requirements

- Python 3.11  
- Python packages:
  - `igraph`
  - `plotly`
  - `numpy`

---

## 📁 AML File Requirements

The provided AML file of the TEP serves both as input and as an example of usage. Additionally, an AML file for a **Fluidized Catalytic Cracking (FCC)** [dataset](https://rdms.rd.ruhr-uni-bochum.de/concern/datasets/2v23vv393?locale=en) has been provided.
To ensure compatibility with this code, the AML file should follow these structural and semantic conventions:

### 🔧 Internal Element Attributes

- **Concentration Sensors**
  - Must include an attribute containing the word `"Measurement_Concentration"`.
  - The specific material being measured should be defined as a **sub-attribute** under `"Measurement_Concentration"`, and these sub-attributes should contain the word `"Concentration"`.

- **Temperature Sensors**
  - Must include an attribute containing the word `"Measurement_Temperature"`.

- **Actuators with Temperature Control**
  - Must include an attribute containing the word `"Temperature"`.

- **Flow Sensors**
  - Must include an attribute containing the word `"Measurement_Flow"`.

- **Actuators with Flow Control**
  - Must include an attribute containing the word `"Flow"`.

- **Pressure Sensors**
  - Must include an attribute containing the word `"Measurement_Pressure"`.

- **Actuators with Pressure Control**
  - Must include an attribute containing the word `"Pressure"`.

### 🌡️ Thermal and Valve Components

- **Thermal Contacts** (e.g., cooling pipes)
  - Should have an attribute named `"Thermal"` or belong to a class that includes the word `"Thermal"`.

- **Valves**
  - Class name should include the word `"Valve"`.
  - Must define an attribute named `"State"`.

### 📊 Sensor Values

- Measurement values for sensors—especially **concentration** and **temperature**—should be explicitly defined in the `value` field of the corresponding attribute.

---

## 🚀 Usage

To run the code, you’ll need to update the file paths in `main.py`. These paths point to your AML file.

### 🔧 Required Inputs

In `main.py`, update the following variables with your local file paths:

```python
aml_file = "C:/path/to/your/file.aml"

```

## 📄 License

License: This project is licensed under the MIT License.




