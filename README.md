# Landslide Susceptibility Modeler

This application is a machine learning tool for landslide susceptibility mapping. It allows you to train and evaluate different machine learning models and use them to generate landslide susceptibility maps from geospatial data.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

*   Python 3.x must be installed on your system.
*   You will need to install the Python libraries listed in the `requirements.txt` file.

### Installation

1.  **Clone the repository or download the source code.**

2.  **Navigate to the project directory:**

    ```bash
    cd path/to/App_PythonGUI
    ```

3.  **Install the required Python packages:**

    It is highly recommended to use a virtual environment to avoid conflicts with other projects.

    ```bash
    # Create a virtual environment (optional but recommended)
    python -m venv venv
    
    # Activate the virtual environment
    # On Windows
    venv\Scripts\activate
    # On macOS and Linux
    source venv/bin/activate
    
    # Install the required packages
    pip install -r requirements.txt
    ```

## Running the Application

Once you have installed the dependencies, you can run the application by executing the `gui.py` script:

```bash
python gui.py
```

This will launch the graphical user interface (GUI) of the Landslide Susceptibility Modeler.

## How to Use the Application

1.  **File Selection:**
    *   **Input CSV:** Select the input CSV file containing your landslide inventory data.
    *   **Raster Folder:** Select the folder containing your raster data (e.g., elevation, slope, aspect).
    *   **Output Folder:** Select the folder where the analysis results will be saved.

2.  **Configuration:**
    *   **Model:** Choose the machine learning model you want to use (e.g., Random Forest, XGBoost).
    *   **Balance Strategy:** Select a strategy to handle imbalanced data (e.g., SMOTE).
    *   **Parallel Workers:** Set the number of CPU cores to use for parallel processing.

3.  **Run Analysis:**
    *   **Compare Models:** Click this button to run a comparison of all available models. The results will be displayed in the log, and the best-performing models will be available in the "Model" dropdown.
    *   **Run Analysis:** After comparing models, select a model from the dropdown and click this button to run the full analysis and generate the landslide susceptibility map.

4.  **Log Output:**
    *   The log window will display the progress and results of the analysis.
