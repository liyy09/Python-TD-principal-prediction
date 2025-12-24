# Python TD Principal Prediction

This repository contains the dataset and code used in our research paper —— `Predicting Technical Debt Principal in Python
Projects Based on Machine Learning`. It includes the dataset of technical debt principal of 10 Python projects, specifically Colossal-AI, DeepSpeed, Gradio, Keras, PyTorch, Ray, Streamlit, Transformers, Ultralytics, YOLOv5.

## Setting Up the Experiment Environment
1. Clone the repository
   ```bash
   git clone https://github.com/liyy09/Python-TD-principal-prediction
   ```

2. Install dependencies

   ```
   pip install -r requirements.txt
   ```

   

3. Modify the config. py configuration file

   ```python
   # set features
   features = ['ncloc', 'lines', 'files', 'classes', 'functions', 'comment_lines',
               'bugs', 'vulnerabilities', 'code_smells', 'uncovered_lines',
               'duplicated_lines', 'duplicated_blocks', 'duplicated_files', 'complexity',
               'LSC', 'LBC', 'LC', 'LTCE', 'CCC', 'LMC', 'MNC', 
               'software_quality_maintainability_remediation_effort']
   
   # set dataset path
   project='textual'
   path = f'{project}_dataset.csv'
   
   # set prediction period
   start = 7
   end = start+1



