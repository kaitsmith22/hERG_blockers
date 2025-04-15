# hERG_blockers
Repository for predicting if molecules are hERG blockers or not. Models include various types of graph neural networks. The false positive rate is controlled with conformal risk control.

## Repo Structure
- `data/`: Contains datasets used for training and testing the models.
- `models/`: Includes implementations of various graph neural network architectures.
- `scripts/`: Utility scripts for finetuning using raytune
- `notebooks/`: Jupyter notebooks for exploratory data analysis and model experimentation.
- `utils/`: Utility functions for plotting, loading data, and running experiments
- `final_report`: Final report summarizing findings

## Running Code 

### Setup environment
1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
2. Activate the virtual environment:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Look at data distributions in `notebooks\EDA.ipynb`
5. Train models in `notebooks\hERG_project.ipynb`

