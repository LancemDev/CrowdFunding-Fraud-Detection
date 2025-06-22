# Crowdfunding Fraud Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Project Overview
This project develops a machine learning model to classify crowdfunding campaigns as potentially fraudulent or legitimate. By analyzing project descriptions, funding patterns, and metadata, the system helps identify suspicious campaigns before they can cause financial harm to backers.

## Problem Statement
Crowdfunding platforms face significant challenges with fraudulent campaigns that can mislead backers and damage platform credibility. This project addresses this issue through a data-driven approach that combines:
- Natural Language Processing (NLP) of project descriptions and titles
- Analysis of funding patterns and backer behavior
- Examination of creator history and campaign metadata

## Features

### Data Collection
- Automated scraping of crowdfunding platforms
- Integration with Kaggle datasets
- Data preprocessing and cleaning pipelines

### Analysis
- Exploratory Data Analysis (EDA) with Jupyter Notebooks
- Interactive visualizations
- Statistical analysis of campaign patterns

### Machine Learning
- Feature engineering for fraud detection
- Multiple classification algorithms (Random Forest, XGBoost, etc.)
- Model evaluation and interpretation

## Dataset
The project utilizes multiple data sources:
1. **Kickstarter Projects Dataset**
   - Project titles, descriptions, and updates
   - Funding goals and amounts raised
   - Backer statistics and campaign durations
   - Project categories and outcomes

2. **Additional Sources**
   - Live campaign data from various platforms
   - Historical success/failure rates
   - Creator history and reputation metrics

## Project Structure
```
crowdfunding-project/
├── data/                   # Raw data files
├── preprocessed_data/      # Cleaned and processed data
├── notebooks/              # Jupyter notebooks for analysis
├── scripts/                # Python scripts for data processing
├── models/                 # Trained models and weights
├── visualizations/         # Generated plots and figures
├── requirements.txt        # Python dependencies
└── README.md              # Project documentation
```

## Getting Started

### Prerequisites
- Python 3.8+
- pip (Python package manager)
- Jupyter Notebook (for interactive analysis)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/crowdfunding-fraud-detection.git
   cd crowdfunding-fraud-detection
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Kaggle API** (for dataset access)
   - Create a Kaggle account if you don't have one
   - Go to Account -> Create New API Token
   - Place the downloaded `kaggle.json` in `~/.kaggle/`
   - Make it read-only: `chmod 600 ~/.kaggle/kaggle.json`

### Usage

1. **Run the data pipeline**
   ```bash
   python scripts/data_pipeline.py
   ```

2. **Explore the data**
   ```bash
   jupyter notebook notebooks/exploratory_analysis.ipynb
   ```

3. **Train the model**
   ```bash
   python scripts/train_model.py
   ```

## Technologies Used

### Core Technologies
- Python 3.8+
- Scikit-learn
- Pandas & NumPy
- Matplotlib/Seaborn
- Jupyter Notebooks

### Additional Tools
- Kaggle API for dataset access
- Scrapy for web scraping
- Plotly for interactive visualizations

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Kaggle for providing the Kickstarter dataset
- Open-source contributors to the Python data science ecosystem
- Academic research on fraud detection methodologies

## License
This project is open-source and available under the [MIT License](LICENSE).

## Contributing
Contributions are welcome! Please read our contributing guidelines before submitting pull requests.
