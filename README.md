# Crowdfunding Fraud Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)

## Project Overview
This project develops a machine learning model to classify crowdfunding campaigns as potentially fraudulent or legitimate. The system helps identify suspicious campaigns before they can cause financial harm to backers through a user-friendly web interface.

## Problem Statement
Crowdfunding platforms face significant challenges with fraudulent campaigns that can mislead backers and damage platform credibility. This project addresses this issue through a data-driven approach that combines:
- Natural Language Processing (NLP) of project descriptions and titles
- Analysis of funding patterns and campaign metadata
- Interactive web interface for real-time predictions

## Features

### Web Application
- Interactive Streamlit-based web interface
- Real-time fraud prediction
- User-friendly input forms for campaign details
- Visual feedback on prediction results

### Machine Learning
- Pre-trained classification models
- Feature engineering for fraud detection
- Model evaluation and interpretation

## Project Structure
```
crowdfundingproject/
├── crowdfunding/           # Main package directory
│   └── (package files)     # Python package files
│
├── models/                # Trained model files
│   └── (model files)       # Saved model checkpoints
│
├── notebooks/             # Jupyter notebooks for analysis
│   └── (notebook files)    # Data exploration and model development
│
├── processed_data/        # Processed and cleaned data
│   └── (data files)        # Intermediate and final data files
│
├── scripts/               # Python scripts
│   ├── streamlit_app.py    # Main Streamlit application
│   ├── kemikal.py          # Data processing and model utilities
│   └── kiva.py             # Kiva API integration
│
├── visualizations/        # Generated visualizations and plots
│   └── (image files)       # Charts, graphs, and other visual assets
│
├── feature_matrices/      # Feature extraction and transformation
│   ├── X_all.npy           # Processed feature matrix
│   ├── *_model.pkl         # Trained model files
│   └── (other feature files)
│
├── .env                   # Environment variables
├── .gitignore             # Git ignore file
├── requirements.txt       # Python dependencies
└── README.md             # Project documentation
```

## Getting Started

### Prerequisites
- Python 3.8+
- pip (Python package manager)
- Streamlit for the web interface

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/LancemDev/CrowdFunding-Fraud-Detection.git
   cd CrowdFunding-Fraud-Detection
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

### Usage

1. **Run the Streamlit application**
   ```bash
   streamlit run scripts/streamlit_app.py
   ```

2. **Access the web interface**
   - Open your web browser and navigate to the local URL provided in the terminal (typically http://localhost:8501)
   - Fill in the campaign details in the form
   - Click "Predict" to get the fraud prediction

## Technologies Used

### Core Technologies
- Python 3.8+
- Streamlit for web interface
- Scikit-learn for machine learning
- Pandas & NumPy for data processing

### Dependencies
- python-dotenv for environment variables
- scikit-learn for machine learning models
- pandas for data manipulation
- numpy for numerical operations

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
- Streamlit for the amazing web framework
- Open-source contributors to the Python data science ecosystem
- Academic research on fraud detection methodologies
