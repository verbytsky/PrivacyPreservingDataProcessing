# Privacy-Preserving Data Processing

This project demonstrates privacy-preserving data processing techniques using financial data. The project implements a privacy-preserving machine learning model and data processing pipeline using the `diffprivlib` library. The primary focus of the project is to showcase the use of differential privacy in machine learning while discussing the trade-offs between privacy and utility.

## Table of Contents
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Privacy and Utility Trade-offs](#privacy-and-utility-trade-offs)
- [License](#license)

## Project Structure

```
.
├── data
│   ├── processed
│   └── raw
├── models
└── src
    ├── features
    ├── models
    └── utils
```

- `data`: Contains raw and processed financial data
- `models`: Contains the trained machine learning models
- `src`: Contains the source code for data processing, feature extraction, model training, and evaluation

## Getting Started

1. Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/jrslww/PrivacyPreservingDataProcessing.git
cd PrivacyPreservingDataProcessing
```

2. Set up a virtual environment and install the required packages:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. Run the data preprocessing script to process the raw financial data:

```bash
python src/data/preprocess_data.py
```

4. Train the privacy-preserving machine learning model using differential privacy:

```bash
python src/models/train_model.py
```

5. Evaluate the trained model and calculate the privacy and utility trade-offs:

```bash
python src/models/evaluate_model.py
```

## Privacy and Utility Trade-offs

This project uses differential privacy to provide a balance between privacy and utility in machine learning. The primary trade-off is controlled by the `epsilon` parameter. A smaller `epsilon` value provides stronger privacy guarantees at the cost of reduced model accuracy. Conversely, a larger `epsilon` value results in weaker privacy guarantees but increased model accuracy.

The choice of privacy-preserving technique and its implications for data processing and analysis is crucial. In this project, we used the `diffprivlib` library, which provides a user-friendly interface for implementing differential privacy in Python. Some limitations of the current implementation include:

- The use of a single privacy-preserving technique (differential privacy) without exploring other methods like homomorphic encryption
- The potential for additional privacy leakage when calculating data bounds, which could be mitigated by specifying bounds for each dimension

Possible improvements and alternative methods include:

- Implementing other privacy-preserving techniques like homomorphic encryption or secure multi-party computation
- Comparing the performance and privacy trade-offs of various techniques to find the optimal balance for the specific use case

## License

This project is licensed under the MIT License.