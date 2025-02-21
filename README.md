# Evaluating the Effectiveness of Large Language Model Powered Agents for Material Selection

## Abstract

Material selection is fundamental to the design process, as it significantly impacts a product’s cost, performance, appearance, and manufacturability. It is a complex, open-ended challenge that forces designers to continuously adapt to new information, balance diverse stakeholder demands, weigh trade-offs, and navigate uncertainties to achieve the optimal outcome. Previous studies have explored the potential of Large Language Models (LLMs) to assist in the material selection process, with findings suggesting that LLMs could provide valuable support. However, discrepancies between LLM outputs and expert recommendations indicate the need for further research. To address the limitations of LLMs, AI agents have been developed by building additional capabilities on top of LLMs, allowing them to retrieve and analyze external information, use specialized tools for reasoning, and refine their outputs based on iterative feedback. This study investigates how LLM-powered agents, equipped with advanced search tools, can more effectively emulate expert decision-making, offering insights into the integration of AI into the design process.

## Repository

```
📦 Agents-for-Material-Selection
├─ Data  # Data collected
├─ Data Evaluation  # Scripts used to evaluate the results
├─ Data Generation  # Scripts used to generate the results
├─ Search Logs  # Search logs collected
└─ Search Logs Evaluation  # Scripts used to evaluate the search logs
```

## Installation

If running the code with Google search:
```
pip install pandas seaborn matplotlib scipy scikit-learn statsmodels numpy requests pathvalidate langchain-community load-dotenv transformers regex llama-cpp markdownify puremagic mammoth python-pptx beautifulsoup4 pdfminer.six youtube-transcript-api
```

If running the code with Wikipedia search:
```
pip install pandas seaborn matplotlib scipy scikit-learn statsmodels numpy langchain-community transformers llama-cpp
```
