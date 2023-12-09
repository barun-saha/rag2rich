
# RAG2Rich  
  
***"Rich" answers don't hallucinate. Almost.***  
  
  
## Overview  
  
In Retrieval-Augmented Generation (RAG), document(s) are at first divided into multiple chunks.  The size of chunks and the overlap between two chunks, among other parameters, play an important role in retrieving the correct context and answers.  
  
RAG2Rich computes and uses the optimal configurations based on three metrics:  
- Context relevance (C )  
- Answer relevance (A)  
- Groundedness (G)  
  
We compute these metrics using [TruLens Eval](https://www.trulens.org/). Subsequently, we combine these metrics to define "Rich score" as follows:  
$R(X; P) = \frac{1}{1 + e^{-WX}}$,

where $X = (G, C, A)$, $P$ is a set of parameters, and $W$ is a weight vector.  


## RAG Optimization
  
We use the publicly available Technical Report titled [Description of IEC 61850 Communication](https://www.fit.vut.cz/research/publication-file/11832/TR-61850.pdf) as the data source to demonstrate RAG2Rich. The report is used solely for the purpose of demonstration, and it is not distributed with the code.  
  
RAG2Rich is tuned by considering the following questions:  
- What is IEC 61850?  
- Tell me about digital subtations.  
- What the expansion of GOOSE?  
- What are the different fields in a GOOSE packet?  
- What are physical and logical devices?  
- How so the stNum and sqNum values change?  
- Show me a list of different data types supported by the standard.  
- How does MMS communication work?  
- Are GOOSE messages encrypted?  
- What is a data set?  
  
Based on the above-mentioned document, the application generates answers for each of these questions.  The output score vector $X$ is obtained by varying the different parameters in $P$, such as:  
- chunk size  
- chunk overlap  
- top-k 
  
For each such set, the average richness, $R$, is computed.  The set with the highest value of $R$ is used as the optimal configuration. 

The currently used RAG parameters are: 
- chunk size = 512
- chunk overlap = 75
- top-k = 4
- Cohere re-ranker top-n = 3


## Usage

Install the dependencies:

`pip install -r requirements.txt`

To run RAG2Rich, the document Q&A application, execute the following command:
```python
chainlit run app_llamaindex.py
```

To run RAG fine-tuning experiments, execute the following command in the `evaluation_trulens` directory:

`python experiments.py`

Currently, the measurements are manually copied from the dashboard to CSV files (see examples in the `evaluation_trulens` directory). When these data are available, find the optimal settings by running:

`python optimal_settings.py`

The output shows the optimal value of $R$, the corresponding index of data, and the measurements, $X$.