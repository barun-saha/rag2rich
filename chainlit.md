# RAG2Rich

***"Rich" answers do not hallucinate. Almost.***


## Overview

In Retrieval-Augmented Generation (RAG), document(s) are at first divided into multiple chunks. 
The size of chunks and the overlap between two chunks play an important role in retrieving 
the correct context and answers.

RAG2Rich computes and uses the optimal chunk size and chunk overlap based on three metrics:
- Context relevance (C)
- Answer relevance (A)
- Groundedness (G)

We compute these metrics using TruLens Eval. Subsequently, we combine these metrics to define
the "Rich score" as follows:
$$R = \frac{1}{1 + e^{-W X}},$$ where $$X = (G, C, A)$$ and $$W$$ is a weight vector.


## Data

We use the following [Technical Report](https://www.fit.vut.cz/research/publication-file/11832/TR-61850.pdf) as a data source to demonstrate RAG2Rich:
```
@TECHREPORT{FITPUB11832,
   author = "Petr Matou\v{s}ek",
   title = "Description of IEC 61850 Communication",
   pages = 88,
   year = 2018,
   location = "FIT-TR-2018-01, Brno, CZ",
   publisher = "Faculty of Information Technology BUT",
   language = "english",
   url = "https://www.fit.vut.cz/research/publication/11832"
}
```

The report is used solely for the purpose of demonstration.


## Questions

We submit the following questions to the RAG app:
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

The application generates answers for each of these questions based on the aforementioned document.

We also consider different sets of chunk size and overlap values. 
For each such set, the average richness, $R$, is computed.
The set with the highest value of $R$ is used as the optimal configuration.