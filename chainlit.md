
# RAG2Rich  
  
***"Rich" answers don't hallucinate. Almost.***  
  

RAG2Rich uses the [TruLens](https://www.trulens.org/) to compute context relevance (C ), answer relevance (A), and groundedness (G). Subsequently, these metrics are combined to define a new composite metric, Rich score, as follows:

![Rich score](https://raw.githubusercontent.com/barun-saha/rag2rich/main/img/rich01.png "Rich score")

where X = (G, C, A), P is a set of parameters, and W is a weight vector. For example, based on the optimization, we set: chunk size = 512, chunk overlap = 100, and top-k = 6.
