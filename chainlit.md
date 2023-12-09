
# RAG2Rich  
  
***"Rich" answers don't hallucinate. Almost.***  
  

RAG2Rich defines and computes the "Rich" score of answers based on the context relevance (C ), answer relevance (A), and groundedness (G) measures provided by [TruLens](https://www.trulens.org/). The Rich score can be largely thought of as indicative "richness" of answers, on average. Specifically, the Rich metric is computed as:

![Rich score](https://raw.githubusercontent.com/barun-saha/rag2rich/main/img/rich01.png "Rich score")

where X = (G, C, A), P is a set of parameters, and W is a weight vector. 

The currently used RAG parameters are: 
- chunk size = 512
- chunk overlap = 75
- top-k = 4
- Cohere re-ranker top-n = 3


---

![Visitors](https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Frag2rich-vc4smj6o3q-uc.a.run.app%2F&countColor=%23263759&labelStyle=none)