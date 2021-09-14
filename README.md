# Joint Shapley values: a measure of joint feature importance

#### This repo contains the source code for the research paper http://arxiv.org/abs/2107.11357. The paper proposes a _Joint Shapley_ value that extends the widely-used Shapley value, a key measure of feature importance in game theory. The Joint Shapley value can be used to establish the importance of coalitions of features, as well as individual features, whilst maintaining the axiomatic foundations of the original Shapley value.

## Additional resources on the Joint Shapley Values

- Presentation given at the University of Toronto's Vector Institute Seminar, 26 July 2021:

https://www.youtube.com/watch?v=Wrd7JzYA2sE

## Repo Contents
The [Rotten Tomatoes Walkthrough](./rotten-tomatoes-walkthrough.ipynb) Jupyter notebook is a walk-through of the Joint Shapley methodology described in the associated paper, using the [Rotten Tomatoes movie review database](https://www.cs.cornell.edu/people/pabo/movie-review-data/).

The [Boston Housing Dataset Walkthrough](./boston-housing-walkthrough.ipynb) offers a similar approach, but for the Boston housing dataset.

The `datasets` folder contains datasets used in the notebooks:

  - `rt-polaritydata` contains the Rotten Tomatoes data. This is divided into positive reviews (`*.pos`) and negative reviews (`*.neg`). The 'mangled' suffix indicates that the order of the reviews has been scrambled. 
  
The `presentations` folder holds presentation materials regarding the Joint Shapley approach.

## About the authors
Chris Harris - chrisharriscjh@gmail.com

Richard Pymar - r.pymar@bbk.ac.uk

Colin Rowat - c.rowat@bham.ac.uk
