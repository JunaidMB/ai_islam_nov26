# Process Reward Models (PRM)

- PRMs are reward models trained to output scores at every step in a chain-of-thought reasoning process.

- A standard RM outputs a score only at an EOS token. An ORM outputs a score at every token.

- PRMs require supervision at the end of each reasoning step and are trained similarly where then, tokens in the reasoning step are trained to their relevant target.

- A binary labeled PRM is commonly optimised with a per-step cross entropy loss:

$$L_{PRM}(\theta) = -\mathbb{E}_{(x,s) \sim \mathcal{D}} \left[ \sum_{i=1}^{K} y_i \log r_\theta(s_i|x) + (1-y_{s_{i}}) \log (1 - r_\theta(s_i|x)) \right]$$

where
- $s$ is a sampled chain of thought with $K$ annotated steps
- $y_{s_{i}} \in \{0, 1\}$ denotes whether the $i$-th step is correct
- $r_\theta(s_i|x)$ is the PRM predicted probability that step $s_i$ is valid conditioned on the prompt $x$
- PRMs are trained with a language modeling head that outputs a token only at the end of a reasoning step - when a token corresponding to a double new line or special token, the predictions tend to be $-1$ for incorrect step and $0$ for neutral and $1$ for correct step.