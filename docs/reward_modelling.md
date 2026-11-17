# Reward Modelling

In RLHF, reward models output a probability of a given input being high quality — chosen answer among a pairwise preference relation.

Let $x$ denote a prompt and $y$ denote a completion.

## Training Reward Models

- A reward model canonically uses the **Bradley-Terry model** of preference.
- A Bradley-Terry model of preference defines that, in a pairwise comparison between 2 items $i$ & $j$, A judge prefers $i$ over $j$:

$$\Pr(i > j) = \frac{p_i}{p_i + p_j} \tag{1}$$

Each item has a probability $p_i > 0$. It is common to reparametrize the Bradley-Terry model with unbounded scores:

$$\Pr(i > j) = \frac{e^{r_i}}{e^{r_i} + e^{r_j}} = \sigma(r_i - r_j) \tag{2}$$

where $p_i = e^{r_i}$ and $\sigma(z) = \frac{1}{1 + e^{-z}}$.

- To train a reward model, a loss function must satisfy the equation in (2).

- In practice, a language model is converted to a model that outputs a scalar score via small linear head that produces a single logit. Given a prompt $x$ and 2 sampled completions $y_1$ and $y_2$. Both completions are scored with a reward model $r_\theta$. The conditional scores are given as $r_\theta(y_i|x)$.

The probability of success for a given reward model in a pairwise comparison is:

$$Pr(y_c > y_r | x) = \frac{\exp[r_\theta(y_1 | x)]}{\sum_{i=1}^{2} \exp[r_\theta(y_i | x)]} \tag{3}$$

- Denote the preferred completion as $y_c$ (chosen) & the rejected completion as $y_r$. $y_c > y_r$ indicates $y_c$ is preferred over $y_r$.
- By maximizing the log likelihood of (3) we get a loss function to train a reward model.

$$\theta^* = \arg\max_\theta Pr(y_c > y_r | x) = \arg\max_\theta \frac{\exp[r_\theta(y_c | x)]}{\exp[r_\theta(y_c | x)] + \exp[r_\theta(y_r | x)]}$$

$$= \arg\max_\theta \frac{1}{1 + \frac{\exp[r_\theta(y_r | x)]}{\exp[r_\theta(y_c | x)]}}$$

$$= \arg\max_\theta \frac{1}{1 + \exp[r_\theta(y_r | x) - r_\theta(y_c | x)]}$$



$$= \arg\max_\theta \sigma(r_\theta(y_c | x) - r_\theta(y_r | x))$$

$$= \arg\min_\theta -\log \sigma[r_\theta(y_c | x) - r_\theta(y_r | x)] \tag{4}$$

Hence the loss function is
$$\mathcal{L}(\theta) = -\log\left(\sigma\left[r_\theta(y_c|x) - r_\theta(y_r|x)\right]\right) \tag{5}$$

or alternatively
$$\mathcal{L}(\theta) = \log\left[1 + \exp\left(r_\theta(y_r|x) - r_\theta(y_c|x)\right)\right] \tag{6}$$

(5) and (6) are equivalent if we let
$$\Delta = r_\theta(y_c|x) - r_\theta(y_r|x)$$
$$\sigma(\Delta) = \frac{1}{1+e^{-\Delta}}$$
$$\Rightarrow -\log\left[\sigma(\Delta)\right] = \log\left[1+\exp\left(r_\theta(y_r|x)-r_\theta(y_c|x)\right)\right]$$

- To implement a reward model, we append a linear head to a language model which performs 2 outcomes - chosen & rejected. At inference time, the model outputs the probability that a piece of text is chosen - in the form of a single logit from the model.