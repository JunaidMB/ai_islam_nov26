# Rejection Sampling

Involves curating new model completions, filtering them based on a trained reward model & instruction fine tuning the base/original pretrained model only on the top filtered completions. Overall process:

1) **Prompt & reward model selection:**
   Select prompts we wish to train on, simply we can reuse every prompt used in the SFT stage. Must also have a trained reward model

2) **Generate prompt completions**
   For each prompt, generate completions from the model we wish to optimize. These completions must vary, we can have many completions for the same prompt - this can be done by altering model settings such as temperature, max sequence length.

3) **Select top completions with a reward model**
   All completions are ranked by a reward model

4) **Perform SFT on top completions**
   Given the starting checkpoint (for the model) we fine tune on selected completions

# Generating Completions

To generate a set of multiple candidate completions per prompt, define a set of $M$ prompts

$$X = [x_1, x_2, \ldots, x_M]$$

For each prompt $x_i$, generate $N$ completions. Represent as a matrix

$$Y = \begin{bmatrix} y_{11} & y_{12} & \cdots & y_{1N} \\ y_{21} & y_{22} & \cdots & y_{2N} \\ \vdots & \vdots & \ddots & \vdots \\ y_{M1} & y_{M2} & \cdots & y_{MN} \end{bmatrix}$$

Where $y_{ij}$ is the $j$-th completion for Prompt $x_i$

## Score Completions

Next, pass all Prompt, completion pairs through a reward model to get a matrix of rewards

$$R = \begin{bmatrix} r_{11} & r_{12} & \cdots & r_{1N} \\ r_{21} & r_{22} & \cdots & r_{2N} \\ \vdots & \vdots & \ddots & \vdots \\ r_{M1} & r_{M2} & \cdots & r_{MN} \end{bmatrix}$$

Where $r_{ij} = \mathcal{R}(y_{ij} \mid x_i)$

There are many ways to use $R$ to select top completions.

## 1) Top per prompt

Select the response that maximizes reward per prompt (row in $R$). As a score it can be expressed as:

$$S(R) = [\arg\max_j r_{1j}, \arg\max_j r_{2j}, \ldots, \arg\max_j r_{Mj}]$$

$S$ returns a vector of indices that correspond to the column with the maximum reward for each row in $R$. Use the indices to select chosen completions:

$$Y_{\text{chosen}} = [y_{1,S(R)_{1}}, y_{2,S(R)_{2}}, \ldots, y_{M,S(R)_{M}}]$$

## 2) Top Overall pairs

Irrespective of prompt, select the prompt-completion pairs that are best. Suppose we want the $K$ best. This leaves a chance that some prompts are not chosen at all.

Thus can be done by flattening $R$ into a vector:

$$R_{\text{flat}} = [r_{11}, \ldots, r_{MN}]$$

Define a selection function $S_K$ that selects the indices of the $K$ highest values in $R_{\text{flat}}$:

$$S_K(R_{\text{flat}}) = \mathrm{argsort}(R_{\text{flat}})[-K:]$$

$\mathrm{argsort}$ returns indices that sorts $R_{\text{flat}}$ into ascending order, then we take last $K$ indices to obtain $K$ highest values.

To recover the corresponding prompt-completion pair, we map a zero-indexed flattened index $k$ to $(i,j)$ via:

$$i = \left\lfloor \frac{k}{N} \right\rfloor + 1 \quad \text{and} \quad j = (k \bmod N) + 1$$

With the selected completions, we can perform standard instruction fine tuning (SFT) on the current version of the model

Rejection sampling is a simple alternative to RL for preference fine tuning. Generate multiple completions per prompt, score them with a reward model, & instruction fine tune the model on only the top scoring completions.