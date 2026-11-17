## Character Training

* A subset of post-training designed around crafting traits within a model to tweak the personality, values and/or manner of its response to the content
* Model weights are changed to craft a stable, base persona for a given model
* Character training with fine-tuning in personality-specific data is shown to be more robust than prompting
* Anthropic uses techniques like Constitutional AI (CAI) and synthetic data generation to train model character

## Persona Vectors

* Persona vectors modify the inner workings of a model at inference time
* The idea of persona vectors is based on how personality traits correspond to the same class of linear directions in a model's residual stream
* Looking at Figure 17.1, we are computing an activation vector, called $v_c$ to a persona vector. We remove this vector from a model's layers and it should alter its behaviour
* The extraction pipeline generates a representation that compares responses near to and far from a given characteristic (E.g. Evil) - called contrastive activation analysis

- Given trait name & description (e.g. Sycophancy: excessive agreeableness & flattery), use a frontier LLM to generate System prompt pairs, one to elicit the trait & to suppress it.
- A target model generates responses under both conditions and residual stream activations* are extracted from each response, averaged over response tokens at a chosen layer $l$. The layer is chosen empirically.
- The persona vector is the difference in means between the 2 groups
$$V_L = \frac{1}{|S^+|} \sum_{i \in S^+} a_l^{(i)} - \frac{1}{|S^-|} \sum_{j \in S^-} a_l^{(j)}$$
where
- $S^+$ is the set of trait-exhibiting responses
- $S^-$ is the set of trait-Suppressing responses
- $a_l^{(i)}$ is the mean residual stream activation at layer $l$ for sample $i$
- The layer that produces the strongest steering effect is selected as the final Persona vector.
- Once extracted, a Persona vector steers behaviour via an additive intervention applied at every token generation step

$$h_l \leftarrow h_l + \alpha V_l$$
Where
- $h_l$ is the residual stream activation
- $\alpha$ is a steering coefficient
- $\alpha > 0$ amplifies the trait, $\alpha < 0$ suppresses it

### What is a residual stream activation?

As tokens flow through a transformer (at inference time). It passes sequentially through blocks/layers. Many of these layers use Residual connections - adding output of a block back to the main stream.

*   If the representation at layer $l$ is $h_l$, a given block computes a function & adds its output to itself to form the next layer's representation.

$$h_{l+1} = h_l + \text{Block}(h_l)$$

*   Every time we give the model a different input, the hidden states in every layer take a unique trajectory.
*   A transformer is a stack of layers (Layer$_1, \ldots$, Layer$_N$). When data enters Layer-$l$, its current representation in the stream is $h_l$.
*   $h_l$ is the vector flowing through the stream at whichever layer we're looking at.

### Worked Example:

Suppose we are at a specific layer ($l=10$), our input is a matrix of $S \times D$ shape. Let $d=4$ & $s=3$ (our prompt is 3 tokens long). Let the resulting activation matrix $h_L$, at layer 10 and a given sample 1, be

1. 
$$\mathbf{h}_l^{(1)} = \begin{pmatrix} .5 & .2 & -.1 & .4 \\ .7 & .3 & -.2 & .5 \\ .6 & .1 & 0 & .3 \end{pmatrix}$$

Each row is a 4 dimensional vector representing the state of the residual stream at that token position. This is for a given input.

2. Generate multiple response samples for trait positive $S^+$ and trait negative $S^-$ conditions. For each sample prompt $i$, collapse its $s \times d$ matrix down to $1 \times d$ vector by taking the mean across the token rows.

i. Sample 1: Trait positive

The $3 \times 4$ matrix for Sample Prompt 1 averages to $a^{(1)}_{l} = [1.1, 0.5, -0.1, 0.3]$

ii. Sample 2: Trait Positive

Another response averages out to $a^{(2)}_{l} = [1.3, 0.7, 0.1, 0.5]$

iii. Sample 3: Trait negative

$a^{(3)}_{l} = [0.3, 0.1, 0.4, 0]$

3. Compute Persona vector $\mathbf{v}_l$

i. Mean of Group $S^+$
$$ \frac{1}{|S^+|} \sum_{i \in S^+} a^{(i)}_{l} = \frac{[1.1, 0.5, -0.1, 0.3] + [1.3, 0.7, 0.1, 0.5]}{2} $$
$$ = [1.2, 0.6, 0.0, 0.4] $$

ii. Mean of Group $S^-$
$$ \frac{1}{|S^-|} \sum_{j \in S^-} a^{(j)}_{l} = [0.3, 0.1, 0.4, 0] $$

iii. Persona vector, $\mathbf{v}_l$
$$ \mathbf{v}_{l} = [1.2, 0.6, 0.0, 0.4] - [0.3, 0.1, 0.4, 0] $$
$$ = [0.9, 0.5, -0.4, 0.4] $$

This is a static steering vector $\mathbf{v}_l$ of $1 \times d$ pointing in the Positive trait direction at layer 10.

4) Inference Time Intervention

i) At a later time, a user inputs a new question "What is 2+2?". We process this input at layer 10, we generate a new residual stream matrix $h_l$ (let s=2)

$$h_l = \begin{bmatrix} 0.2 & 0.4 & 0.1 & -0.1 \\ 0.3 & 0.5 & 0.2 & 0.0 \end{bmatrix}$$

ii) With a steering coefficient $\alpha = 2$, add a Persona vector $\mathbf{v}_l$

$$\alpha \times \mathbf{v}_l = 0.2 [0.9, 0.5, -0.4, 0.4] = [1.8, 1.0, -0.8, 0.8]$$

we add to the rows of $h_l$. $\alpha \times \mathbf{v}_{l}$ is broadcasted

$$h_{new} = \begin{bmatrix} 0.2 & 0.6 & 0.1 & -0.1 \\ 0.3 & 0.5 & 0.2 & 0.0 \end{bmatrix} + \begin{bmatrix} 1.8 & 1.0 & -0.8 & 0.8 \\ 1.8 & 1.0 & -0.8 & 0.8 \end{bmatrix}$$

$$= \begin{bmatrix} 2.0 & 1.4 & -0.7 & 0.7 \\ 2.1 & 1.5 & -0.6 & 0.8 \end{bmatrix}$$

The modified $h_{new}$ matrix flows into layer $(l+1)=11$. Since every token's representation has been shifted along the Positive trait direction.

# Assistant Axis

- To extend the persona vector work, extract Persona vectors for over 275 character archetypes - spanning roles like teacher, engineer, chef, Philosopher etc. Using the same persona vector extraction
- Next, PCA is run over the collection to map a persona space. PC1 is said to be the degree to which the model resembles the default assistant; the Assistant Persona vector - this is the model's default identity as a helpful assistant

- Define a contrast vector as the Assistant axis
  $\mathbf{v}_{axis} = \bar{h}_{assistant} - \bar{h}_{roles}$
  
where
$\bar{h}_{assistant}$: The mean residual stream activation across default Assistant responses
$\bar{h}_{roles}$: The mean residual stream activation across all role playing archetypes
$\mathbf{v}_{axis}$ is on the spectrum from role playing character to default helpful assistant

In long running sensitive conversations, a model's internal activations can drift from the Assistant axis. To stop this drift, we introduce activation capping a runtime intervention rule applied to the Post FFN activation at a given layer

$$\mathbf{h}' = \mathbf{h} - \mathbf{v} \min(\langle \mathbf{h}, \mathbf{v} \rangle - \tau, 0)$$

where
$\mathbf{h}$ is the post MLP residual stream at a given layer
$\mathbf{v}$ is the unit normalised Assistant Axis direction

To understand $\mathbf{h}'$, define $p = \langle \mathbf{h}, \mathbf{v} \rangle$. This is a scalar dot Product measuring how 'Assistant like' the current activation $\mathbf{h}$ is and $\tau$ is a safely calibrated threshold. We can have 2 scenarios

i) $p \ge \tau$: implies the model is operating safely within the assistant region. The term $\langle \mathbf{h}, \mathbf{v}\rangle - \tau$ is $0$ or positive. Set the min function evaluator to $0$, so $h' = h$, the activations are identical.

ii) $p < \tau$: The model is in an unsafe zone.

### Worked Example 2:

Create 2 distinct Prompt datasets:
1) $S_{Assistant}$: Standard user queries given to the default Model. The responses are standard model completions to the queries.
2) $S_{roles}$: System prompted character archetypes across 275+ roles. E.g. System: 'You are a chaotic trickster'. System: 'You are a 19th century Philosopher'. The responses are written in character.

Suppose we are evaluating layer $l=10$, and we look at a single prompt/ sample $i$, let this have $s=3$ tokens and the forward pass produces a matrix, $s \times d$, $s=3, d=4$.

$$h^{(l)}_{10} = \begin{bmatrix} 1.6 & 1.2 & 0.1 & 0.9 \\ 1.4 & 1.8 & 0.3 & 0.7 \\ 1.5 & 1.0 & 0.2 & 0.8 \end{bmatrix}$$

pool this over rows (Column/vertical means)

$$a^{(i)}_{10} = [1.5, 1.0, 0.2, 0.8]$$

Compute $\bar{h}_{assistant}$ averages across the assistant dataset. Repeat the above for all $N$ samples in default assistant dataset. The above was only for a single sample.

$\bar{h}_{assistant} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{a}^{(i)} \in \mathbb{R}^{1 \times d}$

For $\bar{h}_{roles}$, do the same. 

For every role playing prompt sample $j$, across all archetypes, extract the matrix $h^{(j)}$. Take average over rows (column means) to get $\mathbf{a}^{(j)}$. Then average across all role samples $|S_{roles}| = M$.

Sample $j=1$ (Teacher role): $\mathbf{a}^{(1)} = [1.8, .4, .5, .2]$
Sample $j=2$ (Trickster role): $\mathbf{a}^{(2)} = [-.2, 0, .7, 0]$
... for all rows

$\bar{h}_{roles} = \frac{1}{M} \sum_{j=1}^{M} a^{(j)} = [.5, .2, .6, .1] \in \mathbb{R}^{1 \times d}$

Extract axis

$\mathbf{v}_{axis} = \bar{h}_{assistant} - \bar{h}_{roles} \in \mathbb{R}^{1 \times d}$

$\mathbf{v}_{axis} = [1.0, .8, -.4, -.7]$

Next, normalise

$\mathbf{v} = \frac{\mathbf{v}_{axis}}{||\mathbf{v}_{axis}||_2} = [0.66, 0.53, -0.26, 0.46]$

2) Simulate Persona Drift across Conversation Turns

Imagine a multi turn conversation where a user asks for emotional support & gradually nudges the model away from its Objective Assistant Persona. We monitor residual stream activation $h$ at layer 10 across a single turn. Set safety threshold $\tau=0.5$

Turn 1: Let this be a 2 token averag, s=2.
$$h^{(1)} = \begin{bmatrix} -0.1 & -0.2 & 0.4 & -0.1 \\ 0.0 & -0.13 & 0.5 & -0.2 \end{bmatrix}$$

compute projection $p = \langle h^{(1)}, \mathbf{v} \rangle$. $h^{(1)}, \mathbf{v} \in \mathbb{R}^{2 \times 1}$

$$\begin{bmatrix} -0.1 & -0.2 & 0.4 & -0.1 \\ 0.0 & -0.13 & 0.5 & -0.2 \end{bmatrix} \begin{bmatrix} 0.66 \\ 0.53 \\ -0.26 \\ 0.46 \end{bmatrix} = \begin{bmatrix} -0.3238 \\ -0.3833 \end{bmatrix} = \mathbf{p}$$

Both components of $\mathbf{p}$ have drifted below $\tau=0.5$. So we cap updates for both tokens.

$$\\min (p_k - T, 0)$$

$$\\min (p_1 - T, 0) = \min (-0.3238 - 0.5, 0) = -0.8238$$
$$\\min (p_2 - T, 0) = \min (-0.3833 - 0.5, 0) = -0.8833$$

Now, compute the updated $h' = h - \mathbf{v} \cdot \min(\mathbf{p} - \tau, 0)$. $\mathbf{v}$ is broadcasted.

$$\begin{bmatrix} h'^{(1)} \\ h'^{(2)} \end{bmatrix} = \begin{bmatrix} h^{(1)} + 0.8238 \times v \\ h^{(2)} + 0.8833 \times v \end{bmatrix} = \begin{bmatrix} 0.44 & 0.24 & 0.18 & 0.28 \\ 0.58 & 0.67 & 0.27 & 0.21 \end{bmatrix}$$

Checking both $\langle h'^{(1)}, v \rangle$ and $\langle h'^{(2)}, v \rangle$ confirms both token positions are within threshold boundary.

## Persona Subnetworks

*   A pre-trained LLM can behave like many different people. Sans fine-tuning. These personas a latent inside the model as a sparse subnetwork: a specific subset of weights that are most associated when the model is asked to act that way.
*   To accentuate the persona, we suppress the weights that aren't part of the desired persona.

## Conceptual Walkthrough

* Inside the LLM, almost every computation is of the form
$$\mathbf{y} = W\mathbf{x} + \mathbf{b}$$

where

$W \in \mathbb{R}^{m \times n}$ is a learned weight matrix
$\mathbf{x} \in \mathbb{R}^n$ is an input
$\mathbf{y} \in \mathbb{R}^m$ is output

*  For each persona $p$, build a binary mask $M_p \in \{0,1\}^{m \times n}$. A 1 tells us to retain the weight & 0 means zero it out.
*  At inference, compute
$$\mathbf{y} = (W \odot M_p)\mathbf{x} + \mathbf{b}$$
*   Switching personas is switching masks

* Masks are built from computed Importance Scores which are numbers per weight that tells us how much a given weight matters, for a given persona, given some calibration data.

* Moreover, some Personas are opposites (Introvert vs Extrovert). We want their masks to be opposing, namely the weights retained for one persona should be different to the weights retained for its inverse. A technique called contrastive pruning is used to enforce this.

### Methods

Let $\mathcal{P}$ be a set of Personas. For each $p \in \mathcal{P}$, we have a small calibration set (with hundreds of instances)

$$ \mathcal{D}_p = \{ (\mathbf{x}_i^p, \mathbf{y}_i^p) \}_{i=1}^{N_p} $$

where
* $\mathbf{x}_i^p$ is a prompt, $\mathbf{y}_i^p$ is a persona consistent response

* We want a mask that maximizes persona alignment while remaining sparse.

$$ \max_{M^p} \mathbb{E}_{(x, y) \sim D_p} [\log P_{M^{P}}(y | x)] $$

$$ s.t. ||M^p||_0 \le (1-\rho)d $$

where
* $\rho \in (0,1)$ is a sparsity ratio
* $d$ is total number of maskable weights (elements in weight matrices)
* Let $\mathcal{M}^p = f(\theta \odot M^p)$ denote the mashed model

### Masked constraint

* $||M^p||_0$ is the count of nonzero entries of $M^p$. The constraint $||M^p||_0 \le (1-\rho)d$ means that at most, $(1-\rho)d$ of the $d$ maskable weights may be kept.

* If $\rho=0 \implies (1-\rho)d = d$, all $d$ weights should be kept no persona at all.
* If $\rho=1 \implies (1-\rho)d = 0$, every output is set to 0
* Empirically, Ye et al. paper suggests a range for $\rho$ between $0.4 - 0.6$. 

At $\rho=0.6 \implies (1-\rho)d = 0.4d$, 60% of weights are kept. At $\rho=0.6 \implies (1-\rho)d = 0.4d$, 40% of weights are kept

# Activation Statistics

* A transformer layer is a big block consisting of a self attention block(s) and a FFN block(s). Each of these contain several linear modules. For a model like llama model these are - $W_q, W_k, W_v, W_o$ (attention) and $W_{gate}, W_{up}, W_{down}$ (FFN).
* Each of these $W$ has a form $W \in \mathbb{R}^{m \times n}$ and each has its own input activation vector/matrix feeding it.

* The input to a layer, $l$, is 

 $$H^{(l)}(\mathbf{x}) \in \mathbb{R}^{T \times d}$$

where 
$T$ is the prompt length
$d$ is the hidden dimension

* Row $t$ of $H^{(l)}(\mathbf{x})$ is the hidden state of token $t$ at depth $l$. $\mathbf{h}^{(l)}(\mathbf{x})$ denotes a single row of $H^{(l)}(\mathbf{x})$ - a single token's hidden state.

* **Example 1:** For a given model, let $d=4096$ and $L=32$ layers
Suppose we feed a token sequence $x = \text{'I love quiet evenings!'}$ with $T=5$. After embedding & $14$ transformer blocks, the full layer-15 hidden state is a matrix

$$
H^{(15)}(x) = \begin{bmatrix} h_1^{(15)}(x) \\ h_2^{(15)}(x) \\ h_3^{(15)}(x) \\ h_4^{(15)}(x) \\ h_5^{(15)}(x) \end{bmatrix} \in \mathbb{R}^{5 \times 4096}
$$

row 3, hidden state for 'quiet' is a single 4096 vector $h_3^{(15)}(x) = [.12, -1.84, \ldots, 2.71] \in \mathbb{R}^{4096}$. 

Component $j$ of $h_3^{(15)}(x)$ tells us how strongly the $j$-th learned feature is active on token 3 at depth 15.

- We want per channel average activation under persona p for $j \in \{1, \dots, d\}$ components.

$$A^{(l)}_{p}[j] = \mathbb{E}_{(x,y) \sim \mathcal{D}_P}[\left|\mathbf{h}^{(l)}_{j}(\mathbf{x})\right|], \quad j=1,\dots,d$$

- The recipe to obtain this is
1. Take the calibration set corresponding to persona $p$ - $\mathcal{D}_p$. This is prompt-response pairs that illustrate the persona.
2. Run a prompt $\mathbf{x}_{p}$ through the model $M$, record layer $l$'s hidden state matrix 

$$H^{(l)}(\mathbf{x}^p_{i}) \in \mathbb{R}^{5 \times 4096}$$

one row per token
3. Vertically stack (rowwise) the matrices from all prompts in $\mathcal{D}_P$ into a single matrix of shape $\sum T_i \times n$

4. Take absolute value of all entries, then take column wise mean (across rows so number of columns is preserved)

As illustration, take 3 calibration prompts for a persona $p$ with lengths $T_1, T_2, T_3$. The hidden state is (at layer $l$)

$$ \begin{bmatrix} \left|H^{(l)}(\mathbf{x}^{p}_{1})\right| \\ \left|H^{(l)}(\mathbf{x}^{p}_{2})\right| \\ \left|H^{(l)}(\mathbf{x}^{p}_{3})\right| \end{bmatrix} \in \mathbb{R}^{(T_1+T_2+T_3) \times n} $$

(Mean over columns after taking absolute values)

$$ A_p^{(l)} = \begin{bmatrix} \overline{h_1} \\ \overline{h_2} \\ \vdots \\ \overline{h_n} \end{bmatrix} \in \mathbb{R}^n $$

-$A_p^{(l)}$ can be seen as telling us - how loud is each input channel, when Persona p is active.

# Magnitude-based Importance

* For a linear layer with $W \in \mathbb{R}^{m \times n}$, we assign an importance to every weight

$$s^{p}_{i,j} = |w_{i,j}| \cdot A_{p}^{(i)}[j] $$

* A weight is important when
	1. Its magnitude is large
	2. The channel (component $j$) is 'loud', derived from $A_{p}^{(i)}[j]$

* Keep the Top-k important weights per row
$$m^p_{i,p} = \begin{cases} 1 & \text{if } s^{p}_{i,j} \in \text{TopK}(\{s^{p}_{ik} \}_{k=1}^{n}) \\ 0 & \text{o.w.} \end{cases}$$

(If importance is in top k, for a row, this is not masked)

where
$K = \lfloor L(1-\rho) \rfloor$
$\rho$ is a sparsity ratio

# Performing Inference

* At test time, for each layer, we have a Mask that we perform elementwise multiplication

$$\mathbf{y} = (W \odot M^P)\mathbf{x} + \mathbf{b}$$

* We can use a soft gating to blend persona and a base model using a hyperparameter $\gamma \in [0,1]$

$$G = M^P + \gamma(1-M^P)$$
 
when $\delta=0$ we have pure persona and base model if $\delta=1$

* Every weight matrix in every layer gets its own persona mask. E.g. For Llama we have $L=7$ linear Modules: Attention: $W_{(Q)}^{(l)}, W_{(K)}^{(l)}, W_{(V)}^{(l)}, W_{(O)}^{(l)}$ and FFN: $W_{(gate)}^{(l)}, W_{(up)}^{(l)}, W_{(down)}^{(l)}$. Each $W^{(l)}_{*}$ gets its own mask 

$M^{p, (i)}_{*} \in \{0,1\}^{m_{*} \times n_{*}}$ matched to its weight matrix

- Only the token embedding & LM head is excluded from masking. This is because they operate on vocabulary indices, pruning them would blur the input/output interface for every persona.

- The entire set/collection of masks is called the global mask.

$$\mathcal{M}^P = \{ M^{p, (i)}_{*}\}_{{l=1,...,L}; * \in \{q, k, v, o, \text{gate, up, down}\}} $$
