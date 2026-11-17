
## Reinforcement Learning Review 

Reinforcement learning (RL) is a framework for learning from interaction. An agent observes the current state of an environment, performs an action, receives a reward and observes a new state. In the long run, an agent will discover if its behaviour was useful after several actions.

An agent takes actions, receives rewards in a loop. A policy is the rules an agent uses to choose actions from states. With neural networks in reinforcement learning, the policy is represented by model parameters. Model training changes the parameters such that chosen actions that lead to higher long-term reward become more likely.

## Application to language generation

Token generation is a sequential decision process. The state at time $t$ is the prompt given by the user and all tokens the model has generated so far in the current response. The action corresponds to the next sampled token. The new state is the old state plus the sampled token. When the model finishes a response, a reward function scores the response.

RL with language models typically uses sparse rewards - rewards given at the end of a rollout that covers if the set of actions as a whole were good. Sparse rewards create a credit assignment problem. If the answer is wrong, which token caused the error?Policy gradient methods update probabilities of all sampled actions in proportion to the outcome

## Updating LLMs with reward

A Language Model (LM) produces a probability distribution over a discrete vocabulary at each step. During a rollout, the model samples a token from the probability distribution.

The response (a sequence of sampled tokens) receives a reward. We want to increase the probability of sampled tokens that are associated with a higher reward and decrease the probability of sampled tokens that are associated with a lower reward. A policy gradient method estimates how altering policy parameters would affect expected reward.

For LMs, the policy parameters are the weights of the LLM. The policy is the model's probability distribution over the next tokens. When selecting a token from the probability distribution, we implement sampling, not taking the argmax. This is required for exploration. With LMs, a full response is a trajectory: a sequence of states & sampled actions. 

The reward is attached to the trajectory and an update to the policy (model parameters) changes token probabilities along that trajectory.

# Rewards & Gradient Updates

We require a way to score generated responses - typically they are scalars.

For instance, if a model is generating reasoning results for a mathematical problem, we can assign scalar rewards for obtaining

i. the correct answer

ii. having a correct working step

iii. the correct output format for the answer. 

We want our model to generate responses that will maximise the reward scores. These reward scores must be normalised via the Z-score normalisation.

A simplified policy gradient update can be written as

$$\theta_{t+1} = \theta_t + \alpha \sum_{k} \nabla_{\theta} \log[\pi_\theta(a_k|s_k)] R_t \quad (1)$$

where
- $\theta$ are the model parameters
- $\alpha$ is the learning rate
- $s_k$ is the state at token step $k$ (Model input at $k$)
- $a_k$ is the token sampled at step $k$
- $\pi_\theta$ is the policy model
- The gradient of the log probability tells us how to change the parameters so the sampled action is more likely.

The Multiplication by $R$ controls the direction of the update. If $R$ is high, the sampled actions are reinforced, if $R$ is low relative to a baseline, the sampled actions are discouraged.

Using raw rewards directly is a blunt tool. A raw reward doesn't indicate how good a response is relative to what was expected for that specific prompt.

Due to the bluntness of raw rewards, we can use advantage values. An advantage measures how much better or worse an outcome is compared to a baseline. If a response is better than expected, the advantage is positive. Else if a response is worse than expected, the advantage is negative.

$$A_k = R_k - \text{baseline}(S_k) \quad (2)$$

# Group Relative Policy Optimisation (GRPO)

The GRPO approach to updating a policy involves sampling a group of model completions for the same prompt or query, $q$. Each completion receives a reward. All the rewards in that group are normalized. 

If a rollout is better than the group average, its advantage is **positive**. If a rollout is worse than the group average, its advantage is **negative**.

Since rewards are a blunt tool, we use advantages

$$\hat{A}_i = \frac{r_i - \mu_r}{\sigma_r + \epsilon}$$

where

$$\mu_r = \frac{1}{G} \sum_{i=1}^{G} r_i$$

$$\sigma_r = \sqrt{\frac{1}{G} \sum_{i=1}^{G} (r_i - \mu_r)^2}$$

When sampling a group of completions, we want their responses to be as diverse as possible. This is controlled by the temperature parameter of the language model.

The advantage $\hat{A}_i$ is sequence-level and uses the mean ($\mu_r$) and standard deviation ($\sigma_r$) of the rewards within that specific group:


## GRPO Objective Function

The GRPO loss function for all tokens is

$$\mathcal{L}_{GRPO}(\theta) = \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} -\left[ \min\left(\rho_t(\theta)\hat{A}_i, \text{clip}\left(\rho_t(\theta), 1-\epsilon, 1+\epsilon\right)\hat{A}_i\right) - \beta \mathbb{D}_{KL}(\pi_\theta \| \pi_{\theta_{ref}})\right]$$

where

$$\rho_t(\theta) = \frac{\pi_\theta(o_{i, t}|q, o_{i, <t})}{\pi_{\theta_{old}}(o_{i, t}|q, o_{i, <t})}$$

$$\mathbb{D}_{KL} = \exp\left(\log \pi_\theta(o_{i, t}|q, o_{i, <t}) - \log \pi_{\theta_{ref}}(o_{i, t}|q, o_{i, <t})\right) - \left[\log \pi_\theta(o_{i, t}|q, o_{i, <t}) - \log \pi_{\theta_{ref}}(o_{i, t}|q, o_{i, <t})\right] - 1$$
$\pi_{\theta_{old}}$: The state of trainable policy $\pi_\theta$ before current trajectory rollouts are sampled. Outputs are generated using $\pi_{\theta_{old}}$. We perform multiple gradient updates on $\pi_\theta$ using the same batch of sampled trajectories. $\pi_\theta$ starts to diverge from $\pi_{\theta_{old}}$. It is updated to match $\pi_\theta$ every time new completions are sampled from the environment.

$\pi_{ref}$: The base model checkpoint. It remains fixed across the entire RL training run.

## Worked Example – Update for a single group

1.  Given a query $q$, generate $G$ responses to the query $\pi_{old}(q) \to \{y_i\}_{i=1}^G$.

2.  Compute Advantage values for each of the $G$ responses.
$$\hat{A}_i = \frac{r_i - \mu_r}{\sigma_r + \epsilon}$$
3.  To compute GRPO Loss for a particular group, we compute the following for all tokens:
$$\mathcal{L}_{GRPO} = -\frac{1}{G} \sum_{i=1}^{G} \sum_{t=1}^{|o_i|} \min[p_t(\theta)\hat{A}_i, \text{clip}(p_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_i]  - \beta \mathbb{D}_{KL}(\pi_{\theta} || \pi_{\theta_{ref}})$$


where 
$$P_t(\theta) = \frac{\pi_{\theta}(o_t | o_{i,<t}, q)}{\pi_{\theta_{old}}(o_t | o_{i, <t}, q)}$$

Let $G=2$ and the query be $q=$ "What is the capital of France?"

Let the responses being considered be
$o_1$: "It is London"
$o_2$: "The capital of France is Paris"

Let the advantage values associated with responses be

$\hat{A}_1 = -1$, $\hat{A}_2 = 1$.

Set $\epsilon = 0.1$.

Further remarks before computing the ratio:
i. At first, when $\pi_{\theta_{old}} = \pi_{\theta}$, the ratio will be 1. To avoid being trivial, we will assume $\pi_{\Theta_{old}}$ has been trained for some iterations so it yields a different softmax probability

ii. Actual softmax values will be simulated for illustration

iii. We omit computing the KL divergence for simplicity

Consider the simplified ratio sum for response $1, o_1$

$$\sum_{t=1}^{|o_1|} p_t(\theta) = \sum_{t=1}^{|o_1|} p_t(\theta) = p_1(\theta) + p_2(\theta) + p_3(\theta)$$

The tokens are $o_{11}=\text{It}, o_{12}=\text{is}, o_{13}=\text{London}.$

The ratio is

$p_1(\theta) = \frac{\pi_{\theta}(o_{11}|q)}{\pi_{\theta_{\text{old}}}(o_{11}|q)} = \frac{0.14}{0.15} = 0.933$

$p_2(\theta) = \frac{\pi_{\theta}(o_{12}|q, o_{11})}{\pi_{\theta_{\text{old}}}(o_{12}|q, o_{11})} = \frac{0.25}{0.28} = 0.893$

$p_3(\theta) = \frac{\pi_{\theta}(o_{13}|q, o_{11}, o_{12})}{\pi_{\theta_{\text{old}}}(o_{13}|q, o_{11}, o_{12})} = \frac{0.27}{0.36} = 0.750$

Substitute in minimum. This Advantage for this response is $\hat{A}_t = \hat{A}_1 = -1$

$\min [p_t(\theta) \hat{A}_t, \text{clip}(p_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t]$
$\min [p_1(\theta)(-1), \text{clip}(p_1(\theta), 1-0.10, 1+0.10)(-1)]=\min [\frac{1}{3}(-1), \text{clip}(0.933, 0.90, 1.10)(-1)] = -0.933$$\min [p_2(\theta)(-1), \text{clip}(p_2(\theta), 1-0.10, 1 + 0.10)(-1)]= \min [\frac{1}{4}(-1), \text{clip}(0.893, 0.90, 1.10)(-1)] = -0.90$
$\min [p_3(\theta)(-1), \text{clip}(p_3(\theta), 1 - 0.10, 1 + 0.10)] = \min [\frac{1}{12}(-1), \text{clip}(0.750, 0.90, 1.10)] = -0.750$

Average across tokens

$$\frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min [p_t(\theta) \hat{A}_t, \text{clip}(p_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t] = \frac{1}{3} [-0.933 + -0.90 + -0.90] = -0.911$$

Next, we do the same for

$o_2 = \text{"The capital of France is Paris"}$. The tokens are denoted

$o_{21} = \text{The}, o_{22} = \text{capital}, o_{23} = \text{of}, o_{24} = \text{France}, o_{25} = \text{is}, o_{26} = \text{Paris}$

The ratios (truncated) are

$p_1(\theta) = \frac{\pi_{\theta}(o_{26}|q, o_{21}, \cdots, o_{25})}{\pi_{old}(o_{26}|q, o_{21}, \cdots, o_{25})}$

$\cdots$

$p_6(\theta) = \frac{\pi_{\theta}(o_{26}|q, o_{21}, \cdots, o_{25})}{\pi_{old}(o_{26}|q, o_{21}, \cdots, o_{25})}$

Skipping to the sum across tokens. The Advantage for this response is $\hat{A}_t = \hat{A}_2 = 1$

$$\frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min [p_t(\theta) \hat{A}_t, \text{clip}(p_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t] = \frac{1}{6}$$

Once we have processed the tokens in every response, we average across every response in the group (we omit the KL divergence for simplicity). Since we have 2 responses in the group, we sum and divide by 2.

$$\frac{1}{2} [-0.911 + \frac{1}{6}] = -0.375 = -\frac{3}{8}$$


## Rolling up to Batch Level

The defined GRPO loss was for all group responses for a single prompt/query. However, in a batch we have many prompts. To compute a loss over a batch, we compute the average again.

$$\mathcal{L}_{Batch} = \frac{1}{B} \sum_{b=1}^{B} \mathcal{L}_{GRPO}$$

## Gradient with clip

It is instructive to see how the clipping function will affect the gradient computed for the loss.

$$\rho_t = \frac{\pi_\theta(a|s_t)}{\pi_{\theta_{old}}(a|s_t)}$$

$$f_t = \min(\rho_t \hat{A}_t, \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon) \hat{A}_t)$$

#### Rewrite $\rho_t$
$$\rho_t = \exp(\log \pi_\theta(a|s_t) - \log \pi_{\theta_{old}}(a|s_t))$$

#### Take derivative $\frac{\partial \rho_t}{\partial \log \pi_\theta}$

Let $u = \log \pi_\theta(a|s_t) - \log \pi_{\theta_{old}}(a|s_t)$, then $\rho_t = e^u$

$$\frac{\partial \rho_t}{\partial \log \pi_\theta} = \frac{\partial \rho_t}{\partial u} \times \frac{\partial u}{\partial \log \pi_\theta}$$

$$= e^u \times 1 = e^u = \rho_t$$

$$\frac{\partial \rho_t}{\partial \log \pi_\theta} = \rho_t$$

**How $f_t$ changes with sign of $\hat{A}_t$ & value of $\rho_t$?**

**Case 1: $\hat{A}_t > 0$**

$$\text{clip}(\rho_t, 1-\epsilon, 1+\epsilon) \hat{A}_t = \begin{cases} (1-\epsilon) \hat{A}_t & \text{if } \rho_t < 1-\epsilon \\ \rho_t \hat{A}_t & \text{if } 1-\epsilon \le \rho_t \le 1+\epsilon \\ (1+\epsilon) \hat{A}_t & \text{if } \rho_t > 1+\epsilon \end{cases}$$

corrected:
set $\epsilon=0.10, \hat{A}_t=-1$. New probabilities

$$\pi_\theta(o_{11}|q) = 0.14, \ \pi_{\theta_{old}}(o_{11}|q) = 0.15, \rho_1(\theta) = 0.933$$ 

$$\pi_\theta(o_{12}|q) = 0.25,\ \pi_{\theta_{old}}(o_{12}|q) = 0.28, \rho_2(\theta) = 0.893$$ 

$$\pi_\theta(o_{13}|q) = 0.27,\ \pi_{\theta_{old}}(o_{13}|q) = 0.36, \rho_3(\theta) = 0.750$$

Next we compute,

$$\min[P_t(\theta)\hat{A}_t, \text{clip}(P_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t]$$

1) $$\min[-0.933, \text{clip}(0.933, 0.90, 1.10)(-1)]$$
$$\min[-0.933, 0.933 \times -1] = -0.933$$


2) $$\min[-0.893, \text{clip}(0.893, 0.90, 1.10)(-1)]$$
$$\min[-0.893, -0.90] = -0.90$$


3) $$\min[-0.750, \text{clip}(0.750, 0.90, 1.10)(-1)]$$
$$\min[-0.750, -0.90] = -0.90$$

$$\frac{1}{|o_{i}|} \sum_{t=1}^{|o_{i}|} \min[\rho_t(\theta)\hat{A}_t, \text{clip}(\rho_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t]$$

$$= \frac{1}{3}[-0.933 + -0.90 + -0.90] = -0.911$$

$$\frac{1}{2}[-0.91 + 0.16] = -0.375 = -\frac{3}{8}$$


when taking minimum

$$min(\rho_t \hat{A}_t, \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\hat{A}_t)$$

**Subcase 1:** $\rho_t \ge 1+\epsilon$

$$min(\rho_t \hat{A}_t, \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\hat{A}_t) = (1+\epsilon)\hat{A}_t$$

$$\frac{\partial f_{t}}{\partial \log \pi_\theta} = \frac{\partial}{\partial \log \pi_\theta} [(1+\epsilon)\hat{A}_t] = 0 \quad (\text{Doesn't depend on } \theta)$$

**Subcase 2:** $\rho_t \le 1+\epsilon$

$$min(\rho_t \hat{A}_t, \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\hat{A}_t) = \rho_t \hat{A}_t$$

$$\frac{\partial f_{t}}{\partial \log \pi_\theta} = \hat{A}_t \frac{\partial \rho_t}{\partial \log \pi_\theta} = \hat{A}_t \frac{\partial}{\partial \log \pi_\theta} [\exp[\log \pi_\theta(a_t|s_t) - \log \pi_{\theta_{old}}(a_t|s_t)]] = \hat{A}_t \rho_t$$

**Case 2:** $\hat{A}_t < 0$

**Subcase 1:** $\rho_t < 1-\epsilon$

$$min(\rho_t \hat{A}_t, \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\hat{A}_t) = (1-\epsilon)\hat{A}_t$$

$$\frac{\partial f_{t}}{\partial \log \pi_\theta} = \frac{\partial}{\partial \log \pi_\theta} [(1-\epsilon)\hat{A}_t] = 0 \quad (\text{Doesn't depend on } \theta)$$

**Subcase 2:** $\rho_t \ge 1-\epsilon$

$$min(\rho_t \hat{A}_t, \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\hat{A}_t) = \rho_t \hat{A}_t$$

$$\frac{\partial f_{t}}{\partial \log \pi_\theta} = \frac{\partial}{\partial \log \pi_\theta} [\rho_t \hat{A}_t] = \rho_t \hat{A}_t$$
**Final derivative**
$$\frac{\partial f_t}{\partial \log \pi_{\theta}} = \begin{cases} 0 & \hat{A}_t > 0 \land \rho_t > 1+\epsilon \\ 0 & \hat{A}_t < 0 \land \rho_t < 1-\epsilon \\ \rho_t \hat{A}_t & \text{otherwise} \end{cases}$$

Applied to usecase. $\epsilon=0.1, \hat{A}_t=-1 < 0$

Token 1: 
$$\rho_1 = 0.933, 1-\epsilon = 1-0.1 = 0.9$$

Since $\rho_1 > 1-\epsilon = 0.9$, the derivative is $\rho_1 \hat{A}_t$

Token 2: $$\rho_2 = 0.893, 1-\epsilon = 0.9$$
Since $\rho_2 < 1-\epsilon = 0.9$, the derivative is $0$

Token 3: $$\rho_3 = 0.750, 1-\epsilon = 0.9$$
Since $\rho_3 < 1-\epsilon = 0.9$, the derivative is $0$

From this we see that the derivative influences the effect the $\rho_{t}$ on the gradient computation.

# Limitations of GRPO

Empirically, a phenomena has been observed where response length kept increasing as RL training length increases. This is related to the normalisation term $\frac{1}{|o_{i}|}$ which leads to a behaviour in minimising negative advantage by producing long length responses. If a response is bad and hence gets a negative advantage, if it is longer (has many tokens) then when we divide it by the number of tokens, $|o_{i}|$, we can scale this negative advantage down.

One amendment we can make is to equalise token level contributions.

Problem:
$\frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_{i}|} \sum_{t=1}^{|o_{i}|}$

Proposed Solution:

DAPO: $\frac{1}{\sum_{c=1}^{G} |o_{i}|} \sum_{i=1}^{G} \sum_{t=1}^{|o_{i}|}$ 


Dr. GPO: $\frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_{i}|} \sum_{t=1}^{|o_{i}|}$

Another big disadvantage of GRPO is that every token in an output gets the same advantage.

# Group Sequence Policy Optimisation

A reward is almost always given to the entire completed response. GSPO calculates importance ratios, applies clipping and weight gradients at full sequence level. Every token in a given response is treated equally.

Suppose a 10 token response, $y$.

GSPO sums log probabilities across all tokens to get Sequence level ratio $\rho^{seq}$ between models.

$$\rho^{seq} = \frac{\pi_{\theta}(y|x)}{\pi_{\theta_{old}}(y|x)}$$

we compute a clipped ratio like before with GRPO
$$\text{clip}(\rho^{seq}, 1-\eta, 1+\eta) = u$$

GSPO takes the single Sequence level multiplier, $u$, and multiplies it by the group normalised advantage and applies it uniformly across all tokens.

Note:
$$\pi_{\theta}(y|x) = \prod_{t=1}^{T} \pi_{\theta}(y_t|x, y_{<t}) \quad (\text{for our 10 token case})$$

$$\pi_{\theta}(y|x) = \pi_{\theta}(y_{10}|x, y_{<10}) \dots \pi_{\theta}(y_1|x)$$

So in GRPO we compute a separate, independent ratio $\rho_t$ for every individual token $t$ based on its specific conditional probability. In GSPO we compute a single aggregate ratio for the entire response, $\rho^{seq}$, and this is applied uniformly to each token in that sequence.

## Reinforcement Learning with Verifiable Rewards (RLVR)

In practice we use a training approach like RLVR. In RLVR, the reward is produced by a deterministic checker vs a learned reward model. For instance, given a math problem, a checker can compare the final answer vs known answer.

A model can generate many candidate solutions to a prompt and receive reliable signal without human labels for every response. RLVR applies when correctness is objectively verifiable. For more open-ended problems we require learned reward models, preference data, human evaluation or other alignment signals. 

Practically speaking; we use verifiers where they are reliable and use preference signals where the target behaviour is not mechanically checkable.

A RLVR training run should be monitored with more than average reward. Average reward can rise while other behaviours degrade. E.g response length may increase but correctness may not. Useful monitoring metrics measure all aspects of desirable responses such as mean reward, pass rate, response length, KL divergence from reference policy, invalid format rate, duplicate output rate and reward variance within groups.

## Practical Considerations

When performing model improvements via GRPO, we require 3 copies of the language model:
1. A frozen copy of the reference language model
2. The policy language model which is updated via GRPO
3. The policy language model which is used to generate rollouts for a given input $q_i$. Typically served via vLLM.

![[model_copies.png]]

At every step, we have the following process

![[grpo_models_step.jpeg]]