
## Pre Training

Pretraining is the most compute-intensive phase of building a frontier model. Dividing it into a multi-stage curriculum is necessary to balance the mathematical constraints of the Transformer architecture with the sheer cost of compute.

Initially we give the model large volumes of text data (upwards of 30 Trillion tokens for GLM5), of varying types, and have it learn language structure via next token prediction. The loss function used here would be cross entropy because the next token represents a classification task over the vocabulary of tokens available. 
$$ \mathcal{L}_{CE} = \frac{1}{T} \sum_{t=1}^{T} log\ p_{\theta}(x_t | x_{<t})$$


Since we have all of the text anyway, we can use teacher forcing and parallelise the process of learning. E.g. Take the sequence "[sos] The capital of France is Paris. [eos]".

We can split this up to get each token's cross entropy loss in parallel (input token -> target token):
"[sos]" -> "The"
"[sos] The" -> "capital"
...
"[sos] The capital of France is" -> "Paris"

We take the cross entropy loss for each sequence and token and average it to get the loss that will be used to update  model parameters. Since we have the above tokens, already we don't have to wait for them to be decoded like in inference. 

Multiple stages allow for a **length curriculum**. The model spends the first 80% of its pretraining budget learning facts, syntax, and world knowledge using a short, cheap context window (e.g., 4,096 tokens). Only in the final stages do engineers scale the context window up to 32k or 128k tokens, adjusting the positional embeddings (like RoPE) to handle long-range dependencies efficiently.

If you mix all data types—low-quality web scrapes, dense textbook mathematics, raw source code, and academic papers—into a single massive data bucket, the training signals conflict. A **data mixture curriculum** structures the learning process. The model first builds a broad, stable "cognitive foundation" using general multilingual web text. In this data mixture, repeated tokens and data has been found empirically important. So we may include duplicates in the corpus and space it out.

Once this foundation is secure, the data mixture is intentionally shifted in a later stage (often called "mid-training") to heavily favour code, logic, and STEM data. Because the model already understands basic language, it can rapidly assimilate these complex structural rules without breaking its underlying fluency.

## Mid Training

Aims at injecting knowledge on tasks which are more aligned with a use case. It strengthens capabilities on targeted higher-quality data aligned with tasks. For instance, if we want a model that is good at code generation, now we will give it high quality, long context coding datasets at this stage still using next token prediction.

Part of extending models on longer context/ sequences involves extending positional encoding methods like RoPE. If we train short context models on long context inputs without any adaptation, then the model will struggle. One such adaptation is called YaRN (Yet another RoPE extensioN). Length is scaled up via a curriculum, we start from 4k tokens and move to 32k, 128k, 1 million tokens and so on.

## Post Training

Post-training transitions the model from a raw text-predictor into a controllable, helpful assistant. With modern frontier models acting as "hybrid" reasoning engines (capable of both instant chat and long-form thinking), a single-stage fine-tuning process is mathematically impossible.

Modern models must natively support two different operational modes - thinking and non-thinking. If we attempt to train a model on both behaviours simultaneously using a flat dataset, the capabilities cannibalise each other. The model becomes confused. For a simple query like _"What is the capital of France?"_, it might spin into a 400-token existential reasoning loop before answering. For a complex math problem, it might skip the thinking entirely and output a wrong answer.

Post-training requires **isolated optimisation followed by fusion**.
- **Stage 1 (Isolation):** Engineers train separate, specialised "expert" variants or use rule-based reinforcement learning (RL) to develop the deep reasoning trace capability in a vacuum.
- **Stage 2 (Fusion):** They use distillation techniques to merge these conversational and deep-thinking behaviours into a single weight checkpoint.
- **Stage 3 (Routing):** They apply final alignment to teach the model _when_ to trigger each mode based on the user's prompt or specified token budgets.

Moreover, splitting post-training into separate stages allows engineers to leverage the right feedback loops at the right time:

- **The Verifiable RL Stage:** The model is subjected to intense, automated Reinforcement Learning (like RLAIF or rejection sampling) using hard math and coding environments. Because the feedback is objective, the model's pure logical reasoning ceiling can be pushed to its absolute limit without human bottlenecks.
- **The Preference Alignment Stage:** Once the model is highly intelligent, it undergoes a separate alignment stage (using DPO or PPO) focused on human preferences, safety guardrails, tone, and formatting constraints.

## Case Study: Qwen 3

The Qwen3 models are pre-trained through a three-stage process: 

1. General Stage (S1): At the first pre-training stage, all Qwen3 models are trained on over 30 trillion tokens using a sequence length of 4,096 tokens. At this stage, the models have been fully pre-trained on language proficiency and general world knowledge, with training data covering 119 languages and dialects.

2. Reasoning Stage (S2): To further improve the reasoning ability, we optimize the pre-training corpus of this stage by increasing the proportion of STEM, coding, reasoning, and synthetic data. The models are further pre-trained with about 5T higher-quality tokens at a sequence length of 4,096 tokens. We also accelerate the learning rate decay during this stage. 

3. Long Context Stage: In the final pre-training stage, we collect high-quality long context corpora to extend the context length of Qwen3 models. All models are pre-trained on hundreds of billions of tokens with a sequence length of 32,768 tokens. The long context corpus includes 75% of text between 16,384 to 32,768 tokens in length, and 25% of text between 4,096 to 16,384 in length. Following Qwen2.5 (Yang et al., 2024b), we increase the base frequency of RoPE from 10,000 to 1,000,000 using the ABF technique (Xiong et al., 2023). Meanwhile, we introduce YARN (Peng et al., 2023) and Dual Chunk Attention (DCA, An et al., 2024) to achieve a four-fold increase in sequence length capacity during inference.

The post-training pipeline of Qwen3 is strategically designed with two core objectives: 
1. Thinking Control: This involves the integration of two distinct modes, namely the “non-thinking” and “thinking” modes, providing users with the flexibility to choose whether the model should engage in reasoning or not, and to control the depth of thinking by specifying a token budget for the thinking process. 
2. Strong-to-Weak Distillation: This aims to streamline and optimise the post-training process for lightweight models. By leveraging the knowledge from large-scale models, we substantially reduce both the computational costs and the development efforts required for building smaller scale models.


![[qwen3_posttraining_pipeline.png]]
## Distillation 

Involves using a larger (teacher) model giving responses to a set of inputs which create a dataset. This dataset will be used to train smaller (student) models via SFT. Hard distillation trains a student model on the tokens generated by the teacher LLM. Soft distillation trains the student LLM on logit values over the vocabulary at each token generation step. In other words, for a given input, we want the student to have its own softmax probability distribution (for each token position) to match the teacher's as closely as possible.

Off-Policy Distillation involves having the dataset of inputs ahead of time, obtaining teacher model completions and using that dataset downstream. On-Policy involves generating responses during training, a student model generates a completion to a prompt then a teacher model also generates a completion. The student updates its parameters by comparing what it produced with what the teacher produced.

The On/Off-Policy and Hard/Soft approaches are taken together, they can be seen as axes or combinations. We can have Off-Policy Models distilled via Hard Distillation, Off-Policy Models distilled via Soft Distillation, On-Policy Models distilled via Hard Distillation and On-Policy Models distilled via Soft Distillation. 


With hard distillation, we create a dataset for training the student model. In Rashka's example, 12,000 Math Problems for training & 500 math problems & solutions for testing. The training problems are given to an existing reasoning model, Deepseek-R1 (teacher model) and collect its response as training targets.

![](./images/distillation_dataset.png)

We compare the student model's generated tokens with the teacher generated tokens - the teacher response becomes the target sequence.

Take a model input, $x_i$, and give it to teacher model $f_i$ and get a sequence of generated responses. We also give this input to the student model $f_j$, yielding a sequence of generated responses. The comparison of these solutions allows us to compute training signal.

For a student model, it is fed the prompt combined with the teacher generated completion as the target sequence and the loss is computed via next token cross entropy. The paradigm used for fine tuning: a student model is supervised fine tuning (prompt masked, response tokens have loss computed).

**Evaluation** in this example involves taking a single instance from the test dataset which is a (prompt, completion) pair. The trained student model generates a response. A rules based verifier script is used to compare the student's answer against the ground truth completion. If it matches, we score it a 1 else 0. E.g. If a reference completion is '14' and the student model is '14' we score it 1. We compute overall accuracy over the dataset.





