# RLHF and Post Training

## Instruction Fine Tuning (IFT)

- IFT is a method for adapting language models to a desired task distribution. It prepares a model for a format of instructions known as question-answering.

## Chat Templates

- The beginning of the post training process is defining a pattern to format user queries so they are readable by a language model that processes information by a tokenizer
- To prompt a base model, the user inputs a sequence of tokens for the model to continue / complete sequence, e.g.
    
    `<bos_token> The capital of Uzbekistan is`

- The model would generate tokens until it runs out of context window tokens or it generates the end of sequence token.
- All post training stages rely on the above formatting to train the model. The tool that handles the structure of the interaction with the user is called a **chat template**.
- Chat templates will transform a list of user interactions into a sequence that the language model can use to generate new tokens.

# Best Practices of Instruction Tuning

1) High quality data is key to performance. The prompt completions are what the model learns from

2) ~1 Million prompts can be used to create a model capable of excellent post-training

3) The best prompts are those in a similar distribution to downstream tasks of interest

## Implementation

- Post training loss function is often the same as pretraining - next token prediction. Some best practices are:

1) Smaller batch sizes vs pretraining

2) For instruction tuning, prompt tokens are masked out so the model doesn't learn to accurately predict user queries - only responses.

3) Multi-turn Masking: For multi turn masking, there are 2 masking approaches:

   i) Final turn only: Only the tokens in the final assistant turn are included in the loss - all other earlier assistant turns are masked, loss is not computed from them. Long conversations can be decomposed into many training samples

   ii) Mask user turns only: All user turns are masked but every assistant turn is included in the loss - the intermediate assistant replies are trained on directly.