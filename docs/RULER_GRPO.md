### RULER (Relative Universal LLM-Elicited Rewards)

Traditionally, training an agent via RL involves handcrafting reward functions - passing a model response & awarding scalar rewards based on model response features. RULER replaces this handcrafting with a prompt that uses an LLM as a judge to score & rank model responses.

RULER serves as an LLM as a judge mechanism used to score & grade trajectories. The primary design philosophy is **relative evaluation**: instead of scoring an AI agent's trajectory in isolation, it bundles multiple trajectories that share the same goal and gives them to a single high capability judge model (E.g. o3-model) and forces the judge to evaluate them alongside one another based on a specific rubric.

RULER executes the following pipeline for every single training step:
1. **Rollout Generation:** For a single prompt, the model being trained creates a group of $G$ trajectories. A trajectory is a complete multi-turn sequence of model generation turns $y_t$ and environment/tool responses.
$$\tau = (y_1, e_1, y_2, e_2, \dots, y_T)$$

2. **Context Deduplication:** RULER groups identical trajectories, such as the system prompt & user query. Only the suffixes are unique.

3. **Multi-Trajectory Evaluation:** The shared prompt prefix, the unique suffixes, and a plain-English rubric are sent to a LLM judge model.

4. **Scoring Generation:** Judge evaluates trajectories, outputting a score between 0 & 1 and a reasoning rationale.

The plain English rubric in RULER is a natural language description of what the user wishes to reward. RULER gives the rubric to an LLM which maps them to a score.

RULER evaluates trajectories against structural rules:
1. Goal achievement
2. **Efficiency:** Trajectories that solve the query in fewer turns are rewarded more.
3. **Tool adherence:** Did the LLM agent interact with its environment well? Did it call tools appropriately?

Next, these RULER trajectory scores are used by GRPO to update model weights. This can be seen in the example below.

## Worked Example

Here is the complete, end-to-end workflow of the RULER-guided GRPO training loop, combining the token-optimized data structures, the precise formatting of the judge's prompt, and the resulting downstream mathematical optimization. The steps below include both RULER and GRPO.

### Step 1: Generation of the Trajectory Group

The policy model (our experimental coding assistant) is prompted with a string-reversal task. It generates a group of four distinct trajectories ($G=4$) to allow for relative comparison.

- **Trajectory 1 (Agent A):**
```
- `[User]`: "Write a Python function to reverse a string." 
- `[Assistant]`: "Here is your function: `def rev(s): return s[::-1]`" 
- `[User]`: "Great, now make it handle `None` inputs without crashing." 
- `[Assistant]`: "Sure! `def rev(s): return s[::-1] if s else ''`"
```

- **Trajectory 2 (Agent B):**
```
- `[User]`: "Write a Python function to reverse a string."
- `[Assistant]`: "Let me search my database... No wait, I will write it. `def reverse(string): return ''.join(reversed(string))`"
- `[User]`: "Great, now make it handle `None` inputs without crashing."
- `[Assistant]`: "Error: `NoneType` object is not iterable. Oh wait, I didn't fix it. Let me try again."
```

- **Trajectory 3 (Agent C):**
```
- `[User]`: "Write a Python function to reverse a string."
- `[Assistant]`: "To reverse a string in Python, you can utilize slicing. Here is the code: `def reverse_str(text): return text[::-1]`"
- `[User]`: "Great, now make it handle `None` inputs without crashing."
- `[Assistant]`: "To handle a `None` input, we add a conditional check. Updated code: `def reverse_str(text): return text[::-1] if text is not None else ''`"
```

- **Trajectory 4 (Agent D):**
```
- `[User]`: "Write a Python function to reverse a string."
- `[Assistant]`: "Here is your function: `def rev(s): return s[::-1]`"
- `[User]`: "Great, now make it handle `None` inputs without crashing."
- `[Assistant]`: "I can help with that. _[Provides a verbose tutorial on None types, but forgets to output the code block]_"

```
### Step 2: Ruler Token Optimization (Prefix Stripping)

Ruler identifies that user turns turns are identical across all inputs. It extracts this shared history into a single `<context>` block, leaving only the unique assistant actions inside numbered `<trajectory>` tags.

This optimized payload becomes the **User Message** sent to the judge:

```
<context>
[
  {"role": "user", "content": "Write a Python function to reverse a string."},
  {"role": "user", "content": "Great, now make it handle None inputs without crashing."}
]
</context>

**Trajectories:**

<trajectory id="1">
[
  {"role": "assistant", "content": "Here is your function: def rev(s): return s[::-1]"},
  {"role": "assistant", "content": "Sure! def rev(s): return s[::-1] if s else ''"}
]
</trajectory>

<trajectory id="2">
[
  {"role": "assistant", "content": "Let me search my database... No wait..."},
  {"role": "assistant", "content": "Error: NoneType object is not iterable..."}
]
</trajectory>

<trajectory id="3">
[
  {"role": "assistant", "content": "To reverse a string in Python, you can utilize slicing..."},
  {"role": "assistant", "content": "To handle a None input, we add a conditional check..."}
]
</trajectory>

<trajectory id="4">
[
  {"role": "assistant", "content": "Here is your function: def rev(s): return s[::-1]"},
  {"role": "assistant", "content": "I can help with that. [Provides long tutorial, misses code text]"}
]
</trajectory>
```

### Step 3: Relative Evaluation via Structured Judge Prompt

Ruler wraps the context payload above with instructions and guidelines. This full interaction is sent to a high-capability judge model (e.g., `o3`) to evaluate the trajectories side-by-side.

#### The Complete Prompt Sent to the Judge Model

- **System Message (`judge_prompt`):**
    > All of the trajectories below have been given the same goal. Your job is to consider each of them and give them a score between 0 and 1. Take into consideration your best judgment of the agent's goal.
    > 
    > Grading standards:
    > 
    > - A trajectory that achieves its goal should always get a significantly higher score than a trajectory that does not achieve its goal.
    >     
    > - A trajectory that achieves its goal more efficiently (eg. by avoiding unproductive detours) should get a higher score than a trajectory that achieves its goal less efficiently.
    >     
    > - If one trajectory is only slightly better than another, the difference in scores should be small. If it is significantly better, the difference in scores should be large.
    >     
    > - You may give some partial credit for a trajectory that makes progress towards its goal but does not complete it.

- **User Message (`user_text`):**
	This contains the actual data payload, structurally separated by XML tags so the model doesn't get confused about where one conversation ends and another begins:

```xml
<context> 
	[The shared conversation turns] 
</context> Trajectories: 

<trajectory id="1">[Agent A's unique turns </trajectory> 
<trajectory id="2">[Agent B's unique turns] </trajectory> ...
```

#### The Judge's Structured JSON Output (`Response`)

Because the judge reviews the trajectories in a single **joint-evaluation session,** it maps them directly against each other to provide a precise, calibrated distribution of scores:


```json
{
  "scores": [
    {
      "trajectory_id": "1",
      "explanation": "Excellent, highly efficient, and correctly handles the requested edge case.",
      "score": 1.0
    },
    {
      "trajectory_id": "2",
      "explanation": "Failed entirely. Wandered off-task on turn 1 and crashed on turn 2.",
      "score": 0.1
    },
    {
      "trajectory_id": "3",
      "explanation": "Achieved the goal completely and handled the edge case safely, but used a lot of verbose conversational filler.",
      "score": 0.8
    },
    {
      "trajectory_id": "4",
      "explanation": "Partial success. Handled turn 1 perfectly, but failed to provide the updated code snippet on turn 2, opting instead for a conversational explanation.",
      "score": 0.5
    }
  ]
}
```

### Step 4: Downstream GRPO Statistics & Advantages

The `art_ruler` wrapper assigns these values back to the trajectory objects ($R = [1.0, 0.1, 0.8, 0.5]$). GRPO treats this group as an isolated statistical distribution to compute how well each trajectory performed relative to its peers.

#### 1. Group Calculations

- **Group Mean ($\mu$):**

$$\mu = \frac{1.0 + 0.1 + 0.8 + 0.5}{4} = 0.60$$

- **Group Standard Deviation ($\sigma$):**

$$\text{Variance} = \frac{(1.0-0.6)^2 + (0.1-0.6)^2 + (0.8-0.6)^2 + (0.5-0.6)^2}{4} = 0.115$$

$$\sigma = \sqrt{0.115} \approx 0.339$$

#### 2. Conversion to Advantage Scores ($A_i = \frac{R_i - \mu}{\sigma}$)

GRPO standardizes the rewards to establish clear directional mathematical signals:

- **Agent A Advantage ($A_1$):** $\frac{1.0 - 0.60}{0.339} = \mathbf{+1.18}$ _(Far above average)_
- **Agent B Advantage ($A_2$):** $\frac{0.1 - 0.60}{0.339} = \mathbf{-1.47}$ _(Far below average)_
- **Agent C Advantage ($A_3$):** $\frac{0.8 - 0.60}{0.339} = \mathbf{+0.59}$ _(Moderately above average)_
- **Agent D Advantage ($A_4$):** $\frac{0.5 - 0.60}{0.339} = \mathbf{-0.29}$ _(Slightly below average)_

### Step 5: GRPO Policy Weight Update

During the backward optimization pass, the log-probabilities of the generated tokens are multiplied by their respective Advantage scores.

- **Positive Updates ($A_1, A_3$):** The weights responsible for Agent A and Agent C's generations are reinforced. The model learns to favor valid code syntax and defensive programming checks. Agent A receives double the reinforcement of Agent C, penalizing the model for generating unnecessary chat filler.
- **Negative Updates ($A_2, A_4$):** The weights responsible for Agent B and Agent D's tokens are suppressed. The model actively updates its parameters to avoid runtime exceptions, internal monologue looping, and forgetting to output critical code snippets when instructed.