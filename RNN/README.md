\# RNN (Recurrent Neural Network) — Quick Revision Notes



\## 1. What is RNN?



A \*\*Recurrent Neural Network (RNN)\*\* is a neural network used for sequential data like text, speech and time series.



It reads data one step at a time and remembers useful information from previous steps using a \*\*hidden state\*\*.



\*\*Example:\*\* In the sentence "I love this movie", the meaning of "movie" depends partly on the words before it.



\## 2. How does RNN work?



At every step, RNN:



1\. Takes the current input.

2\. Uses information from the previous step.

3\. Updates its hidden state.

4\. Produces an output if needed.



\### Main formula



`h\_t = tanh(W\_xh x\_t + W\_hh h\_(t-1) + b)`



Don't memorise it blindly. Understand these parts:



\* `x\_t` → current input

\* `h\_(t-1)` → memory from the previous step

\* `h\_t` → updated memory

\* `W` → weights learned during training

\* `b` → bias

\* `tanh` → activation function that keeps the value between -1 and 1



\*\*Remember:\*\* Current input + Previous memory = Updated memory.



\## 3. Important RNN terms



\### Hidden State



The hidden state is the RNN's \*\*short-term memory\*\*. It stores useful information from the sequence read so far.



Example: After reading "I love", the hidden state may carry information that helps process the next word, "movie".



\### Word Embedding



A word embedding converts a word's ID into a dense vector of numbers that a neural network can learn from.



Example: Word ID `25` might become a vector like `\[0.2, -0.4, 0.7, ...]`.



Words with related meanings may learn similar vectors, but this is not guaranteed.



\### BPTT (Backpropagation Through Time)



RNN learns by comparing predictions with actual answers and updating its weights.



BPTT sends the error backward through the sequence to update the weights used at different time steps.



\## 4. Vanishing Gradient Problem



During training, the error signal can become extremely small as it travels backward through many time steps.



\*\*Result:\*\* A basic RNN struggles to learn relationships between words or events that are far apart in a sequence.



Example: It may struggle to connect a word at the beginning of a long sentence with its meaning at the end.



\## 5. What is LSTM?



\*\*Long Short-Term Memory (LSTM)\*\* is a special type of RNN designed to learn long-term dependencies better.



It uses:



\* A \*\*cell state\*\* to carry long-term information.

\* Three gates to control which information to forget, store and output.



LSTM helps reduce the vanishing gradient problem, but it does not eliminate every training problem.



\### Cell State



The cell state is the LSTM's \*\*long-term memory\*\*. It carries information across time steps and can preserve useful information for longer.



\### What is a Gate?



A gate controls how much information passes through. Its values are between 0 and 1.



\* `0` → block the information.

\* `1` → allow the information through.

\* `0.5` → allow some information through.



\### Three LSTM Gates



\*\*1. Forget Gate\*\*

Decides how much old information to keep or forget.



Example: Forget an old detail that is no longer useful.



\*\*2. Input Gate\*\*

Decides how much new information should be stored in the cell state.



Example: Store an important new detail.



\*\*3. Output Gate\*\*

Decides how much information from the cell state is used to form the current hidden state.



Example: Use relevant stored information to make the current prediction.



\*\*Easy memory trick:\*\*

Forget = Remove | Input = Store | Output = Use



\## 6. RNN vs LSTM vs GRU



| Model | Main idea                 |

| ----- | ------------------------- |

| RNN   | Simple sequence memory    |

| LSTM  | Cell state + 3 gates      |

| GRU   | 2 gates: update and reset |



LSTM and GRU are often better choices when long-term dependencies matter.



\## 7. Where is RNN used?



\* Sentiment analysis

\* Text and speech processing

\* Time-series forecasting

\* Sequence prediction



For example, in sentiment analysis, a model reads a review and predicts whether its sentiment is positive or negative.



\## 8. Important Interview Questions



\*\*Q1. What is RNN?\*\*

RNN is a neural network designed for sequential data that uses a hidden state to carry information between steps.



\*\*Q2. Why is RNN useful for text?\*\*

Because word order matters, and RNN processes words as a sequence.



\*\*Q3. What is a hidden state?\*\*

It is the information carried from previous time steps to help process the current input.



\*\*Q4. What is word embedding?\*\*

It converts word IDs into dense numerical vectors that the model can learn from.



\*\*Q5. What is BPTT?\*\*

It is the training method that propagates errors backward through the sequence.



\*\*Q6. What is the vanishing gradient problem?\*\*

Gradients become very small during training, making long-term relationships difficult to learn.



\*\*Q7. Why is LSTM better than a basic RNN for long sequences?\*\*

Its cell state and gates help preserve useful information for longer.



\*\*Q8. What are the three LSTM gates?\*\*

Forget gate, input gate and output gate.



\*\*Q9. What is the difference between hidden state and cell state?\*\*

The hidden state carries information used at the current step; the cell state carries longer-term memory in an LSTM.



\*\*Q10. What is the difference between RNN and ANN?\*\*

A basic ANN has no built-in recurrent memory, while an RNN passes a hidden state across sequence steps.



\---



\## 9. One-Minute Revision



\* RNN → Sequential data

\* Hidden state → Short-term memory

\* Word embedding → Word ID to dense vector

\* BPTT → Backpropagation through sequence

\* Vanishing gradient → Difficulty learning long-term relationships

\* LSTM cell state → Long-term memory

\* Forget gate → Remove or retain old information

\* Input gate → Control new information

\* Output gate → Control current hidden state

\* GRU → Update and reset gates



\*\*Revision goal:\*\* 15–20 minutes for theory and formulas, then revise the practical and interview questions.



