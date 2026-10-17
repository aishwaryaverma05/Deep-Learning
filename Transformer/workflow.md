Transformer ka complete process — ek example se
Hum ek English-to-Hindi translation example lenge:
INPUT (English)
I love machine learning

EXPECTED OUTPUT (Hindi)
मुझे मशीन लर्निंग पसंद है

Transformer ka kaam hai English sentence ko samajhna aur uske context ke according Hindi sentence generate karna.
Ismein do main parts hote hain:
Encoder
English sentence ko context ke saath represent karta hai.
Encoder output → Decoder ko information

Decoder
Encoder ki information aur pehle generate hue words se Hindi sentence banata hai.

Ab ek-ek step samjho.
Step 1: Tokenization — sentence ko tokens mein todna
Input sentence:
I love machine learning
Tokenizer ise tokens mein convert karega. Simplified example mein maan lo har word ek token hai:
I

love

machine

learning


Actual tokenizer subwords bhi bana sakta hai, jaise kisi word ke multiple tokens.

Har token ko ek integer ID milegi, for example:
I → 15, love → 82, machine → 214, learning → 97
Ye numbers sirf example hain. Real model ki vocabulary mein IDs alag hongi.
Important: Token IDs ka numerical size word ki importance ya meaning nahi batata.
Step 2: Embedding — token IDs ko vectors mein convert karna
Model directly words ke meaning par calculations nahi kar sakta, isliye har token ID ko ek learned vector mein convert kiya jata hai.
For example, simplified 3-dimensional vectors:
Token	Embedding (illustrative)
I	[0.2, 0.5, 0.1]
love	[0.8, 0.3, 0.6]
machine	[0.4, 0.9, 0.2]
learning	[0.6, 0.7, 0.8]
Real Transformers mein vectors usually kaafi bade hote hain, jaise 512 ya 768 dimensions, model par depend karta hai.
Embedding token ki learned representation hai. Abhi tak model ne sentence ke har word ko doosre words ke context mein fully interpret nahi kiya hai.
Step 3: Positional Encoding — word ka order batana
Ek problem: attention sentence ke tokens ko ek saath process karta hai. Toh model ko kaise pata chalega ki I love learning aur learning love I ka order alag hai?
Isliye embeddings mein positional information add hoti hai.
Token embedding
+
Position information


Input representation


Original Transformer mein sinusoidal positional encoding use hua tha; doosre models learned positional embeddings ya other positional methods use karte hain.
Ab har token ke paas do tarah ki information hai:
- Token ka content.
- Sentence mein uski position.
Step 4: Encoder Self-Attention — words ek doosre ko samajhte hain
Ye Transformer ka sabse important concept hai.
Sentence hai:
I love machine learning.

learning word ko samajhne ke liye model ko machine se relation samajhna helpful hai. Isi tarah love ka relation I ke saath hai.
Self-Attention har token ko input sentence ke doosre tokens ki information use karke context-aware representation banane deta hai.
4.1 Query, Key aur Value (Q, K, V)
Har input vector se teen vectors bante hain:
- Query (Q): Ye token kis tarah ki information dhoondh raha hai?
- Key (K): Ye token kis tarah ki information provide karta hai?
- Value (V): Agar is token par attention diya, toh kaunsi information use hogi?
Ye vectors learned weight matrices se bante hain:
\[
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
\]

Yahan \(X\) positional information add hone ke baad input representations ka matrix hai.
Ek simple intuition:
Token	Query ka intuitive sawal
I	Mere action se kaun related hai?
love	Kaun action kar raha hai aur kis cheez se related hai?
machine	Kya main learning se related hoon?
learning	Kaunse words mere meaning ko clarify karte hain?
Ye actual literal questions nahi hain. Model learned vectors aur attention scores se relationships calculate karta hai.
4.2 Attention scores calculate karna
Har Query ko har Key ke saath compare kiya jata hai using dot product:
\[
S=\frac{QK^T}{\sqrt{d_k}}
\]

- \(QK^T\): har token-pair ka compatibility score.
- \(d_k\): Key vector ki dimension.
- \(\sqrt{d_k}\) se divide karne par scores ka scale control hota hai.
Example ke liye, maan lo learning ke scores kuch aise aaye:
Token jise learning attend karta hai	Illustrative score
I	0.2
love	0.4
machine	2.0
learning	1.0
Yahan machine ka score high hai. Iska matlab is example mein learning ki representation banate waqt machine ki information ko zyada weight mil sakta hai.
Ye actual trained model ke scores nahi hain — sirf concept samjhane ke liye hain.
4.3 Softmax — scores ko attention weights banana
Softmax scores ko probabilities jaisi normalized weights mein convert karta hai. Upar ke illustrative scores par approximate weights honge:
Token	Attention weight
I	0.10
love	0.12
machine	0.60
learning	0.18
Total weight approximately \(1\) hai.
4.4 Weighted sum of Values
Ab model corresponding Value vectors ko in weights ke according combine karta hai:
\[
\text{Attention}(Q,K,V)
=\text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

Isse har token ka naya, context-aware representation milta hai.
Dhyan do: learning ki representation ab sirf uske apne embedding par depend nahi karti; doosre relevant tokens ki information bhi usmein aa sakti hai.
Step 5: Multi-Head Attention — multiple perspectives
Ek attention head relationships ka ek learned pattern capture kar sakta hai. Transformer mein multiple heads parallel kaam karte hain.
Same input representations

Head 1
Ek type ka relationship


Head 2
Doosra relationship


Head 3
Ek aur pattern


Head 4
Different learned pattern




Concatenate heads → Linear projection


Ye heads grammatical relationships, nearby words ya doosre useful patterns learn kar sakte hain, lekin kaunsa head kya seekhega, woh guaranteed ya manually assigned nahi hota.
Step 6: Residual Connection + Layer Normalization
Attention ke baad Transformer seedha next layer par nahi jaata. Original architecture mein har sublayer ke around residual connection aur Layer Normalization hoti hai.
Simplified conceptual form:
\[
Y=\operatorname{LayerNorm}(X+\operatorname{Sublayer}(X))
\]

- \(X\): sublayer ka original input.
- Sublayer(X): attention ya feed-forward network ka output.
- + X: original information ko aage carry karne mein help karta hai.
- LayerNorm: activations ko normalize karke training ko stable banane mein help karti hai.
Dropout bhi original architecture mein use hua tha. Exact ordering implementation ke hisaab se vary kar sakti hai.
Step 7: Feed-Forward Network (FFN)
Attention tokens ke beech information mix karta hai. FFN phir har token ki representation ko independently transform karta hai.
Original Transformer ke example mein:
\[
512\rightarrow2048\rightarrow512
\]

Activation function ReLU use hota hai:
\[
\operatorname{FFN}(x)
=\operatorname{ReLU}(xW_1+b_1)W_2+b_2
\]

Yahan same FFN weights har token position par reuse hote hain.
Phir residual connection + LayerNorm hoti hai. Ye poora block encoder layer complete karta hai.
Step 8: Encoder ki multiple layers
Original Transformer mein 6 encoder layers stack ki gayi thi. Har layer mein:
1. Multi-Head Self-Attention
2. Residual Connection + LayerNorm
3. Feed-Forward Network
4. Residual Connection + LayerNorm
Token embeddings + positional encoding
Encoder Layer 1
Self-Attention → FFN

Encoder Layers 2–5
Same type of blocks, learned parameters

Encoder Layer 6
Final contextual representations


Ab encoder ke output mein English sentence ke tokens ki context-aware representations hain. In outputs ko decoder cross-attention mein Keys aur Values ki tarah use karega.
Yaad rakho: 6 layers original paper ki configuration hai, har modern Transformer mein exactly 6 layers hona zaroori nahi.

Ab Decoder ki baari
Encoder English sentence ko process kar chuka hai. Ab Decoder ko Hindi sentence generate karna hai:
मुझे मशीन लर्निंग पसंद है
Decoder ko ye sentence pehle se nahi diya jaata during inference. Woh output token by token banata hai.
Step 9: Decoder input — <SOS> token
Decoder generation start karne ke liye special token leta hai:
<SOS> = Start of Sequence
Training ke waqt target sentence ko shift karke decoder ko input diya jata hai.
Example:
Decoder input	Expected target
<SOS>	मुझे
<SOS> मुझे	मशीन
<SOS> मुझे मशीन	लर्निंग
<SOS> मुझे मशीन लर्निंग	पसंद
<SOS> मुझे मशीन लर्निंग पसंद	है / <EOS>
Ye simplified example hai. Real tokenization mein words ke beech ka split aur sentence-ending convention model ke hisaab se alag ho sakta hai.
Training mein target sequence known hoti hai. Inference mein decoder ko apne pehle predicted tokens diye jaate hain.
Step 10: Decoder embeddings + positional encoding
Decoder ke input tokens bhi:
1. Token IDs mein convert hote hain.
2. Embeddings mein convert hote hain.
3. Positional information add hoti hai.
Ab decoder ko pata hai ki usne ab tak kya generate kiya hai aur tokens ka order kya hai.
Step 11: Masked Self-Attention — future words ko hide karna
Maan lo training ke waqt decoder input hai:
<SOS> मुझे मशीन लर्निंग
Jab decoder मशीन ki prediction kar raha hai, toh woh future target token लर्निंग ko dekh nahi sakta. Warna training mein answer leak ho sakta hai.
Causal mask future positions ko block karta hai.
Causal mask ka simplified view
Rows = current token; columns = token positions jinhe attend kiya ja sakta hai. ✓ = allowed, ✕ = blocked.
	1	2	3	4
Position 1	✓	✕	✕	✕
Position 2	✓	✓	✕	✕
Position 3	✓	✓	✓	✕
Position 4	✓	✓	✓	✓

Attention scores mein blocked positions ko Softmax se pehle \(-\infty\) jaisa value diya jata hai. Softmax ke baad unki attention probability zero ho jaati hai.
Formula:
\[
\operatorname{MaskedAttention}(Q,K,V)
=\operatorname{Softmax}\left(
\frac{QK^T}{\sqrt{d_k}}+M
\right)V
\]

Yahan \(M\) causal mask hai.
Important difference: Encoder ke self-attention mein future tokens ko block karne wala causal mask nahi hota. Decoder ke masked self-attention mein hota hai.
Step 12: Cross-Attention — decoder encoder se information leta hai
Ab sabse important connection samjho.
Decoder ke paas abhi tak generated Hindi tokens hain. Lekin usko English input sentence ki information bhi chahiye, taaki translation sahi bane.
Cross-attention ye connection provide karta hai.
Encoder output
English sentence ki contextual information

Keys (K)

Values (V)



Cross-Attention
Query (Q) decoder ke current representations se aati hai


Decoder ko relevant English information milti hai

Self-attention aur cross-attention ka difference:
Feature	Self-Attention	Cross-Attention
Q kahan se?	Same sequence	Decoder
K aur V kahan se?	Same sequence	Encoder output
Purpose	Apni sequence ke tokens ka context	Input sequence se relevant information lena
Cross-attention bhi attention scores calculate karta hai, Softmax apply karta hai aur Values ka weighted sum banata hai:
\[
\operatorname{CrossAttention}(Q,K,V)
=\operatorname{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

Yahan encoder ki representations se relevant information decoder tak pahunchti hai. Phir decoder ke block mein FFN aur residual connections + LayerNorm aate hain.
Step 13: Decoder ki multiple layers
Original Transformer decoder mein 6 decoder layers hoti hain. Har layer mein teen main sublayers hote hain:
1. Masked Multi-Head Self-Attention
2. Encoder–Decoder Cross-Attention
3. Feed-Forward Network
Har sublayer ke saath residual connection aur LayerNorm bhi hoti hai.
1. Masked Self-Attention
Ab tak generated tokens ko dekho

2. Cross-Attention
Encoder se relevant information lo

3. Feed-Forward Network
Har position ki representation transform karo

Har sublayer ke around residual connection + LayerNorm hoti hai; ye block stacked layers mein repeat hota hai.

Step 14: Linear layer + Softmax — next token predict karna
Decoder ke final output ko Linear layer vocabulary ke scores mein convert karti hai. In scores ko logits kehte hain.
Maan lo current step par model ko in candidates ke scores milte hain:
Candidate token	Illustrative probability
मुझे	0.82
मैं	0.10
हम	0.05
अन्य tokens	0.03
Ye sirf example probabilities hain. Model inmein se decoding strategy ke according next token choose karta hai. Greedy decoding mein highest-probability token choose hoga.
\[
P(\text{token})=\operatorname{Softmax}(\text{logits})
\]

Pehla predicted token maan lete hain: मुझे.
Ek important point: Softmax har possible token ki probability nikalta hai; model ko sirf in chaar options tak limited nahi kiya gaya hai. Table mein baaki vocabulary ko simplify karke group kiya hai.

Step 15: Inference — poora sentence kaise generate hota hai?
Ab saare steps ko actual generation loop mein dekhte hain.
Generation step 1 of 6
Predicting

Decoder input
<SOS>


Next token prediction
मुझे

Decoder ko start token milta hai. Encoder English input ki information provide karta hai.
PreviousNext step

Ye loop tab tak chalta hai jab tak <EOS> generate na ho jaye, ya maximum output length jaisi koi stopping condition hit na ho.
Isko autoregressive generation kehte hain: har next token prediction mein pehle generated tokens ka context use hota hai.
Training aur inference mein difference:
Training	Inference
Correct target tokens available hote hain	Correct output pehle se available nahi hota
Shifted target sequence input mein di ja sakti hai	Start token se shuru hota hai
Causal mask ki wajah se positions parallel train ho sakti hain	Output token by token generate hota hai
Predictions aur correct targets se loss calculate hota hai	Predicted tokens ko next step ke input mein add karte hain
Training mein predicted output aur actual target ke beech loss calculate hota hai, phir backpropagation aur optimizer weights update karte hain. Inference mein aam taur par model weights update nahi hote.

Final revision: poora Transformer ek flow mein
English input sentence
I love machine learning

Tokenization → Embedding + Positional Encoding

ENCODER
Self-Attention → Add & Norm → FFN → Add & Norm
Repeat across layers

DECODER
Masked Self-Attention
Cross-Attention (encoder K/V + decoder Q)
FFN, with residual connections + LayerNorm

Linear → Softmax → Next token

Predicted token ko decoder input mein add karo; repeat until EOS
Final Hindi output
मुझे मशीन लर्निंग पसंद है


Bas ye 6 differences yaad rakhna
- Embedding: Token ID ko vector banata hai.
- Positional Encoding: Token ki position ki information deta hai.
- Self-Attention: Same sequence ke tokens ke relationships samajhta hai.
- Masked Self-Attention: Decoder ko future tokens dekhne se rokta hai.
- Cross-Attention: Decoder ko encoder ki input information use karne deta hai.
- FFN: Har token ki representation ko independently transform karta hai.
