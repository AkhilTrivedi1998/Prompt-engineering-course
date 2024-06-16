# Prompt Engineering

**Prompt** - Instruction or context provided to an AI for a certain task.

Chatgpt has a training cut-off date meaning it has been trained with data before this date.

If you want to access data after the date, it still has the feature of internet browsing and can get the data that way.

Chatgpt also has the feature to interact with images, it can both generate and analyse the images.

 **LLMs** are **non-deterministic**. That means that even if we use the same LLM, and provide it with the same prompt, we'll get at least slightly different responses! That means even if you're using the same LLM that I'm using in a video, and do exactly as I do, you may get a different result.

 First providing context to the LLM and then asking for solution is a better and efficient approach. 
 For example, if we directly ask the LLM to create a snake game in python. It will write the code for it but the code might not be efficient.
 Better approach is to do it in two steps i.e first ask LLM what are the different features of a good snake game and then ask it to give the code of a snake game in python containing the above features. Because the LLM has better context this time, it will provide code implementing all good features of a snake game. 

 ## Tokens

 In deep learning, particularly in natural language processing (NLP), a token is a unit of text that a model uses as input. Tokens can represent different levels of linguistic units, such as:
 
 1. Words: The most common tokenization unit, where each word in a sentence is considered a token. For example, in the sentence "I love deep learning" the tokens are ["I", "love", "deep", "learning"]

2. Subwords: Tokens can also be parts of words, especially in models like Byte-Pair Encoding or WordPiece. This allows handling of unknown or rare words by breaking them into familiar subunits. For example "unhappiness" might be tokenized as ["un", "happiness"]

3. Characters: In character level tokenization, each character is a token. For example, the sentence "Hello" would be tokenized as ["H", "e", "l", "l", "o"]

4. Sentences: In some contexts, entire sentences can be treated as single tokens.


### Purpose of Tokens in Deep Learning

1. Model Input: Tokens are the basic units that models process. In NLP models like transformers, token sequences are input to embeddings layers, which convert tokens into numerical vectors.

2. Context Representation: Tokens help in understanding the structure and meaning of the text. For example, the sequence of tokens allows models to learn context and relationships between words or subwords.

3. Handling Large Vocabularies: Using subword tokenization helps in managing large vocabularies and rare words, making it feasible to train models on diverse text corpora.

### Tokenization process

Tokenization involves splitting text into tokens and often includes normalization steps like converting to lowercase, removing punctuation, or handling special characters. The choice of tokenization method can significantly impact the performance and behavior of a model.

### Example in Practice

For a transformer-based model like BERT, the tokenization process might include:

* **Splitting**: Breaking down text into tokens.
* **Mapping**: Converting tokens to unique IDs using a vocabulary.
* **Embedding**: Mapping token IDs to dense vectors that represent their meaning in a high-dimensional space.

Here's a simple example with BERT tokenization:

* **Input Sentence**: "The quick brown fox jumps over the lazy dog."
* **Tokenized**: ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
* **IDs**: [101, 1996, 4248, 2829, 4419, 2058, 1996, 13971, 3899, 102] (IDs are hypothetical and represent unique tokens in BERT's vocabulary)
* **Embedding**: Each ID is converted into a vector of fixed dimensions, representing its position in the model's learned space.

### Illustration of different tokenization processes

Here's a diagram illustrating the tokenization process:

* **Original Sentence**: "Natural Language Processing is fascinating."
* **Word Tokenized**: ["Natural", "Language", "Processing", "is", "fascinating", "."]
* **Subword Tokenized**: ["Natural", "Lang", "##uage", "Processing", "is", "fasc", "##inating", "."]
* **Character Tokenized**: ["N", "a", "t", "u", "r", "a", "l", "L", "a", "n", "g", "u", "a", "g", "e", "P", "r", "o", "c", "e", "s", "s", "i", "n", "g", " ", "i", "s", " ", "f", "a", "s", "c", "i", "n", "a", "t", "i", "n", "g", "."]

GPT models use **subword tokenization** to tokenize text into subword units. This allows them to handle a diverse range of vocabulary efficiently, maintain a manageable vocabulary size, and generalize to unseen words by breaking them into known subwords.

### NOTE

Yes, the statement **"subword tokenization is complex but gives better results"** is generally correct, which is why modern NLP models like GPT, BERT, T5 and others uses it.

**Temperature** hyperparameter controls the randomness of the text generated. For example, for a completion model if temperature = 0, then each time the word with the highest probability will get selected, if temperature = 1, then there would be some randomness i.e words with lesser probability can get selected but the entire sentence after completion will still make sense. If temperature = 2(maximum), then there will be extreme randomness and the sentence after completion will become jibrish.  

when we write on a model with temperature = 0 to generate a number between 1 and 30, we always get 17. This is because it is not randomly generating the number 17, somehow based on its training the statistical probability of 17 is highest that is why it generated 17. To simulate randomness we can adjust the temperature value which allows it to select lower probabilistic data as well.

**GPT3** is a LLM with 175 billion parameters spread across 96 layers, trained on 300 billion tokens i.e 45 TB of text data.

Predictive text was used before LLMs, but the problem with predictive texting was that it didn't have long-range attention. It was in 2017 that Google published a paper called "Attention is all you need", which introduced transformer architecture having long-range attention. This enabled it to predict the next word keeping the entire context in mind.

**GPT stands for Generative Pre-trained Transformer.**

There are two phases to training a LLM
1. Pre-training (Base model) - This is where we train the model with random blobs of text.
2. Fine-tuning (Final assistant model) - This is where it is trained to reply in a question-answer format.

### Reversal Curse

This is the phenomenon where LLM is trained to know A = B but because it is not trained specifically to know B = A, it is not able to decipher it intuitively .

**Standard Prompt** - a prompt consisting of only a question or instruction

**System message/System prompt** - is used to communicate instructions or provide context to the model at the beginning of a conversation. It is displayed in a different format compared to user messages, helping the model understand its role in the conversation.

**Context** - the parts of a written or spoken statements that precede or follow a specific word or passage,usually influencing its meaning or effect. To get a proper response from the LLM, it is important useful context to it. General rule is:
***More Context = Better Result***, but this is not always true.  
LLM has no memory of its own, everytime you message the entire previous conversation is sent along with it. This acts as context. But there is a **token limit** for each LLM. If you go beyone that token limit, it will not end the conversation instead it will shift the **context window**, which might lead to loss of important context. This is an example where **More Context != Better Result**

### Key Takeaways about Context

1.  Research has shown that model performance is highest when relevant information occurs at the beginning or at the end of the context window. Model performs worst when relevant context is in the middle (in some cases even worse than when no context is provided), this phenomenon is called **Lost in the middle** . It is somewhat similar to the working of human brain.

2. Model performance decreases as context grows longer (model strugles to retrieve relevant information out of longer context). So, we should always aim to provide only the context which is required.

3. Larger context models are not necessarly better at using context than Shorter context models. Therefore, while judging our focus should be on how much context is required rather than how much context does the model allow.

### Personas and Roles

* Personas help a model give you more accurate outputs related to the role specified by the persona.
* Personas are nothing but additional context
* General Rule - always provide model with a persona relevant to you task.
* Personas can make LLMs more intuitive (and engaging) to interact with. You can give personas a unique tone, style and voice making it both fun and functional.

### Custom Instructions

It is a feature of ChatGPT plus. It is not a system message but it gets passed alongwith the system message, thus acting like a sub-system instruction providing context to the LLM. We can describe our characteristics as well as persona of chatgpt which then gets passed on as a custum instruction.

**IMPORTANT** - LLMs like chatgpt are not good at keeping secrets. You can pass in instructions or prompts in ways which will manipulate or push LLMs into revealing the secrets.

**User Message** - It is nothing but the input passed to the LLM.

#### Proper way of writing user message

1. Write clear and specific instructions. Ex -
    * statement:
        *    Write an article on black hole.
    * More clear and specific:
        *    Write a 1000 word article detailing the progress in imaging black hole from 2010 onwards.
    * Even clearer:
        *    Write a 1000 word article detailing the progress in imaging black hole from 2010 onwards. The article should be written in an engaging tone, and should include techinal details that are explained so that a person with no previous astronomy knowledge could understand.

2. Use **Delimiters** to provide structure to your prompt.  
    Delimiter is a sequence of one or more characters that specify the boundary between separate, independent regions in text. LLMs are trained on code containing data with extensive use of delimiters, so when we use delimiters in our prompts the LLMs is able to recognize the delimiter pattern the understand the instructions more clearly.  
    Example instruction:  
    *   ```
            Using the provided CSV like data, list the names of individuals who have 'Python' listed as one of their skills.  
            ###  
            Name | Age | Occupation, Location; Skills
            John Doe | 32 | Software Engineer, San Francisco; Python, javascript, SQL
            Jane Smith | 28 | Data Scientist, New York; R, Python, Machine Learning
            Ella Brown | 40 | Web Developer, Los Angelos; HTML, CSS, Javascript  
        ```
    Result:  
    *   ```
            John Doe  
            Jane Smith  
        ```
    
    **Now, change the age column value of Ella Brown to python, let's see what happens:**    
    
    Example instruction:  
    *   ```
            Using the provided CSV like data, list the names of individuals who have 'Python' listed as one of their skills.  
            ###  
            Name | Age | Occupation, Location; Skills
            John Doe | 32 | Software Engineer, San Francisco; Python, javascript, SQL
            Jane Smith | 28 | Data Scientist, New York; R, Python, Machine Learning
            Ella Brown | Python | Web Developer, Los Angelos; HTML, CSS, Javascript  
        ```
    Result:  
    *   ```
            John Doe  
            Jane Smith  
        ```
    Still the result is John Doe and Jane Smith, this means that LLM is recognizing the delimiter pattern and is considering Age and Skills as seperate columns. It is looking for Python only within the Skills column. The delimiter above are- **'\###', '|', ',' , ';'**
