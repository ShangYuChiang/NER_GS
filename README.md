# Automatically extracting gene and species name(s) from abstract text

Recognizing entities in text is the first step towards machines that can extract insights out of enormous document repositories like pubmed.<br>

## Getting Started
> ## Prerequisites
Using the Python NLP software library spaCy to extract genes from pubmed text
* python IDLE / Anaconda / Visual Studio Code
* spaCy - Open-source library for industrial-strength Natural Language Processing (NLP) in Python

> ## Installation

spaCy is compatible with 64-bit CPython 2.7 / 3.5+ and runs on Unix/Linux, macOS/OS X and Windows.The latest spaCy releases are available over 

Windows & OS X & Linux
* Run the below command in Command Prompt<br> 
( Make sure you Add Python to PATH )
```sh
pip install -U spacy
```
* Run the below command in Anaconda Prompt<br>
( Run as administrator )
```sh
conda install -c conda-forge spacy
```

> ## Usage example and Code processing walkthrough
* Load the model, or create an empty model<br>
We can create an empty model and train it with our annotated dataset or we can use existing spacy model and re-train with our annotated data.<br>

```python
if model is not None:
    nlp = spacy.load(model)  # load existing spaCy model
    print("Loaded model '%s'" % model)
else:
    nlp = spacy.blank("en")  # create blank Language class
    print("Created blank 'en' model")

if 'ner' not in nlp.pipe_names :
    ner = nlp.create_pipe('ner')
    nlp.add_pipe(ner, last=True)
else :
    ner = nlp.get_pipe("ner")
```
* Adding Labels or entities<br>

```python
# add labels
for _, annotations in train_data:
    for ent in annotations.get('entities'):
        ner.add_label(ent[2])

other_pipe = [pipe for pipe in nlp.pipe_names if pipe != 'ner']

# Only training NER
with nlp.disable_pipes(*other_pipe) :
    if model is None:
        optimizer = nlp.begin_training()
    else:
        optimizer = nlp.resume_training()
```
* Training and updating the model<br>
Training data : Annotated data contain both text and their labels<br>
Text : Input text the model should predict a label for.<br>
Label : The label the model should predict.<br>
```python
# Spacy Training Data Format
Train_data = [
    ( "Text 1", entities : {
                [(start,end, "Label 1"), (start,end, "Label 2"), (start,end, "Label 3")]
                }
    ),
    ( "Text 2", entities : {
             [(start,end, "Label 1"), (start,end,"Label 2")]
             }
    ),
    ( "Text 3", entities : {
            [(start,end, "Label 1"), (start,end, "Label 2"), 
            (start,end,"Label 3"),(start,end, "Label 4 ")]
            }
    )
]
```

1. We will train our model for a number of iterations so that the model can learn from it effectively.<br>


```python
for int in range(iteration) :
    print("Starting iteration" + str(int))
    random.shuffle(train_data)
    losses = {}
```
2. At each iteration, the training data is shuffled to ensure the model doesn’t make any generalisations based on the order of examples.<br>
3. We will update the model for each iteration using  <b>`nlp.update()`</b>. 
```python
    for text, annotation in train_data :
        nlp.update(
        [text],
        [annotation],
        drop = 0.2,
        sgd = optimizer,
        losses = losses
        )
  #print(losses)
new_model = nlp
```

* Evaluate the model<br>

```python
# Spacy Testing Data Format
test_data = [
    ('Text 1',
     [(start, end, 'Label 1')]),
    ('Text 2',
     [(start, end, 'Label 1'), (start, end, 'Label 2')])
]
```
```python
import spacy
from spacy.gold import GoldParse
from spacy.scorer import Scorer

def evaluate(model, examples):
  scorer = Scorer()
  for input_, annot in examples:
    #print(input_)
    doc_gold_text = model.make_doc(input_)
    gold = GoldParse(doc_gold_text, entities=annot['entities'])
    pred_value = model(input_)
    scorer.score(pred_value, gold)
  return scorer.scores

test_result = evaluate(new_model, test_data)
```
*  Visualization <br>
```python
import spacy
from spacy import displacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("""Epigenetic Silencing of the mutL homolog 1 (MLH1) Promoter in
Relation to the Development of Gastric Cancer (GC) and its use as a
Biomarker for Patients with Microsatellite Instability.""")
# Since this is an interactive Jupyter environment, we can use displacy.render here
displacy.render(doc,jupyter=True,style='ent')
```

## Reference
See the [spaCy Tutorials](https://spacy.io/usage/spacy-101) for more details and examples<br>
[1] [How to create custom NER in Spacy](https://confusedcoders.com/data-science/deep-learning/how-to-create-custom-ner-in-spacy)<br>
[2] [How to extract genes from text with Sysrev and spaCy](https://blog.sysrev.com/simple-ner/)<br>
[3] [Custom Named Entity Recognition Using spaCy](https://towardsdatascience.com/custom-named-entity-recognition-using-spacy-7140ebbb3718)
