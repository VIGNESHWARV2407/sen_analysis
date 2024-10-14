from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

input = 'the internet is slow and down'

roberta = "cardiffnlp/twitter-roberta-base-sentiment"

model = AutoModelForSequenceClassification.from_pretrained(roberta)
tokenizer = AutoTokenizer.from_pretrained(roberta)

labels = ['High priority', 'Moderate priority', 'Low priority']
# labels = ['Low', 'Neutral', 'High']

# sentiment analysis
encoded_input = tokenizer(input, return_tensors='pt')
output = model(**encoded_input)

scores = output[0][0].detach().numpy()
scores = softmax(scores)

for i in range(len(scores)):
    l = labels[i]
    s = scores[i]
    # print(l,s)

max_index = scores.argmax()
max_label = labels[max_index]
max_score = scores[max_index]

print(max_label, max_score)
