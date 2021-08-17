import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_score
from collections import defaultdict
from textwrap import wrap
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from utils import get_data
    
"""
References:
https://pytorch.org/tutorials/recipes/recipes_index.html
https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
https://curiousily.com/posts/sentiment-analysis-with-bert-and-hugging-face-using-pytorch-and-python/
"""


def get_train_test_set(count,randomise_coeff=False):
    
    filename='rate_helpfulness\dataset\All_Amazon_Review.json.gz'


    helpful,sentences = get_data(filename,randomise_coeff=randomise_coeff,count=count)

    # load, and create train test sets
    X = pd.DataFrame(sentences, columns=['Text'])
    y = pd.DataFrame([x if x<1 else 1 for x in helpful],columns=['Pred'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, shuffle=True)
    print('Loaded Dataset')

    return X_train, X_test, y_train, y_test


class AmazonReviewDataset(Dataset):
    def __init__(self, reviews, targets, tokenizer, max_len):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)


    def __getitem__(self, item):
        try:
            review = str(self.reviews.values.flatten()[item])
            target = self.targets.values.flatten()[item]
            encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
            )
        except Exception as e:
            pass

        return {
        'review_text': review,
        'input_ids': encoding['input_ids'].flatten(),
        'attention_mask': encoding['attention_mask'].flatten(),
        'targets': torch.tensor(target, dtype=torch.long)
        }


def create_data_loader(X, y, tokenizer, max_len, batch_size):
    ds = AmazonReviewDataset(
        reviews=X['Text'],
        targets=y,
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4
    )

def display_density_tokencount_plot(df):
    # token length of each review
    token_lens = []
    for txt in df.values:
        tokens = tokenizer.encode(txt[0], max_length=512)
        token_lens.append(len(tokens))

    sns.distplot(token_lens)
    plt.xlim([0, 256]);
    plt.xlabel('Token count');
    plt.show();


class HelpfulnessClassifier(nn.Module):
    def __init__(self, n_classes):
        super(HelpfulnessClassifier, self).__init__()
        torch.cuda.empty_cache()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
    def forward(self, input_ids, attention_mask):
        o = self.bert(input_ids=input_ids,attention_mask=attention_mask)
        output = self.drop(o['pooler_output'])
        return self.out(output)

def train_epoch(
    model,
    data_loader,
    loss_fn,
    optimizer,
    device,
    scheduler,
    n_examples
    ):
    model = model.train()
    losses = []
    correct_predictions = 0
    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)
        outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
        )
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)
        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return correct_predictions.double() / n_examples, np.mean(losses)



def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)
            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())
    return correct_predictions.double() / n_examples, np.mean(losses)


def get_predictions(model, data_loader):
    model = model.eval()
    review_texts = []
    predictions = []
    prediction_probs = []
    real_values = []
    with torch.no_grad():
        for d in data_loader:
            texts = d["review_text"]
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)
            review_texts.extend(texts)
            predictions.extend(preds)
            prediction_probs.extend(outputs)
            real_values.extend(targets)
    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()
    return review_texts, predictions, prediction_probs, real_values

def predict_review(review_text,model_file):

    def get_model(model_file,device):
        model = HelpfulnessClassifier(2)
        model.load_state_dict(torch.load(model_file, map_location=device))
        model.to(device)
        return model
    # raw text prediction
    #review_text = "I love completing my todos! Best app ever!!!"
    MAX_LEN = 200
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    model=get_model(model_file,device)
    encoded_review = tokenizer.encode_plus(
        review_text,
        max_length=MAX_LEN,
        add_special_tokens=True,
        return_token_type_ids=False,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
        )
    input_ids = encoded_review['input_ids'].to(device)
    attention_mask = encoded_review['attention_mask'].to(device)
    output = model(input_ids, attention_mask)
    _, prediction = torch.max(output, dim=1)

    return prediction

PRE_TRAINED_MODEL_NAME = 'bert-base-cased'

if __name__=='__main__':
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    X_train, X_test, y_train, y_test=get_train_test_set(count=50000,randomise_coeff=False)
    
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

    MAX_LEN = 200
    BATCH_SIZE = 8
    train_data_loader = create_data_loader(X_train,y_train,tokenizer, MAX_LEN, BATCH_SIZE)
    test_data_loader = create_data_loader(X_test, y_test, tokenizer, MAX_LEN, BATCH_SIZE)

    #train data
    data = next(iter(train_data_loader))
    print(data.keys());print(data['input_ids'].shape);print(data['attention_mask'].shape);print(data['targets'].shape);

    #gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    #initialise model
    model = HelpfulnessClassifier(2)
    model = model.to(device)
    
    #inputs to gpu
    input_ids = data['input_ids'].to(device)
    attention_mask = data['attention_mask'].to(device)
    
    #training
    EPOCHS = 6
    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    total_steps = len(train_data_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
        )
    loss_fn = nn.CrossEntropyLoss().to(device)

    #training loop
    history = defaultdict(list)
    best_accuracy = 0
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)
        train_acc, train_loss = train_epoch(
            model,
            train_data_loader,
            loss_fn,
            optimizer,
            device,
            scheduler,
            len(X_train)
        )
        print(f'Train loss {train_loss} accuracy {train_acc}')
        test_acc, test_loss = eval_model(
            model,
            test_data_loader,
            loss_fn,
            device,
            len(X_test)
        )
        print(f'test   loss {test_loss} accuracy {test_acc}')
        print()
        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['test_acc'].append(test_acc)
        history['test_loss'].append(test_loss)
        if test_acc > best_accuracy:
            torch.save(model.state_dict(), 'best_model_state.bin')
            best_accuracy = test_acc

    plt.plot(history['train_acc'], label='train accuracy')
    plt.plot(history['test_acc'], label='validation accuracy')
    plt.title('Training history')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.ylim([0, 1]);
    plt.savefig('./train_val_epochs.png')

    y_review_texts, y_pred, y_pred_probs, y_test = get_predictions(model,test_data_loader)
    print('Training Metrics')
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    # load and predict
    del model
    model = HelpfulnessClassifier(2)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load('best_model_state.bin', map_location=device))
    model.to(device)
    y_review_texts, y_pred, y_pred_probs, y_test = get_predictions(model,test_data_loader)


    # model metrics
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    with open('./bert_models_stat.txt','w',newline='') as f:
        f.write('accuracy:'+str(accuracy_score(y_test, y_pred))+'\n')
        f.write('precision:'+str(precision_score(y_test, y_pred))+'\n')
        f.write('classification_report:\n'+str(classification_report(y_test, y_pred))+'\n')
        f.write('confusion matrix:\n'+str(confusion_matrix(y_test, y_pred))+'\n')


    # write predictions to file
    with open('./predicted_bert_results.csv','w') as f:
        for x,y,y_p,y_pp in zip(X_test.values,y_test,y_pred,y_pred_probs):
            f.write(f"{x}\t{y}\t{y_p}\t{y_pp}\n")


    print("End")