#!/usr/bin/env python
# coding: utf-8

# # 비매너 댓글 식별

# # 1. 라이브러리 로드 & 환경설정

# In[1]:


import pandas as pd
import numpy as np 
import os
import json
import random
import shutil

from attrdict import AttrDict

from sklearn.metrics import f1_score
from datetime import datetime, timezone, timedelta
import random
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils import *
from torch.optim import Adam, AdamW

from transformers import logging, get_linear_schedule_with_warmup

from transformers import (
    BertTokenizer,  
    AutoTokenizer,
    ElectraTokenizer,
    AlbertTokenizer
)

from transformers import (
    BertModel,
    AutoModel, 
    ElectraForSequenceClassification,
    BertForSequenceClassification,
    AlbertForSequenceClassification
)


# ### 1-1 DEVICE 설정

# In[2]:


# 사용할 GPU 지정
print("number of GPUs: ", torch.cuda.device_count())
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
use_cuda = torch.cuda.is_available()
print("Does GPU exist? : ", use_cuda)
DEVICE = torch.device("cuda" if use_cuda else "cpu")


# ### 1-2 DEBUG 설정

# In[3]:


# True 일 때 코드를 실행하면 example 등을 보여줌
DEBUG = False

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default='config.json', help='Specific config file for an experiment')
    return parser.parse_args()

parser = parse_args()
config_path = os.path.join(parser.config_file)


# ### Config 설정

# In[4]:


config_path = os.path.join('config', 'config.json')
print(config_path)

# config 적용 
def set_config(config_path):
    if os.path.lexists(config_path):
        with open(config_path, 'r') as f:
            args = AttrDict(json.load(f))
            
            print('config file loaded')
            print(args.pretrained_model)
    else:
        assert False, 'config json file cannot be found.. please check the path again.'
        
    return args

args = set_config(config_path)

os.makedirs(args.result_dir, exist_ok=True)
os.makedirs(args.config_dir, exist_ok=True)


# # 2. EDA

# ## 2-1. Train data 로드

# In[5]:


train_path = os.path.join(args.data_dir, 'train.csv')


# ## 2-2 Train data 확인

# In[6]:


train_df = pd.read_csv(train_path, encoding='UTF-8-SIG')
train_df.head()


# In[7]:


print("train data 개수 : ", len(train_df))


# ## 2-3 Train 데이터 분포
# 
# title의 길이 분포

# In[8]:


ax = train_df['title'].str.len().hist()
ax


# In[9]:


print('title 최대 길이 : ', train_df['title'].str.len().max())
print('title 평균 길이 : ', train_df['title'].str.len().mean())


# In[10]:


ax = train_df['comment'].str.len().hist()

print()

ax


# In[11]:


print('comment 최대 길이 : ', train_df['comment'].str.len().max())
print('comment 평균 길이 : ', train_df['comment'].str.len().mean())


# ### bias와 hate 비중

# In[12]:


print('bias 클래스 : ', train_df.bias.unique())
print('hate 클래스 : ', train_df.hate.unique())


# **bias 수**

# In[13]:


plt.figure(figsize=(8, 8))
sns.histplot(x=train_df['bias'])

count_none = sum(train_df.bias == 'none')
count_others = sum(train_df.bias == 'others')
count_gender = sum(train_df.bias == 'gender')

print(f'bias-none 개수 : {count_none} / 비율 : {round(count_none/len(train_df)*100, 2)}%')
print(f'bias-others 개수 : {count_others} / 비율 : {round(count_others/len(train_df)*100, 2)}%')
print(f'bias-gender 개수 : {count_gender} / 비율 : {round(count_gender/len(train_df)*100, 2)}%')
plt.show()


# **hate 수**

# In[14]:


plt.figure(figsize=(8, 8))
sns.histplot(x=train_df['hate'])

count_hate_none = sum(train_df.hate == 'none')
count_hate_hate = sum(train_df.hate == 'hate')

print(f'hate-none 개수 : {count_hate_none} / 비율 : {round(count_hate_none/len(train_df)*100, 2)}%')
print(f'hate-hate 개수 : {count_hate_hate} / 비율 : {round(count_hate_hate/len(train_df)*100, 2)}%')

plt.show()


# **bias, hate 비율**

# In[15]:


len(train_df['title'])


# In[16]:


bias_hate = train_df.iloc[:, 1:].pivot_table(index='bias', columns='hate', aggfunc='count')
bias_hate


# In[17]:


plt.figure(figsize=(8, 8))

sns.heatmap(bias_hate, cmap=sns.light_palette('blue', as_cmap=True), annot=True, fmt='g')
plt.title("bias-hate heat map")


# bias와 hate의 관계성? 연관성이 존재하는가?? 

# 

# ## 2-5. Test 데이터 로드

# In[18]:


test_path = os.path.join(args.data_dir,'test.csv')
print("test 데이터 경로가 올바른가요? : ", os.path.lexists(test_path))


# In[19]:


test_df = pd.read_csv(test_path)
test_df.head()


# In[20]:


ax = test_df['title'].str.len().hist()
ax


# In[21]:


print('title 최대 길이 : ', test_df['title'].str.len().max())
print('title 평균 길이 : ', test_df['title'].str.len().mean())


# In[22]:


ax = test_df['comment'].str.len().hist()

print()

ax


# In[23]:


print('comment 최대 길이 : ', test_df['comment'].str.len().max())
print('comment 평균 길이 : ', test_df['comment'].str.len().mean())


# 

# 

# # 3. 데이터 전처리

# ## 3-1. 라벨값 추가

# In[24]:


# 두 라벨의 가능한 모든 조합 만들기
combinations = np.array(np.meshgrid(train_df.bias.unique(), train_df.hate.unique())).T.reshape(-1,2)

if DEBUG==True:
    print(combinations)


# In[25]:


# bias, hate 컬럼을 합친 것
bias_hate = list(np.array([train_df['bias'].values, train_df['hate'].values]).T.reshape(-1,2))

if DEBUG==True:
    print(bias_hate[:5])


# In[26]:


labels = []
for i, arr in enumerate(bias_hate):
    for idx, elem in enumerate(combinations):
        if np.array_equal(elem, arr):
            labels.append(idx)

train_df['label'] = labels
train_df.head()


# # 4. Dataset 로드 

# 

# In[27]:


# config.json 에서 지정 이름별로 가져올 라이브러리 지정

from transformers import AutoTokenizer, AutoModel

TOKENIZER_CLASSES = {
    "BertTokenizer": BertTokenizer,
    "AutoTokenizer": AutoTokenizer,
    "ElectraTokenizer": ElectraTokenizer,
    "AlbertTokenizer": AlbertTokenizer,
    "AutoTokenizer": AutoTokenizer
}


# In[28]:


TOKENIZER = TOKENIZER_CLASSES[args.tokenizer_class].from_pretrained(args.pretrained_model)
if DEBUG==True:
    print(TOKENIZER)


# In[29]:


if DEBUG == True:
    example = train_df['title'][0]
    print('example : ', example)
    
    comment_ex = train_df['comment'][0]
    print('comment_ex : ', comment_ex)
    
    print(TOKENIZER(example, comment_ex))


# In[30]:


if DEBUG == True:
    print('encode : ', TOKENIZER.encode(example), "\n")
    
    print('tokenize : ', TOKENIZER.tokenize(example), "\n")
    
    print('convert tokens to ids : ', TOKENIZER.tokenize(example))


# ## 4-1. 데이터 정제 함수 정의

# In[31]:


import re
import emoji
from soynlp.normalizer import repeat_normalize

emojis = ''.join(emoji.UNICODE_EMOJI.keys())
pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-ㅣ가-힣{emojis}]+')
url_pattern = re.compile(
    r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')

def clean_text(x):
    x = pattern.sub(' ', x)
    x = url_pattern.sub('', x)
    x = x.strip()
    x = repeat_normalize(x, num_repeats=2)
    return x


# In[32]:


x = clean_text("'미스터 션샤인' 변요한, ㅋㅋㅋㅋㅋ     김태리와 같은 양복 입고 학당 방문! 이유는?????")
print(x)


# ## 4-1. Dataset 함수 정의

# In[33]:


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_len, mode='train', clean=None):
        self.data = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mode = mode
        self.clean = clean
        
        if self.mode != 'test':
            try:
                self.labels = df['label'].tolist()
            except:
                assert False, "CustomDataset Error! : \'label\' columns does not exist in the dataframe. check mode is not train"
    
    def __len__(self):
        return len(self.data)
    
    
    def __getitem__(self, idx):
        title = self.data.title.iloc[idx]
        comment = self.data.comment.iloc[idx]
        
        if self.clean is not None:
            title = self.clean(title)
            comment = self.clean(comment)
        
        
        tokenized_text = self.tokenizer(title,
                                       comment,
                                       padding='max_length',
                                       max_length=self.max_len,
                                       truncation=True,
                                       return_token_type_ids=True,
                                       return_attention_mask=True,
                                       return_tensors='pt')
        data = {
            'input_ids': tokenized_text['input_ids'].clone().detach().long(),
            'attention_mask': tokenized_text['attention_mask'].clone().detach().long(),
            'token_type_ids': tokenized_text['token_type_ids'].clone().detach().long()
        }
        
        if self.mode != 'test':
            label = self.data.label.iloc[idx]
            return data, label
        else:
            return data


# In[34]:


train_dataset = CustomDataset(df=train_df, tokenizer=TOKENIZER, max_len=args.max_seq_len, mode='train', clean=clean_text)
print("train dataset loaded.")


# In[35]:


if DEBUG == True:
    print('dataset sample : ')
    print(train_dataset[0])


# ## 4-3. Train/Validation set 나누기

# In[36]:


from sklearn.model_selection import train_test_split
                                                         
train_data, val_data = train_test_split(train_df, test_size=0.1, random_state=args.seed)

train_dataset = CustomDataset(train_data, TOKENIZER, args.max_seq_len, 'train')
val_dataset = CustomDataset(val_data, TOKENIZER, args.max_seq_len, 'validation')

print("Train dataset: ", len(train_dataset))
print("Validation dataset: ", len(val_dataset))


# # 5. 분류 모델 세팅

# In[37]:


from transformers import logging
logging.set_verbosity_error()

BASE_MODEL = {
    "BertForSequenceClassification": BertForSequenceClassification,
    "AutoModel": AutoModel,
    "ElectraForSequenceClassification": ElectraForSequenceClassification,
    "AlbertForSequenceClassification": AlbertForSequenceClassification,
}


# In[38]:


myModel = BASE_MODEL[args.architecture].from_pretrained(args.pretrained_model,
                                                       num_labels = args.num_classes,
                                                       output_attentions=False, 
                                                       output_hidden_states=True)


# In[39]:


if DEBUG == True:
    print(myModel)


# In[40]:


### v2 에서 일부 수정됨
class myClassifier(nn.Module):
    def __init__(self, model, hidden_size = 768, num_classes=args.num_classes, selected_layers=False, params=None):
        super(myClassifier, self).__init__()
        self.model = model
        self.softmax = nn.Softmax(dim=1) 
        self.selected_layers = selected_layers
        
        # 사실 dr rate은 model config 에서 hidden_dropout_prob로 가져와야 하는데 bert에선 0.1이 쓰였음
        self.dropout = nn.Dropout(0.1)


    def forward(self, token_ids, attention_mask, segment_ids):      
        outputs = self.model(input_ids = token_ids, 
                             token_type_ids = segment_ids.long(), 
                             attention_mask = attention_mask.float().to(token_ids.device))
        
        # hidden state에서 마지막 4개 레이어를 뽑아 합쳐 새로운 pooled output 을 만드는 시도
        if self.selected_layers == True:
            hidden_states = outputs.hidden_states
            pooled_output = torch.cat(tuple([hidden_states[i] for i in [-4, -3, -2, -1]]), dim=-1)
            # print("concatenated output shape: ", pooled_output.shape)
            ## dim(batch_size, max_seq_len, hidden_dim) 에서 가운데를 0이라 지정함으로, [cls] 토큰의 임베딩을 가져온다. 
            ## (text classification 구조 참고)
            pooled_output = pooled_output[:, 0, :]
            # print(pooled_output)

            pooled_output = self.dropout(pooled_output)

            ## 3개의 레이어를 합치므로 classifier의 차원은 (hidden_dim, 6)이다
            classifier = nn.Linear(pooled_output.shape[1], args.num_classes).to(token_ids.device)
            logits = classifier(pooled_output)
        
        else:
            logits=outputs.logits
        
    
        # 각 클래스별 확률
        prob= self.softmax(logits)
        # print(prob)
        # logits2 = outputs.logits
        # print(self.softmax(logits2))


        return logits, prob
        
# 마지막 4 hidden layers concat 하는 방법을 쓰신다면 True로 변경        
model = myClassifier(myModel, selected_layers=False)

# if DEBUG ==True :
#     print(model)


# In[41]:


if DEBUG==True:
    params = list(model.named_parameters())

    print('The BERT model has {:} different named parameters.\n'.format(len(params)))

    print('==== Embedding Layer ====\n')

    for p in params[0:5]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    print('\n==== First Transformer ====\n')

    for p in params[5:21]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    print('\n==== Output Layer ====\n')

    for p in params[-4:]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))


# # 6. 학습

# ## 6-1. Early Stopper 함수 정의

# In[42]:


class LossEarlyStopper():
    """Early stopper

        patience (int): loss가 줄어들지 않아도 학습할 epoch 수
        patience_counter (int): loss 가 줄어들지 않을 때 마다 1씩 증가
        min_loss (float): 최소 loss
        stop (bool): True 일 때 학습 중단

    """

    def __init__(self, patience: int)-> None:
        """ 초기화

        Args:
            patience (int): loss가 줄어들지 않아도 학습할 epoch 수
            weight_path (str): weight 저장경로
            verbose (bool): 로그 출력 여부, True 일 때 로그 출력
        """
        self.patience = patience
        self.patience_counter = 0
        self.min_loss = np.Inf
        self.stop = False

    def check_early_stopping(self, loss: float)-> None:
        # 첫 에폭
        if self.min_loss == np.Inf:
            self.min_loss = loss
           
        # loss가 줄지 않는다면 -> patience_counter 1 증가
        elif loss > self.min_loss:
            self.patience_counter += 1
            msg = f"Early stopping counter {self.patience_counter}/{self.patience}"

            # patience 만큼 loss가 줄지 않았다면 학습을 중단합니다.
            if self.patience_counter == self.patience:
                self.stop = True
            print(msg)
        # loss가 줄어듬 -> min_loss 갱신, patience_counter 초기화
        elif loss <= self.min_loss:
            self.patience_counter = 0
            ### v2 에서 수정됨
            ### self.save_model = True -> 삭제 (사용하지 않음)
            msg = f"Validation loss decreased {self.min_loss} -> {loss}"
            self.min_loss = loss

            print(msg)


# ## 6-2. Epoch 별 학습 및 검증

# - [Transformers optimization documentation](https://huggingface.co/docs/transformers/main_classes/optimizer_schedules)
# - [스케줄러 documentation](https://huggingface.co/docs/transformers/main_classes/optimizer_schedules#schedules)
# - Adam optimizer의 epsilon 파라미터 eps = 1e-8 는 "계산 중 0으로 나눔을 방지 하기 위한 아주 작은 숫자 " 입니다. ([출처](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/))
# - 스케줄러 파라미터
#     - `warmup_ratio` : 
#       - 학습이 진행되면서 학습률을 그 상황에 맞게 가변적으로 적당하게 변경되게 하기 위해 Scheduler를 사용합니다.
#       - 처음 학습률(Learning rate)를 warm up하기 위한 비율을 설정하는 warmup_ratio을 설정합니다.
#   

# In[43]:


args = set_config(config_path)

logging.set_verbosity_warning()

# 재현을 위해 모든 곳의 시드 고정
seed_val = args.seed
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

def train(model, train_data, val_data, args, mode = 'train'):
    
    # args.run은 실험 이름 (어디까지나 팀원들간의 버전 관리 및 공유 편의를 위한 것으로, 자유롭게 수정 가능합니다.)
    print("RUN : ", args.run)
    shutil.copyfile("config/config.json", os.path.join(args.config_dir, f"config_{args.run}.json"))

    early_stopper = LossEarlyStopper(patience=args.patience)
    
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.train_batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=args.train_batch_size)

    
    if DEBUG == True:
        # 데이터로더가 성공적으로 로드 되었는지 확인
        for idx, data in enumerate(train_dataloader):
            if idx==0:
                print("batch size : ", len(data[0]['input_ids']))
                print("The first batch looks like ..\n", data[0])
    
    
    criterion = nn.CrossEntropyLoss()
    
    total_steps = len(train_dataloader) * args.train_epochs

    ### v2에서 수정됨 (Adam -> AdamW)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * args.warmup_proportion), 
                                                num_training_steps=total_steps)

    
    if use_cuda:
        model = model.to(DEVICE)
        criterion = criterion.to(DEVICE)
        

    tr_loss = 0.0
    val_loss = 0.0
    best_score = 0.0
    best_loss = np.inf
      

    for epoch_num in range(args.train_epochs):

            total_acc_train = 0
            total_loss_train = 0
            
            assert mode in ['train', 'val'], 'your mode should be either \'train\' or \'val\''
            
            if mode =='train':
                for train_input, train_label in tqdm(train_dataloader):
                    
                    
                    mask = train_input['attention_mask'].to(DEVICE)
                    input_id = train_input['input_ids'].squeeze(1).to(DEVICE)
                    segment_ids = train_input['token_type_ids'].squeeze(1).to(DEVICE)
                    train_label = train_label.long().to(DEVICE)  
                    
                    ### v2에 수정됨
                    optimizer.zero_grad()
 
                    output = model(input_id, mask, segment_ids)
                    batch_loss = criterion(output[0].view(-1,6), train_label.view(-1))
                    total_loss_train += batch_loss.item()

                    acc = (output[0].argmax(dim=1) == train_label).sum().item()
                    total_acc_train += acc
                    
                    ### v2에 수정됨
                    optimizer.zero_grad()
                    
                    batch_loss.backward()
                    optimizer.step()
                    
                    ### v2 에 수정됨
                    scheduler.step()
                    

            total_acc_val = 0
            total_loss_val = 0
            
            # validation을 위해 이걸 넣으면 이 evaluation 프로세스 중엔 dropout 레이어가 다르가 동작한다.
            model.eval()
            
            with torch.no_grad():

                for val_input, val_label in val_dataloader:

                    mask = val_input['attention_mask'].to(DEVICE)
                    input_id = val_input['input_ids'].squeeze(1).to(DEVICE)
                    segment_ids = val_input['token_type_ids'].squeeze(1).to(DEVICE)
                    val_label = val_label.long().to(DEVICE)

                    output = model(input_id, mask, segment_ids)
                    ### v2 에서 일부 수정 (output -> output[0]로 myClassifier 모델에 정의된대로 logits 가져옴)
                    batch_loss = criterion(output[0].view(-1,6), val_label.view(-1))
                    total_loss_val += batch_loss.item()
                    
                    ### v2 에서 일부 수정 (output -> output[0]로 myClassifier 모델에 정의된대로 logits 가져옴)
                    acc = (output[0].argmax(dim=1) == val_label).sum().item()
                    total_acc_val += acc
            
            
            train_loss = total_loss_train / len(train_data)
            train_accuracy = total_acc_train / len(train_data)
            val_loss = total_loss_val / len(val_data)
            val_accuracy = total_acc_val / len(val_data)
            
            # 한 Epoch 학습 후 학습/검증에 대해 loss와 평가지표 (여기서는 accuracy로 임의로 설정) 출력
            print(
                f'Epoch: {epoch_num + 1} \
                | Train Loss: {train_loss: .3f} \
                | Train Accuracy: {train_accuracy: .3f} \
                | Val Loss: {val_loss: .3f} \
                | Val Accuracy: {val_accuracy: .3f}')
          
            # early_stopping check
            early_stopper.check_early_stopping(loss=val_loss)

            if early_stopper.stop:
                print('Early stopped, Best score : ', best_score)
                break

            ### v2 에 수정됨
            ### loss와 accuracy가 꼭 correlate하진 않습니다.
            ### 
            ### 원본 (필요하다면 다시 해제 후 사용)
            # if val_accuracy > best_score : 
            if val_loss < best_loss :
            # 모델이 개선됨 -> 검증 점수와 베스트 loss, weight 갱신
                best_score = val_accuracy 
                
                ### v2에서 추가
                best_loss =val_loss
                # 학습된 모델을 저장할 디렉토리 및 모델 이름 지정
                SAVED_MODEL =  os.path.join(args.result_dir, f'best_{args.run}.pt')
            
                check_point = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                }
                torch.save(check_point, SAVED_MODEL)  
              
            # print("scheduler : ", scheduler.state_dict())


    print("train finished")


train(model, train_dataset, val_dataset, args, mode = 'train')


# # 7. Test dataset으로 추론 (Prediction)
# 
# 
# - v2 에서 수정된 부분
#     - output -> output[0]

# In[44]:


from torch.utils.data import DataLoader

# 테스트 데이터셋 불러오기
test_data = CustomDataset(test_df, tokenizer = TOKENIZER, max_len= args.max_seq_len, mode='test')

def test(model, SAVED_MODEL, test_data, args, mode = 'test'):


    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.eval_batch_size)


    if use_cuda:

        model = model.to(DEVICE)
        model.load_state_dict(torch.load(SAVED_MODEL)['model'])


    model.eval()

    pred = []

    with torch.no_grad():
        for test_input in test_dataloader:

            mask = test_input['attention_mask'].to(DEVICE)
            input_id = test_input['input_ids'].squeeze(1).to(DEVICE)
            segment_ids = test_input['token_type_ids'].squeeze(1).to(DEVICE)

            output = model(input_id, mask, segment_ids)

            output = output[0].argmax(dim=1).cpu().tolist()

            for label in output:
                pred.append(label)
                
    return pred

SAVED_MODEL =  os.path.join(args.result_dir, f'best_{args.run}.pt')

pred = test(model, SAVED_MODEL, test_data, args)


# In[45]:


print("prediction completed for ", len(pred), "comments")


# ### 

# In[46]:


# 0-5 사이의 라벨 값 별로 bias, hate로 디코딩 하기 위한 딕셔너리
bias_dict = {0: 'none', 1: 'none', 2: 'others', 3:'others', 4:'gender', 5:'gender'}
hate_dict = {0: 'none', 1: 'hate', 2: 'none', 3:'hate', 4:'none', 5:'hate'}

# 인코딩 값으로 나온 타겟 변수를 디코딩
pred_bias = ['' for i in range(len(pred))]
pred_hate = ['' for i in range(len(pred))]

for idx, label in enumerate(pred):
    pred_bias[idx]=(str(bias_dict[label]))
    pred_hate[idx]=(str(hate_dict[label]))
print('decode Completed!')


# In[47]:


submit = pd.read_csv(os.path.join(args.data_dir,'sample_submission.csv'))
submit


# In[48]:


submit['bias'] = pred_bias
submit['hate'] = pred_hate
submit


# In[49]:


submit.to_csv(os.path.join(args.result_dir, f"submission_{args.run}.csv"), index=False)

