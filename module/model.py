import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import BertModel
from .page import Causal_Classifier,PaG

class GuidedMoEBasic(nn.Module):
    def __init__(self, dropout=0.5, n_speaker=2, n_emotion=7, n_cause=2, n_expert=2, guiding_lambda=0, **kwargs):
        super(GuidedMoEBasic, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.emotion_linear = nn.Linear(self.bert.config.hidden_size, n_emotion)
        self.n_expert = n_expert
        self.guiding_lambda = guiding_lambda
        self.gating_network = nn.Linear(2 * (self.bert.config.hidden_size + n_emotion + 1), n_expert)
        self.cause_linear = nn.ModuleList()
        #window窗口大小，utter_dim句子维度，num_bases,max_len句子最大长度,posi_dim位置维度
        self.pag = PaG(2,776,2,10,100)
        self.classifier = Causal_Classifier(776,776)

        for _ in range(n_expert):
            self.cause_linear.append(nn.Sequential(nn.Linear(2 * (self.bert.config.hidden_size + n_emotion + 1), 256),
                                                    nn.Linear(256, n_cause)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask, token_type_ids, speaker_ids):
        '''
        input:
            input_ids:torch.Size([1, 31, 75])
            attention_mask:torch.Size([1, 31, 75])
            token_type_ids:torch.Size([1, 31, 75])
            speaker_ids:torch.size([1,31])

            31:dailydailog_train.json中最多的对话数，即max_doc_len的数目
            75:所有的对话语句中，最长的句子的单词数目，是预定义的
        output:
            emotion_pred=torch.Size([31, 7])
            cause_pred.shape=([496,2])
            维度是固定的
            496:       
            for end_t in range(max_doc_len):
                for t in range(end_t + 1):
            上面这两个循环产生了496个数据，max_doc_len=31
            (0,0),(1,0),(1,1),(2,0),(2,1),(2,2),...(30,30)    
                
            
        '''
        emotion_pred = self.emotion_classification_task(input_ids, attention_mask, token_type_ids)
        cause_pred = self.binary_cause_classification_task(emotion_pred, input_ids, attention_mask, token_type_ids, speaker_ids)
        return emotion_pred, cause_pred
    
    def forward2(self, input_ids, attention_mask, token_type_ids, speaker_ids):
        emotion_pred = self.emotion_classification_task(input_ids, attention_mask, token_type_ids)
        batch_size, max_doc_len, max_seq_len = input_ids.shape

        _, pooled_output = self.bert(input_ids=input_ids.view(-1, max_seq_len), attention_mask=attention_mask.view(-1, max_seq_len), token_type_ids=token_type_ids.view(-1, max_seq_len), return_dict=False)
        utterance_representation = self.dropout(pooled_output)
        concatenated_embedding = torch.cat((utterance_representation, emotion_pred, speaker_ids.view(-1).unsqueeze(1)), dim=1) # 여기서 emotion_prediction에 detach를 해야 문제가 안생기겠지? 해보고 문제생기면 detach 고고
        #utter_emb句子编码，形状：torch.Size([31, 776])，rel_emb_ k和V 形状：torch.Size([31, 31, 100])
        utter_emb,rel_emb_k,rel_emb_v = self.pag(concatenated_embedding)
        # logits = self.classifier(utter_emb,rel_emb_k,rel_emb_v,mask)
        #  emotion_pred.shape,rel_emb_k.shape,logits.shape：torch.Size([31, 7]) torch.Size([31, 31, 100]) torch.Size([31, 31])
        logits=torch.ones([496,2]).to(emotion_pred.device)
        return emotion_pred,logits

        

    #情感分类任务
    def emotion_classification_task(self, input_ids, attention_mask, token_type_ids):
        batch_size, max_doc_len, max_seq_len = input_ids.shape
        _, pooled_output = self.bert(input_ids=input_ids.view(-1, max_seq_len),
                                     attention_mask=attention_mask.view(-1, max_seq_len),
                                     token_type_ids=token_type_ids.view(-1, max_seq_len),
                                     return_dict=False)
        utterance_representation = self.dropout(pooled_output)
        return self.emotion_linear(utterance_representation)
    #二进制原因分类任务
    def binary_cause_classification_task(self, emotion_prediction, input_ids, attention_mask, token_type_ids, speaker_ids):
        '''
        torch.Size([31, 7]) torch.Size([1, 31, 75]) torch.Size([1, 31, 75]) torch.Size([1, 31, 75]) torch.Size([1, 31])
        '''
        #获得对编码
        pair_embedding = self.get_pair_embedding(emotion_prediction, input_ids, attention_mask, token_type_ids, speaker_ids)
        # pair_embedding.shape:[1, 496, 1552]
        #gating_prob.shape:torch.Size([496, 4])
        gating_prob = self.gating_network(pair_embedding.view(-1, pair_embedding.shape[-1]).detach())#门控网络
        # print("******************lin70",gating_prob.shape)
        gating_prob = self.guiding_lambda * self.get_subtask_label(input_ids, speaker_ids, emotion_prediction).view(-1, self.n_expert) + (1 - self.guiding_lambda) * gating_prob

        pred = []
        for _ in range(self.n_expert):
            expert_pred = self.cause_linear[_](pair_embedding.view(-1, pair_embedding.shape[-1]))
            # print("***************lin76",expert_pred.shape)
            expert_pred *= gating_prob.view(-1, self.n_expert)[:, _].unsqueeze(-1)
            #expert_pred:torch.Size([496, 2])
            # print("******************lin77",expert_pred.shape)
            pred.append(expert_pred)

        cause_pred = sum(pred)
        return cause_pred
    #门控网络训练
    # def gating_network_trainf(self, emotion_prediction, input_ids, attention_mask, token_type_ids, speaker_ids):
    #     pair_embedding = self.get_pair_embedding(emotion_prediction, input_ids, attention_mask, token_type_ids, speaker_ids)
    #     return self.gating_network(pair_embedding.view(-1, pair_embedding.shape[-1]).detach())
    #获取对编码
    def get_pair_embedding(self, emotion_prediction, input_ids, attention_mask, token_type_ids, speaker_ids):
        # 1,31,776
        batch_size, max_doc_len, max_seq_len = input_ids.shape

        _, pooled_output = self.bert(input_ids=input_ids.view(-1, max_seq_len), attention_mask=attention_mask.view(-1, max_seq_len), token_type_ids=token_type_ids.view(-1, max_seq_len), return_dict=False)
        utterance_representation = self.dropout(pooled_output)
        #torch.Size([31, 75]),max_seq:75
        # concatenated_embedding.shape=[31,776]
        concatenated_embedding = torch.cat((utterance_representation, emotion_prediction, speaker_ids.view(-1).unsqueeze(1)), dim=1) # 여기서 emotion_prediction에 detach를 해야 문제가 안생기겠지? 해보고 문제생기면 detach 고고
        #utterance_representation torch.Size([31, 768]),emotion_prediction torch.Size([31, 7]),speaker_ids torch.Size([31, 1])
        #utter_emb句子编码，形状：torch.Size([31, 776])，rel_emb_ k和V 形状：torch.Size([31, 31, 100])
        utter_emb,rel_emb_k,rel_emb_v = self.pag(concatenated_embedding)

        #逻辑搞明白，
        pair_embedding = list()
        for batch in utter_emb.view(batch_size, max_doc_len, -1):
            pair_per_batch = list()
            # 这儿是产生496的原因
            for end_t in range(max_doc_len):
                for t in range(end_t + 1):
                    pair_per_batch.append(torch.cat((batch[t], batch[end_t]))) # backward 시, cycle이 생겨 문제가 생길 경우, batch[end_t].detach() 시도.
            pair_embedding.append(torch.stack(pair_per_batch))
            # print("**********line123",len(pair_per_batch))
            #pair_embedding:[496, 1552]
        pair_embedding = torch.stack(pair_embedding).to(input_ids.device)
        # print("*********line125",pair_embedding.shape)

        
        return pair_embedding

    def get_subtask_label(self, input_ids, speaker_ids, emotion_prediction):
        # After Inheritance, Define function.
        pass
#PRG_MOE模型
class PRG_MoE(GuidedMoEBasic):
    def __init__(self, dropout=0.5, n_speaker=2, n_emotion=7, n_cause=2, n_expert=4, guiding_lambda=0, **kwargs):
        super().__init__(dropout=dropout, n_speaker=n_speaker, n_emotion=n_emotion, n_cause=n_cause, n_expert=4, guiding_lambda=guiding_lambda)
    #获取子任务标签
    def get_subtask_label(self, input_ids, speaker_ids, emotion_prediction):
        batch_size, max_doc_len, max_seq_len = input_ids.shape

        pair_info = []
        for speaker_batch, emotion_batch in zip(speaker_ids.view(batch_size, max_doc_len, -1), emotion_prediction.view(batch_size, max_doc_len, -1)):
            info_pair_per_batch = []
            for end_t in range(max_doc_len):
                for t in range(end_t + 1):
                    speaker_condition = speaker_batch[t] == speaker_batch[end_t]
                    emotion_condition = torch.argmax(emotion_batch[t]) == torch.argmax(emotion_batch[end_t])

                    if speaker_condition and emotion_condition:
                        info_pair_per_batch.append(torch.Tensor([1, 0, 0, 0])) # if speaker and dominant emotion are same
                    elif speaker_condition:
                        info_pair_per_batch.append(torch.Tensor([0, 1, 0, 0])) # if speaker is same, but dominant emotion is differnt
                    elif emotion_condition:
                        info_pair_per_batch.append(torch.Tensor([0, 0, 1, 0])) # if speaker is differnt, but dominant emotion is same
                    else:
                        info_pair_per_batch.append(torch.Tensor([0, 0, 0, 1])) # if speaker and dominant emotion are differnt
            pair_info.append(torch.stack(info_pair_per_batch))
        
        pair_info = torch.stack(pair_info).to(input_ids.device)

        return pair_info
