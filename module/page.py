import torch
import torch.nn as nn
from torch_geometric.nn.conv.rgcn_conv import RGCNConv


    
class PaG(nn.Module):
    def __init__(self,window,utter_dim,num_bases,max_len,posi_dim):
        super(PaG, self).__init__()
        self.max_len = max_len
        self.posi_dim = posi_dim
        self.pe_k = nn.Embedding(max_len+1, posi_dim, padding_idx=0)
        self.pe_v = nn.Embedding(max_len+1, posi_dim, padding_idx=0)
        self.window = window
        self.rel_num = self.window + 2
        self.rgcn = RGCNConv(utter_dim,utter_dim,self.rel_num,num_bases=num_bases)

    
    def forward(self,x):
        x_dim = x.shape[1]#776
        slen = x.shape[0]#31

        #slen是句子的长度，src_pos和tgt_pos分别表示源语言和目标语言的位置编码。
        # unsqueeze()函数用于在指定的维度上添加一个维度。
        # unsqueeze(0)将src_pos的维度从(句子长度,)变为(1, 句子长度)
        # unsqueeze(1)将tgt_pos的维度从(句子长度,)变为(句子长度, 1)。
        src_pos = torch.arange(slen).unsqueeze(0)
        tgt_pos = torch.arange(slen).unsqueeze(1)
        pos_mask = (tgt_pos - src_pos) + 1#pos_mask形状;torch.Size([31, 31])
        pos_mask = pos_mask.to(x.device)
        #裁剪 pos_mask 张量的值，将其限制在范围 [0, self.max_len] 内，并将结果转换为整数类型。
        position_mask = torch.clamp(pos_mask, min=0, max=self.max_len).long()#torch.Size([31, 31])
        rel_emb_k = self.pe_k(position_mask)
        rel_emb_v = self.pe_v(position_mask)
        
        rel_emb_k = rel_emb_k.expand( slen, slen, self.posi_dim)
        rel_emb_v = rel_emb_v.expand( slen, slen, self.posi_dim)
        rel_adj = (src_pos - tgt_pos).to(x.device)
        
        self.rgcn.to(x.device)

        rel_adj = rel_adj_create(rel_adj,slen,self.window)
        index = index_create(slen).to(x.device)
        
        edge_type = torch.flatten(rel_adj).long().to(x.device)

        # out = self.rgcn(x[0],index,edge_type)
        # for i in range(1,batch_size):
        #     h = self.rgcn(x[i],index,edge_type)
        #     out = torch.cat((out,h.unsqueeze(0)),dim=0)
        # print("***************line50",x.shape,index.shape,edge_type.shape)

        # x=torch.ones([200,768]).to(x.device)
        # print("**************line53,x.shape",x.shape)
        #out代表句子编码，维度：torch.Size([31, 776])
        # rel_emb_k维度 torch.Size([31, 31, 100])
        # rel_emb_v维度 torch.Size([31, 31, 100])
        
        out=self.rgcn(x,index,edge_type)
        # print("***********lin55 out,rel_emb_k,rel_emb_v",out.shape,rel_emb_k.shape,rel_emb_v.shape)
            
        return out,rel_emb_k,rel_emb_v
    
class Causal_Classifier(nn.Module):
    def __init__(self, input_dim, mlp_dim, mlp_dropout=0.1):
        super(Causal_Classifier, self).__init__()
        self.input_dim = input_dim
        self.mlp_dim = mlp_dim
        self.mlp_dropout = mlp_dropout

        self.mlp = nn.Sequential(nn.Linear(2*input_dim+200, mlp_dim, False), nn.ReLU(), nn.Dropout(mlp_dropout),
                                 nn.Linear(mlp_dim, mlp_dim, False), nn.ReLU(), nn.Dropout(mlp_dropout))
        self.predictor_weight = nn.Linear(mlp_dim, 1, False)

    def forward(self, x,rel_emb_k,rel_emb_v):

        '''
        torch.Size([31, 31, 100]) torch.Size([31, 31])
        '''
        # batch_size = x.shape[0]
        x_dim = x.shape[1]
        slen = x.shape[0]

        # x_source = x.unsqueeze(1).expand( slen, slen, x_dim)
        # x_target = x.unsqueeze(2).expand(slen, slen, x_dim)
        x_source = x.expand( slen, slen, x_dim)
        x_target = x.expand(slen, slen, x_dim)


        x_source = torch.cat([x_source,rel_emb_k],dim=-1)
        x_target = torch.cat([x_target,rel_emb_v],dim=-1)
        x_cat = torch.cat([x_source, x_target], dim=-1)  


        predict_score = self.predictor_weight(self.mlp(x_cat)).squeeze(-1)
        predict_score = torch.sigmoid(predict_score) 
        # predict_score = torch.sigmoid(predict_score) * mask
       
        return predict_score


def rel_adj_create(rel_adj,slen,window):
    for i in range(slen):
        for s in range(i+1,slen):
            rel_adj[i][s] = 1
    
    for i in range(slen):
        num = 1     
        for o in range(i-1,-1,-2):
            if((o-1)<0):
                rel_adj[i][o] = -num
            else:
                rel_adj[i][o] = -num
                rel_adj[i][o-1] = -num
            num+=1
    
    for i in range(slen):
        for o in range(i-1,-1,-1):
            if(rel_adj[i][o]<-(window+1)):
                rel_adj[i][o] = - (window + 1) 
    
    return rel_adj

def index_create(slen):
    index = []
    start = []
    end = []     
    
    for i in range(0,slen):
        for j in range(0,slen):
            start.append(i)
    for i in range(0,slen):
        for j in range(0,slen):
            end.append(j)

    index.append(start)
    index.append(end)
    
    index = torch.tensor(index).long()
    
    return index
 



    
