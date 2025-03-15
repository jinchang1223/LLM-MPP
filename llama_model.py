from transformers import LlamaModel, AutoTokenizer, LlamaForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
import torch
import torch.nn as nn
from torch_geometric.nn import MLP, AttentiveFP

from configure import args

class PredictHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PredictHead, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, int(hidden_dim / 2))
        self.linear3 = nn.Linear(int(hidden_dim / 2), output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        return x
    

class CrossAttention(nn.Module):
    def __init__(self, query_dim, key_value_dim, num_heads, query_weight, key_value_weight):
        super(CrossAttention, self).__init__()
        self.query_dim = query_dim
        self.key_value_dim = key_value_dim
        self.num_heads = num_heads
        self.query_weight = query_weight
        self.key_value_weight = key_value_weight
        # adjust key/value dimension
        self.key_value_dim_adjust = nn.Linear(key_value_dim, query_dim)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=query_dim, num_heads=num_heads)

    def forward(self, query, key_value):
        # adjust key_value dimension
        key_value_adjusted = self.key_value_dim_adjust(key_value)
        # calculate cross attention
        attn_output, _ = self.multihead_attn(query=query, key=key_value_adjusted, value=key_value_adjusted)
        attn_output_weighted = self.query_weight * query + self.key_value_weight * key_value_adjusted + (1 - self.query_weight - self.key_value_weight) * attn_output
        return attn_output_weighted
    

class LoraLlamaWithEmbedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model_name = args.model_name
        self.model = LlamaForCausalLM.from_pretrained(
            self.model_name, 
            torch_dtype=torch.float16,
            output_hidden_states=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.lora_config = LoraConfig(
            # task_type=TaskType.CAUSAL_LM, 
            r=args.lora_rank, 
            lora_alpha=args.lora_alpha, 
            target_modules=["q_proj", "v_proj"],
            lora_dropout=args.lora_dropout,
        )
        print('parameter: ', sum(p.numel() for p in self.model.parameters()))
        if args.use_lora:
            self.peft_model = get_peft_model(self.model, self.lora_config)
        else:
            self.peft_model = self.model
            for param in self.peft_model.parameters():
                param.requires_grad = False
    
    
    def cot_process_generate(self, smiles_cot_prompt):
        # when cot, freeze all the parameters
        for param in self.model.parameters():
            param.requires_grad = False
    
        unique_text = []
        for i in range(len(smiles_cot_prompt)):
            smiles_cot_encoded_prompt = self.tokenizer(
                smiles_cot_prompt[i],
                truncation=True, 
                padding='max_length', 
                max_length=self.args.smiles_max_length, 
                return_tensors='pt'
            )
            
            generate_input = {
                "input_ids": smiles_cot_encoded_prompt['input_ids'].to(args.device),
                "attention_mask": smiles_cot_encoded_prompt['attention_mask'].to(args.device),
                "eos_token_id": self.tokenizer.eos_token_id,
                "bos_token_id": self.tokenizer.bos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "max_length": self.args.smiles_max_length+self.args.cot_max_length, 
                "early_stopping": True,  
                "num_return_sequences": 1,  
                "no_repeat_ngram_size": 2,  
                "temperature": 0.5,
            }
            generate_ids = self.model.generate(**generate_input)
            text = self.tokenizer.decode(generate_ids[0], skip_special_tokens=True)
            
            end_index = text.find(smiles_cot_prompt[i]) + len(smiles_cot_prompt[i])
            unique_text.append(text[end_index:])
        
        return unique_text


    def forward(self, prompt):
        prompt_encoded_prompt = self.tokenizer(
            prompt,
            truncation=True, 
            padding='max_length', 
            max_length=self.args.final_prompt_max_length, 
            return_tensors='pt'
        )
        
        outputs = self.peft_model(
            input_ids=prompt_encoded_prompt['input_ids'].to(args.device), 
            attention_mask=prompt_encoded_prompt['attention_mask'].to(args.device)
        )
        last_layer_embeddings = outputs.hidden_states[-1]  # outputs[0] 是模型的最后一层输出
        
        ## average pooling
        last_layer_embeddings = torch.mean(last_layer_embeddings, dim=1)  # [batch_size, hidden_size]
        
        return last_layer_embeddings
    
    
class TopModel(nn.Module):
    def __init__(self, args):
        super(TopModel, self).__init__()
        
        self.llm_model = LoraLlamaWithEmbedding(args)
        self.llm_model_output_size = 4096
        
        self.NodeFeatEmbed = MLP([args.atom_feature_size, args.attentionfp_input_size], dropout = args.gnn_dropout_ratio)

        self.ATFP = AttentiveFP(in_channels = args.attentionfp_input_size,
                                hidden_channels = args.attentionfp_hidden_size,
                                out_channels = args.attentionfp_output_size,
                                edge_dim = args.bond_feature_size,
                                num_layers = args.atom_layers,
                                num_timesteps = args.mol_layers,
                                dropout = args.gnn_dropout_ratio)
        

        self.linear = nn.Linear(self.llm_model_output_size, args.attentionfp_output_size)
        self.cross_attention = CrossAttention(
            query_dim=args.attentionfp_output_size, 
            key_value_dim=args.attentionfp_output_size, 
            num_heads=args.cross_attn_num_heads,
            query_weight=args.gnn_weight,
            key_value_weight=args.llm_weight
        )
        self.predict_head = PredictHead(
            input_dim=args.attentionfp_output_size, 
            hidden_dim=args.mlp_hidden_dim, 
            output_dim=args.num_labels
        )
    
    def cot_process_generate(self, smiles_cot_prompt):
        return self.llm_model.cot_process_generate(smiles_cot_prompt)
    
    def forward(self, prompt, atom_features_list, edge_index, edge_attr, batch):
        prompt_embd = self.llm_model(prompt)
        prompt_embd = self.linear(prompt_embd)
        
        # print('atom_features_list: ', atom_features_list)
        gnn_x = self.NodeFeatEmbed(atom_features_list.to(torch.float32))
        gnn_emb = self.ATFP(gnn_x, edge_index, edge_attr, batch)
        
        y_embd = self.cross_attention(gnn_emb, prompt_embd)
        # y_embd = torch.cat((prompt_embd, gnn_emb), dim=1)
        y_pred = self.predict_head(y_embd).unsqueeze(-1)
        
        return prompt_embd, gnn_emb, y_embd, y_pred
