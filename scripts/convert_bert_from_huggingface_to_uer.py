import torch
import argparse
import collections

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--input_model_path", type=str, default="pytorch_model.bin",
                        help=".")
parser.add_argument("--output_model_path", type=str, default="google_model.bin",
                        help=".")

args = parser.parse_args()
path = args.input_model_path

input_model = torch.load(args.input_model_path, map_location='cpu')

output_model = collections.OrderedDict()

output_model["embedding.word_embedding.weight"] = input_model["bert.embeddings.word_embeddings.weight"]
output_model["embedding.position_embedding.weight"] = input_model["bert.embeddings.position_embeddings.weight"]
output_model["embedding.segment_embedding.weight"] = torch.cat((torch.Tensor([[0]*768]), input_model["bert.embeddings.token_type_embeddings.weight"]), dim=0)
output_model["embedding.layer_norm.gamma"] = input_model["bert.embeddings.LayerNorm.weight"]
output_model["embedding.layer_norm.beta"] = input_model["bert.embeddings.LayerNorm.bias"]

for i in range(12):
    output_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.0.weight"] = input_model["bert.encoder.layer." + str(i) + ".attention.self.query.weight"]
    output_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.0.bias"] = input_model["bert.encoder.layer." + str(i) + ".attention.self.query.bias"]
    output_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.1.weight"] = input_model["bert.encoder.layer." + str(i) + ".attention.self.key.weight"]
    output_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.1.bias"] = input_model["bert.encoder.layer." + str(i) + ".attention.self.key.bias"]
    output_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.2.weight"] = input_model["bert.encoder.layer." + str(i) + ".attention.self.value.weight"]
    output_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.2.bias"] = input_model["bert.encoder.layer." + str(i) + ".attention.self.value.bias"]
    output_model["encoder.transformer." + str(i) + ".self_attn.final_linear.weight"] = input_model["bert.encoder.layer." + str(i) + ".attention.output.dense.weight"]
    output_model["encoder.transformer." + str(i) + ".self_attn.final_linear.bias"] = input_model["bert.encoder.layer." + str(i) + ".attention.output.dense.bias"]
    output_model["encoder.transformer." + str(i) + ".layer_norm_1.gamma"] = input_model["bert.encoder.layer." + str(i) + ".attention.output.LayerNorm.weight"]
    output_model["encoder.transformer." + str(i) + ".layer_norm_1.beta"] = input_model["bert.encoder.layer." + str(i) + ".attention.output.LayerNorm.bias"]
    output_model["encoder.transformer." + str(i) + ".feed_forward.linear_1.weight"] = input_model["bert.encoder.layer." + str(i) + ".intermediate.dense.weight"]
    output_model["encoder.transformer." + str(i) + ".feed_forward.linear_1.bias"] = input_model["bert.encoder.layer." + str(i) + ".intermediate.dense.bias"]
    output_model["encoder.transformer." + str(i) + ".feed_forward.linear_2.weight"] = input_model["bert.encoder.layer." + str(i) + ".output.dense.weight"]
    output_model["encoder.transformer." + str(i) + ".feed_forward.linear_2.bias"] = input_model["bert.encoder.layer." + str(i) + ".output.dense.bias"]
    output_model["encoder.transformer." + str(i) + ".layer_norm_2.gamma"] = input_model["bert.encoder.layer." + str(i) + ".output.LayerNorm.weight"]
    output_model["encoder.transformer." + str(i) + ".layer_norm_2.beta"] = input_model["bert.encoder.layer." + str(i) + ".output.LayerNorm.bias"]

output_model["target.nsp_linear_1.weight"] = input_model["bert.pooler.dense.weight"]
output_model["target.nsp_linear_1.bias"] = input_model["bert.pooler.dense.bias"]
output_model["target.nsp_linear_2.weight"] = input_model["cls.seq_relationship.weight"]
output_model["target.nsp_linear_2.bias"] = input_model["cls.seq_relationship.bias"]
output_model["target.mlm_linear_1.weight"] = input_model["cls.predictions.transform.dense.weight"]
output_model["target.mlm_linear_1.bias"] = input_model["cls.predictions.transform.dense.bias"]
output_model["target.layer_norm.gamma"] = input_model["cls.predictions.transform.LayerNorm.weight"]
output_model["target.layer_norm.beta"] = input_model["cls.predictions.transform.LayerNorm.bias"]
output_model["target.mlm_linear_2.weight"] = input_model["cls.predictions.decoder.weight"]
output_model["target.mlm_linear_2.bias"] = input_model["cls.predictions.bias"]



# p = torch.load(path)
# for name in p.keys():
#     print(name, p[name].size())
# print(len(p))
# print("______________")

# namelist = []
# for name in p.keys():
#     namelist.append(name)
# for name in namelist:
#     if name == "bert.embeddings.word_embeddings.weight":
#         p['embedding.word_embedding.weight'] = p[name] # [21128, 768]
#         #print(p[name])
#     if name == "bert.embeddings.position_embeddings.weight":
#         p['embedding.position_embedding.weight'] = p[name] # [512, 768]
#         #print(p[name])
#     if name == "bert.embeddings.token_type_embeddings.weight":
#         z=torch.Tensor([[0]*768])
#         p['embedding.segment_embedding.weight'] = torch.cat((z,p[name]) ,dim=0)# [2, 768]
#         #print(p[name])
#     if name == "bert.embeddings.LayerNorm.weight":
#         p['embedding.layer_norm.gamma'] = p[name]
#     if name == "bert.embeddings.LayerNorm.bias":
#         p['embedding.layer_norm.beta'] = p[name]


#     if 'encoder.layer' in name:
#         name_new = 'encoder.transformer.' + str(name[19])
#         if name[20] != '.':
#             name_new += str(name[20])
#         name_new += '.'
#         if 'query' in name:
#             if 'weight' in name:
#                 name_new += 'self_attn.linear_layers.0.weight'
#             else:
#                 name_new += 'self_attn.linear_layers.0.bias'
#         if 'key' in name:
#             if 'weight' in name:
#                 name_new += 'self_attn.linear_layers.1.weight'
#             else:
#                 name_new += 'self_attn.linear_layers.1.bias'
#         if 'value' in name:
#             if 'weight' in name:
#                 name_new += 'self_attn.linear_layers.2.weight'
#             else:
#                 name_new += 'self_attn.linear_layers.2.bias'
#         if 'attention.output' in name:
#             if 'LayerNorm.weight' in name:
#                 name_new += 'layer_norm_1.gamma'
#             if 'LayerNorm.bias' in name:
#                 name_new += 'layer_norm_1.beta'
#             if 'dense.weight' in name:
#                 name_new += 'self_attn.final_linear.weight'
#             if 'dense.bias' in name:
#                 name_new += 'self_attn.final_linear.bias'
#         else:
#             if 'output.dense.weight' in name:
#                 name_new += 'feed_forward.linear_2.weight'
#             if 'output.dense.bias' in name:
#                 name_new += 'feed_forward.linear_2.bias'
#             if 'intermediate.dense.weight' in name:
#                 name_new += 'feed_forward.linear_1.weight'
#             if 'intermediate.dense.bias' in name:
#                 name_new += 'feed_forward.linear_1.bias'
#             if 'LayerNorm.weight' in name:
#                 name_new += 'layer_norm_2.gamma'
#             if 'LayerNorm.bias' in name:
#                 name_new += 'layer_norm_2.beta'
            
#         p[name_new] = p[name]

#     if name == 'bert.pooler.dense.weight':
#         p['target.nsp_linear_1.weight'] = p[name]
#     if name == 'bert.pooler.dense.bias':
#         p['target.nsp_linear_1.bias'] = p[name]

#     if name == 'cls.predictions.decoder.weight':
#         p['target.mlm_linear_2.weight'] = p[name]
#     if name == 'cls.predictions.bias':
#         p['target.mlm_linear_2.bias'] = p[name]
    
#     if name == 'cls.predictions.transform.dense.weight':
#         p['target.mlm_linear_1.weight'] = p[name]
#     if name == 'cls.predictions.transform.dense.bias':
#         p['target.mlm_linear_1.bias'] = p[name]
    
#     if name == 'cls.predictions.transform.LayerNorm.weight':
#         p['target.layer_norm.gamma'] = p[name]
#     if name == 'cls.predictions.transform.LayerNorm.bias':
#         p['target.layer_norm.beta'] = p[name]
    
#     if name == 'cls.seq_relationship.weight':
#         p['target.nsp_linear_2.weight'] = p[name]
#     if name == 'cls.seq_relationship.bias':
#         p['target.nsp_linear_2.bias'] = p[name]
    
#     del p[name]

torch.save(output_model, args.output_model_path)
