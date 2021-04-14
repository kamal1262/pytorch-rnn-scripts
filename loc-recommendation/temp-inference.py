
import os
import json
import logging
import numpy as np

import gzip
import pickle
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

JSON_CONTENT_TYPE = 'application/json'

logger = logging.getLogger(__name__)

class SkipGram(nn.Module):

    def __init__(self, emb_size, emb_dim):
        super().__init__()
        self.emb_size = emb_size
        self.emb_dim = emb_dim
        self.center_embeddings = nn.Embedding(emb_size, emb_dim, sparse=True)
        self.context_embeddings = nn.Embedding(emb_size, emb_dim, sparse=True)
        self.init_emb()

    def init_emb(self):
        """
        Init embeddings like word2vec
        Center embeddings have uniform distribution in [-0.5/emb_dim , 0.5/emb_dim].
        Context embeddings are initialized with 0s.
        Returns:
        """
        emb_range = 0.5 / self.emb_dim

        # Initializing embeddings:
        # https://stackoverflow.com/questions/55276504/different-methods-for-initializing-embedding-layer-weights-in-pytorch
        self.center_embeddings.weight.data.uniform_(-emb_range, emb_range)
        self.context_embeddings.weight.data.uniform_(0, 0)

    def forward(self, center, context, neg_context):
        """
        Args:
            center: List of center words
            context: List of context words
            neg_context: List of list of negative context words
        Returns:
        """
        # Calculate positive score
        emb_center = self.center_embeddings(center)  # Get embeddings for center word
        emb_context = self.context_embeddings(context)  # Get embeddings for context word
        emb_neg_context = self.context_embeddings(neg_context)  # Get embeddings for negative context words

        # Next two lines equivalent to torch.dot(emb_center, emb_context) but for batch
        score = torch.mul(emb_center, emb_context)  # Get dot product (part 1)
        score = torch.sum(score, dim=1)  # Get dot product (part2)
        score = torch.clamp(score, max=10, min=-10)
        score = -F.logsigmoid(score)  # Get score for the positive pairs

        # Calculate negative score (for negative samples)
        neg_score = torch.bmm(emb_neg_context, emb_center.unsqueeze(2)).squeeze()  # Get dot product
        neg_score = torch.clamp(neg_score, max=10, min=-10)
        neg_score = -torch.sum(F.logsigmoid(-neg_score), dim=1)

        # Return combined score
        return torch.mean(score + neg_score)

    def get_center_emb(self, center):
        return self.center_embeddings(center)

    def get_embeddings(self):
        return self.center_embeddings.weight.cpu().data.numpy()

    def save_embeddings(self, file_name):
        embedding = self.center_embeddings.weight.cpu().data.numpy()
        np.save(file_name, embedding)

# kamal will print now

print("========================================================================")
print(" Did we managed to import scipy module, just checking !!!")
from scipy.spatial import distance
print("========================================================================")
print(" if successful, it should not have any error!")


def model_fn(model_dir):
    logger.info('Loading the model.')
    model_checkpoint = {}
    with open(os.path.join(model_dir, 'model_checkpoint.pth'), 'rb') as f:
        model_checkpoint = torch.load(f)
    print('model_checkpointel_info: {}'.format(len(model_checkpoint)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info('Current device: {}'.format(device))

    VOCAB_SIZE = 628
    EMB_SIZE = 128

    model = SkipGram(VOCAB_SIZE, EMB_SIZE).to(device)
    # model = RNNModel(rnn_type=model_info['rnn_type'], ntoken=model_info['ntoken'],
    #                  ninp=model_info['ninp'], nhid=model_info['nhid'], nlayers=model_info['nlayers'],
    #                  dropout=model_info['dropout'], tie_weights=model_info['tie_weights'])

    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))
        # after load the rnn params are not a continuous chunk of memory
        # this makes them a continuous chunk, and will speed up forward pass
        # model.rnn.flatten_parameters()
    model.to(device).eval()
    logger.info('Loading the data.')
    # corpus = data.Corpus(model_dir)
    loc2idx = model_checkpoint['loc2idx']
    idx2loc = model_checkpoint['idx2loc']

    logger.info('Done loading model and corpus. loc2idx dictionary size: {}'.format(len(loc2idx)))
    return {'model': model, 'loc2idx': loc2idx, 'idx2loc': idx2loc}


def input_fn(serialized_input_data, content_type=JSON_CONTENT_TYPE):
    logger.info('Deserializing the input data.')
    if content_type == JSON_CONTENT_TYPE:
        input_data = json.loads(serialized_input_data)
        params = {}
        # params['locationIDInput'] = [ loc2id_dict[input_d] for input_d in input_data['locationIDInput'] ]
        params['locationIDInput'] = [ input_d for input_d in input_data['locationIDInput'] ]
        params['count'] = input_data['count'] if 'count' in input_data else 5;
        
        logger.info(f'Input params: {params}')
        return params


def predict_fn(input_data, model_artifact):
    logger.info(f'model artifact: {model_artifact}')
    model = model_artifact['model']
    loc2idx = model_artifact['loc2idx']
    idx2loc = model_artifact['idx2loc']

    with torch.no_grad():
        embeddings = model.get_embeddings()
    searched_idx = loc2idx[input_data['locationIDInput'][-1]]
    center_embeddings = embeddings[searched_idx]
    
    closest_index = distance.cdist([center_embeddings], embeddings, "cosine")[0]
    result = zip(range(len(closest_index)), closest_index)
    result = sorted(result, key=lambda x: x[1])
    
    res = []
    for idx, score in result[ :input_data['count']+1 ]:
        tmp_dict = { "id": idx, "score": score }
        res.append(tmp_dict)

    print(f"preparing the prediction results: {res}")

    for idx, pred in enumerate(res):
        res[idx]["global_id"] = idx2loc[pred["id"]]
    return res


def output_fn(prediction_output, accept=JSON_CONTENT_TYPE):
    logger.info('Serializing the generated output.')
    if accept == JSON_CONTENT_TYPE:
        logger.info(f'prediction_output : {prediction_output}')
        return json.dumps(prediction_output), accept
    raise Exception('Requested unsupported ContentType in Accept: ' + accept)



# if __name__ == "__main__":
    
#     input_sample = """{"locationIDInput": ["mysta_25733"], "count": 10}"""
#     request_content_type = "application/json"
#     response_content_type = "application/json"
# #     # model_dir = "models/"
#     model_dir = '/Users/md.kamal/work-code-sample/temp/test-location-recommendation/artifact/'

# # #     with open(f"{model_dir}/dict_loc.pickle", 'rb') as f:
# # #         dict_loc = pickle.load(f)

# # #     loc2id_dict = {w: idx for (idx, w) in enumerate(dict_loc)}
# # #     id2loc_dict = {idx: w for (idx, w) in enumerate(dict_loc)}
 

#     input_obj = input_fn(input_sample, request_content_type)
#     model = model_fn(model_dir)
#     prediction = predict_fn(input_obj, model)
#     output = output_fn(prediction, response_content_type)
    
#     print(output)