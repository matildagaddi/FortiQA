# main file to run all steps at once

### TODO: 
# # later: # figure out passing data from one step to next, would like to avoid reading files/calling dataloader again, but not sure what is most efficient. # could make arg usable as either data object or file path, maybe pull info from args files too.
# potentially fix batching? batches of 1?
# collect NER entity level masking accuracy %
# collect # of unmasked PII entities sent to cloud (out of total PII entities in test set)
# add more parameters to functions for ablation experiments and save to args file
# make functions more plug and play for ablation experiments
# ADD explanations to few shots for guidance example


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from stage1 import stage1
from stage2 import stage2
from utils import *
import argparse
import json

# # stage 0 mask
masker_args, masked_data, fail_success_split_qs, masked_data_path = mask(model_name="meta-llama/Llama-2-7b-hf", dataset="CyberNERQA", raw_data_path="data/CyberNERQA_raw/CyberNERQA.json", out_path="data/CyberNERQA_masked/", max_gen_len=50, batch_size=50, few_shot=1)

# stage 0.5 classify
#split and classify new data
cloud_data, edge_data, edge_indices, cloud_indices, mc_data_path = classify(fail_success_split_qs, dataset="CyberNERQA", masked_data_path=masked_data_path, train_split=0.5, fail_split='match')
# go in and add column "send_to_cloud" = 1 or 0

# stage 1 guide
guide_args, guidances, big_output_path = guide(edge_indices, cloud_indices, dataset="CyberNERQA", model_cloud="meta-llama/Llama-2-13b-hf", model_edge="meta-llama/Llama-2-7b-hf", mc_data_path=mc_data_path, masking=1)
#either new dataloader for _masked_classified OR make CyberNERQA loader know when to return CyberNERQA_masked_classified with new parameter? Which is better?
# parameters to add:
# - mask_override = None, "always", "never" (always use "masked question" or raw "question")
# - guide_override = None, "always cloud", "always edge", "some cloud", "some edge", "skip"
### allow to skip guidance entirely by not calling func?


# stage 2 answer
answer_args = answer(stage1_args=guide_args, dataset="CyberNERQA", mc_data_path=mc_data_path, model_name="meta-llama/Llama-2-7b-hf", big_output_path=big_output_path) #for later: guidances=guidances, or add to mc_data_path

print(answer_args)



# if __name__ == "__main__":
#     main()