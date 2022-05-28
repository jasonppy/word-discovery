# Author: David Harwath
import argparse
import os
import pickle
import time
from steps import trainer
from models import audio_encoder, dual_encoder
from datasets import spokencoco_dataset
from logging import getLogger

logger = getLogger(__name__)
logger.info("I am process %s, running on %s: starting (%s)" % (
        os.getpid(), os.uname()[1], time.asctime()))

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--resume", action="store_true", dest="resume", help="load from exp_dir if True")
parser.add_argument("--validate", action="store_true", default=False, help="temp, if call trainer_variants rather than trainer")


# trainer.Trainer.add_args(parser) # trainer_variants' args contains args of trainer
trainer.Trainer.add_args(parser)
audio_encoder.AudioEncoder.add_args(parser) # it will also contains hubert args
dual_encoder.DualEncoder.add_args(parser)
spokencoco_dataset.ImageCaptionDataset.add_args(parser)
args = parser.parse_args()

os.makedirs(args.exp_dir, exist_ok=True)

if args.resume:
    resume = args.resume
    assert(bool(args.exp_dir))
    with open("%s/args.pkl" % args.exp_dir, "rb") as f:
        old_args = pickle.load(f)
    new_args = vars(args)
    old_args = vars(old_args)
    for key in new_args:
        if key not in old_args or old_args[key] != new_args[key]:
            old_args[key] = new_args[key]
    args = argparse.Namespace(**old_args)
    args.resume = resume
else:
    print("\nexp_dir: %s" % args.exp_dir)
    with open("%s/args.pkl" % args.exp_dir, "wb") as f:
        pickle.dump(args, f)
args.places = False
logger.info(args)


if args.validate:
    my_trainer = trainer.Trainer(args)
    my_trainer.validate(hide_progress=False)
else:
    my_trainer = trainer.Trainer(args)
    my_trainer.train()

# import torch
# # old_weights_fn = "/data1/scratch/exp_pyp/TransDAVEnet/plan_b/0_01_1_coco_correct/best_bundle.pth"
# old_weights_fn = "/data1/scratch/exp_pyp/TransDAVEnet/plan_b/0_01_1/best_bundle.pth"
# old_sd = torch.load(old_weights_fn)
# new_sd = {"dual_encoder": {}, "cross_encoder": {}}

# print("convert dual encoder")
# for key in old_sd['model']:
#     if 'cls_token' == key:
#         new_sd['dual_encoder']['visual_cls_token'] = old_sd['model'][key]
#     elif 'visn_fc.' in key:
#         new_sd['dual_encoder'][key] = old_sd['model'][key]
#     elif 'w2v2_model.' in key:
#         new_key = "conv1_trm1_trm3." + ".".join(key.split(".")[1:])
#         new_sd['dual_encoder'][new_key] = old_sd['model'][key]
#     elif 'audio_convnet.' in key:
#         new_key = "conv2." + ".".join(key.split(".")[1:])
#         new_sd['dual_encoder'][new_key] = old_sd['model'][key]
#     elif 'r_layers.' in key:
#         new_key = "trm." + ".".join(key.split(".")[1:])
#         new_sd['dual_encoder'][new_key] = old_sd['model'][key]
#     elif 'audio_cls_token_proj' in key:
#         new_sd['dual_encoder'][key] = old_sd['model'][key]
#     elif 'visual_cls_token_proj_coarse.' in key:
#         new_sd['dual_encoder'][key] = old_sd['model'][key]
#     elif 'attn_after_res.' in key:
#         new_key = "trm2." + ".".join(key.split(".")[1:])
#         new_sd['dual_encoder'][new_key] = old_sd['model'][key]
#     elif 'attn_after_res_proj.' in key:
#         new_key = "trm2_proj." + ".".join(key.split(".")[1:])
#         new_sd['dual_encoder'][new_key] = old_sd['model'][key]
#     else:
#         print(key)

# print("convert cross encoder")
# for key in old_sd['cross_model']:
#     if 'x_layers.' in key:
#         new_key = "xtrm." + ".".join(key.split(".")[1:])
#         new_sd['cross_encoder'][new_key] = old_sd['cross_model'][key]
#     elif 'fc.' in key:
#         new_sd['cross_encoder'][key] = old_sd['cross_model'][key]
#     else:
#         print(key)

# torch.save(new_sd, "/data1/scratch/exp_pyp/fast-vgs-and-friends/fast-vgs-places/best_bundle.pth", _use_new_zipfile_serialization=False)
    
    
# # original names:
# cls_token
# visn_fc.visn_fc.weight
# visn_fc.visn_fc.bias
# visn_fc.visn_layer_norm.weight
# visn_fc.visn_layer_norm.bias
# visn_fc.box_fc.weight
# visn_fc.box_fc.bias
# visn_fc.box_layer_norm.weight
# visn_fc.box_layer_norm.bias
# w2v2_model.mask_emb
# w2v2_model.cls_token
# w2v2_model.feature_extractor.conv_layers.0.0.weight
# w2v2_model.feature_extractor.conv_layers.0.2.weight
# w2v2_model.feature_extractor.conv_layers.0.2.bias
# w2v2_model.feature_extractor.conv_layers.1.0.weight
# w2v2_model.feature_extractor.conv_layers.2.0.weight
# w2v2_model.feature_extractor.conv_layers.3.0.weight
# w2v2_model.feature_extractor.conv_layers.4.0.weight
# w2v2_model.feature_extractor.conv_layers.5.0.weight
# w2v2_model.feature_extractor.conv_layers.6.0.weight
# w2v2_model.post_extract_proj.weight
# w2v2_model.post_extract_proj.bias
# w2v2_model.quantizer.vars
# w2v2_model.quantizer.weight_proj.weight
# w2v2_model.quantizer.weight_proj.bias
# w2v2_model.project_q.weight
# w2v2_model.project_q.bias
# w2v2_model.encoder.pos_conv.0.bias
# w2v2_model.encoder.pos_conv.0.weight_g
# w2v2_model.encoder.pos_conv.0.weight_v
# w2v2_model.encoder.layers.0.self_attn.k_proj.weight
# w2v2_model.encoder.layers.0.self_attn.k_proj.bias
# w2v2_model.encoder.layers.0.self_attn.v_proj.weight
# w2v2_model.encoder.layers.0.self_attn.v_proj.bias
# w2v2_model.encoder.layers.0.self_attn.q_proj.weight
# w2v2_model.encoder.layers.0.self_attn.q_proj.bias
# w2v2_model.encoder.layers.0.self_attn.out_proj.weight
# w2v2_model.encoder.layers.0.self_attn.out_proj.bias
# w2v2_model.encoder.layers.0.self_attn_layer_norm.weight
# w2v2_model.encoder.layers.0.self_attn_layer_norm.bias
# w2v2_model.encoder.layers.0.fc1.weight
# w2v2_model.encoder.layers.0.fc1.bias
# w2v2_model.encoder.layers.0.fc2.weight
# w2v2_model.encoder.layers.0.fc2.bias
# w2v2_model.encoder.layers.0.final_layer_norm.weight
# w2v2_model.encoder.layers.0.final_layer_norm.bias
# w2v2_model.encoder.layers.1.self_attn.k_proj.weight
# w2v2_model.encoder.layers.1.self_attn.k_proj.bias
# w2v2_model.encoder.layers.1.self_attn.v_proj.weight
# w2v2_model.encoder.layers.1.self_attn.v_proj.bias
# w2v2_model.encoder.layers.1.self_attn.q_proj.weight
# w2v2_model.encoder.layers.1.self_attn.q_proj.bias
# w2v2_model.encoder.layers.1.self_attn.out_proj.weight
# w2v2_model.encoder.layers.1.self_attn.out_proj.bias
# w2v2_model.encoder.layers.1.self_attn_layer_norm.weight
# w2v2_model.encoder.layers.1.self_attn_layer_norm.bias
# w2v2_model.encoder.layers.1.fc1.weight
# w2v2_model.encoder.layers.1.fc1.bias
# w2v2_model.encoder.layers.1.fc2.weight
# w2v2_model.encoder.layers.1.fc2.bias
# w2v2_model.encoder.layers.1.final_layer_norm.weight
# w2v2_model.encoder.layers.1.final_layer_norm.bias
# w2v2_model.encoder.layers.2.self_attn.k_proj.weight
# w2v2_model.encoder.layers.2.self_attn.k_proj.bias
# w2v2_model.encoder.layers.2.self_attn.v_proj.weight
# w2v2_model.encoder.layers.2.self_attn.v_proj.bias
# w2v2_model.encoder.layers.2.self_attn.q_proj.weight
# w2v2_model.encoder.layers.2.self_attn.q_proj.bias
# w2v2_model.encoder.layers.2.self_attn.out_proj.weight
# w2v2_model.encoder.layers.2.self_attn.out_proj.bias
# w2v2_model.encoder.layers.2.self_attn_layer_norm.weight
# w2v2_model.encoder.layers.2.self_attn_layer_norm.bias
# w2v2_model.encoder.layers.2.fc1.weight
# w2v2_model.encoder.layers.2.fc1.bias
# w2v2_model.encoder.layers.2.fc2.weight
# w2v2_model.encoder.layers.2.fc2.bias
# w2v2_model.encoder.layers.2.final_layer_norm.weight
# w2v2_model.encoder.layers.2.final_layer_norm.bias
# w2v2_model.encoder.layers.3.self_attn.k_proj.weight
# w2v2_model.encoder.layers.3.self_attn.k_proj.bias
# w2v2_model.encoder.layers.3.self_attn.v_proj.weight
# w2v2_model.encoder.layers.3.self_attn.v_proj.bias
# w2v2_model.encoder.layers.3.self_attn.q_proj.weight
# w2v2_model.encoder.layers.3.self_attn.q_proj.bias
# w2v2_model.encoder.layers.3.self_attn.out_proj.weight
# w2v2_model.encoder.layers.3.self_attn.out_proj.bias
# w2v2_model.encoder.layers.3.self_attn_layer_norm.weight
# w2v2_model.encoder.layers.3.self_attn_layer_norm.bias
# w2v2_model.encoder.layers.3.fc1.weight
# w2v2_model.encoder.layers.3.fc1.bias
# w2v2_model.encoder.layers.3.fc2.weight
# w2v2_model.encoder.layers.3.fc2.bias
# w2v2_model.encoder.layers.3.final_layer_norm.weight
# w2v2_model.encoder.layers.3.final_layer_norm.bias
# w2v2_model.encoder.layers.4.self_attn.k_proj.weight
# w2v2_model.encoder.layers.4.self_attn.k_proj.bias
# w2v2_model.encoder.layers.4.self_attn.v_proj.weight
# w2v2_model.encoder.layers.4.self_attn.v_proj.bias
# w2v2_model.encoder.layers.4.self_attn.q_proj.weight
# w2v2_model.encoder.layers.4.self_attn.q_proj.bias
# w2v2_model.encoder.layers.4.self_attn.out_proj.weight
# w2v2_model.encoder.layers.4.self_attn.out_proj.bias
# w2v2_model.encoder.layers.4.self_attn_layer_norm.weight
# w2v2_model.encoder.layers.4.self_attn_layer_norm.bias
# w2v2_model.encoder.layers.4.fc1.weight
# w2v2_model.encoder.layers.4.fc1.bias
# w2v2_model.encoder.layers.4.fc2.weight
# w2v2_model.encoder.layers.4.fc2.bias
# w2v2_model.encoder.layers.4.final_layer_norm.weight
# w2v2_model.encoder.layers.4.final_layer_norm.bias
# w2v2_model.encoder.layers.5.self_attn.k_proj.weight
# w2v2_model.encoder.layers.5.self_attn.k_proj.bias
# w2v2_model.encoder.layers.5.self_attn.v_proj.weight
# w2v2_model.encoder.layers.5.self_attn.v_proj.bias
# w2v2_model.encoder.layers.5.self_attn.q_proj.weight
# w2v2_model.encoder.layers.5.self_attn.q_proj.bias
# w2v2_model.encoder.layers.5.self_attn.out_proj.weight
# w2v2_model.encoder.layers.5.self_attn.out_proj.bias
# w2v2_model.encoder.layers.5.self_attn_layer_norm.weight
# w2v2_model.encoder.layers.5.self_attn_layer_norm.bias
# w2v2_model.encoder.layers.5.fc1.weight
# w2v2_model.encoder.layers.5.fc1.bias
# w2v2_model.encoder.layers.5.fc2.weight
# w2v2_model.encoder.layers.5.fc2.bias
# w2v2_model.encoder.layers.5.final_layer_norm.weight
# w2v2_model.encoder.layers.5.final_layer_norm.bias
# w2v2_model.encoder.layers.6.self_attn.k_proj.weight
# w2v2_model.encoder.layers.6.self_attn.k_proj.bias
# w2v2_model.encoder.layers.6.self_attn.v_proj.weight
# w2v2_model.encoder.layers.6.self_attn.v_proj.bias
# w2v2_model.encoder.layers.6.self_attn.q_proj.weight
# w2v2_model.encoder.layers.6.self_attn.q_proj.bias
# w2v2_model.encoder.layers.6.self_attn.out_proj.weight
# w2v2_model.encoder.layers.6.self_attn.out_proj.bias
# w2v2_model.encoder.layers.6.self_attn_layer_norm.weight
# w2v2_model.encoder.layers.6.self_attn_layer_norm.bias
# w2v2_model.encoder.layers.6.fc1.weight
# w2v2_model.encoder.layers.6.fc1.bias
# w2v2_model.encoder.layers.6.fc2.weight
# w2v2_model.encoder.layers.6.fc2.bias
# w2v2_model.encoder.layers.6.final_layer_norm.weight
# w2v2_model.encoder.layers.6.final_layer_norm.bias
# w2v2_model.encoder.layers.7.self_attn.k_proj.weight
# w2v2_model.encoder.layers.7.self_attn.k_proj.bias
# w2v2_model.encoder.layers.7.self_attn.v_proj.weight
# w2v2_model.encoder.layers.7.self_attn.v_proj.bias
# w2v2_model.encoder.layers.7.self_attn.q_proj.weight
# w2v2_model.encoder.layers.7.self_attn.q_proj.bias
# w2v2_model.encoder.layers.7.self_attn.out_proj.weight
# w2v2_model.encoder.layers.7.self_attn.out_proj.bias
# w2v2_model.encoder.layers.7.self_attn_layer_norm.weight
# w2v2_model.encoder.layers.7.self_attn_layer_norm.bias
# w2v2_model.encoder.layers.7.fc1.weight
# w2v2_model.encoder.layers.7.fc1.bias
# w2v2_model.encoder.layers.7.fc2.weight
# w2v2_model.encoder.layers.7.fc2.bias
# w2v2_model.encoder.layers.7.final_layer_norm.weight
# w2v2_model.encoder.layers.7.final_layer_norm.bias
# w2v2_model.encoder.layer_norm.weight
# w2v2_model.encoder.layer_norm.bias
# w2v2_model.layer_norm.weight
# w2v2_model.layer_norm.bias
# w2v2_model.final_proj.weight
# w2v2_model.final_proj.bias
# audio_convnet.linear.weight
# audio_convnet.linear.bias
# audio_convnet.bn1.weight
# audio_convnet.bn1.bias
# audio_convnet.layer1.0.conv1.weight
# audio_convnet.layer1.0.bn1.weight
# audio_convnet.layer1.0.bn1.bias
# audio_convnet.layer1.0.conv2.weight
# audio_convnet.layer1.0.bn2.weight
# audio_convnet.layer1.0.bn2.bias
# audio_convnet.layer1.0.downsample.0.weight
# audio_convnet.layer1.0.downsample.1.weight
# audio_convnet.layer1.0.downsample.1.bias
# audio_convnet.layer1.1.conv1.weight
# audio_convnet.layer1.1.bn1.weight
# audio_convnet.layer1.1.bn1.bias
# audio_convnet.layer1.1.conv2.weight
# audio_convnet.layer1.1.bn2.weight
# audio_convnet.layer1.1.bn2.bias
# audio_convnet.layer2.0.conv1.weight
# audio_convnet.layer2.0.bn1.weight
# audio_convnet.layer2.0.bn1.bias
# audio_convnet.layer2.0.conv2.weight
# audio_convnet.layer2.0.bn2.weight
# audio_convnet.layer2.0.bn2.bias
# audio_convnet.layer2.0.downsample.0.weight
# audio_convnet.layer2.0.downsample.1.weight
# audio_convnet.layer2.0.downsample.1.bias
# audio_convnet.layer2.1.conv1.weight
# audio_convnet.layer2.1.bn1.weight
# audio_convnet.layer2.1.bn1.bias
# audio_convnet.layer2.1.conv2.weight
# audio_convnet.layer2.1.bn2.weight
# audio_convnet.layer2.1.bn2.bias
# audio_convnet.layer3.0.conv1.weight
# audio_convnet.layer3.0.bn1.weight
# audio_convnet.layer3.0.bn1.bias
# audio_convnet.layer3.0.conv2.weight
# audio_convnet.layer3.0.bn2.weight
# audio_convnet.layer3.0.bn2.bias
# audio_convnet.layer3.0.downsample.0.weight
# audio_convnet.layer3.0.downsample.1.weight
# audio_convnet.layer3.0.downsample.1.bias
# audio_convnet.layer3.1.conv1.weight
# audio_convnet.layer3.1.bn1.weight
# audio_convnet.layer3.1.bn1.bias
# audio_convnet.layer3.1.conv2.weight
# audio_convnet.layer3.1.bn2.weight
# audio_convnet.layer3.1.bn2.bias
# audio_convnet.layer4.0.conv1.weight
# audio_convnet.layer4.0.bn1.weight
# audio_convnet.layer4.0.bn1.bias
# audio_convnet.layer4.0.conv2.weight
# audio_convnet.layer4.0.bn2.weight
# audio_convnet.layer4.0.bn2.bias
# audio_convnet.layer4.0.downsample.0.weight
# audio_convnet.layer4.0.downsample.1.weight
# audio_convnet.layer4.0.downsample.1.bias
# audio_convnet.layer4.1.conv1.weight
# audio_convnet.layer4.1.bn1.weight
# audio_convnet.layer4.1.bn1.bias
# audio_convnet.layer4.1.conv2.weight
# audio_convnet.layer4.1.bn2.weight
# audio_convnet.layer4.1.bn2.bias
# r_layers.0.attention.self.query.weight
# r_layers.0.attention.self.query.bias
# r_layers.0.attention.self.key.weight
# r_layers.0.attention.self.key.bias
# r_layers.0.attention.self.value.weight
# r_layers.0.attention.self.value.bias
# r_layers.0.attention.output.dense.weight
# r_layers.0.attention.output.dense.bias
# r_layers.0.attention.output.LayerNorm.weight
# r_layers.0.attention.output.LayerNorm.bias
# r_layers.0.intermediate.dense.weight
# r_layers.0.intermediate.dense.bias
# r_layers.0.output.dense.weight
# r_layers.0.output.dense.bias
# r_layers.0.output.LayerNorm.weight
# r_layers.0.output.LayerNorm.bias
# r_layers.1.attention.self.query.weight
# r_layers.1.attention.self.query.bias
# r_layers.1.attention.self.key.weight
# r_layers.1.attention.self.key.bias
# r_layers.1.attention.self.value.weight
# r_layers.1.attention.self.value.bias
# r_layers.1.attention.output.dense.weight
# r_layers.1.attention.output.dense.bias
# r_layers.1.attention.output.LayerNorm.weight
# r_layers.1.attention.output.LayerNorm.bias
# r_layers.1.intermediate.dense.weight
# r_layers.1.intermediate.dense.bias
# r_layers.1.output.dense.weight
# r_layers.1.output.dense.bias
# r_layers.1.output.LayerNorm.weight
# r_layers.1.output.LayerNorm.bias
# r_layers.2.attention.self.query.weight
# r_layers.2.attention.self.query.bias
# r_layers.2.attention.self.key.weight
# r_layers.2.attention.self.key.bias
# r_layers.2.attention.self.value.weight
# r_layers.2.attention.self.value.bias
# r_layers.2.attention.output.dense.weight
# r_layers.2.attention.output.dense.bias
# r_layers.2.attention.output.LayerNorm.weight
# r_layers.2.attention.output.LayerNorm.bias
# r_layers.2.intermediate.dense.weight
# r_layers.2.intermediate.dense.bias
# r_layers.2.output.dense.weight
# r_layers.2.output.dense.bias
# r_layers.2.output.LayerNorm.weight
# r_layers.2.output.LayerNorm.bias
# r_layers.3.attention.self.query.weight
# r_layers.3.attention.self.query.bias
# r_layers.3.attention.self.key.weight
# r_layers.3.attention.self.key.bias
# r_layers.3.attention.self.value.weight
# r_layers.3.attention.self.value.bias
# r_layers.3.attention.output.dense.weight
# r_layers.3.attention.output.dense.bias
# r_layers.3.attention.output.LayerNorm.weight
# r_layers.3.attention.output.LayerNorm.bias
# r_layers.3.intermediate.dense.weight
# r_layers.3.intermediate.dense.bias
# r_layers.3.output.dense.weight
# r_layers.3.output.dense.bias
# r_layers.3.output.LayerNorm.weight
# r_layers.3.output.LayerNorm.bias
# r_layers.4.attention.self.query.weight
# r_layers.4.attention.self.query.bias
# r_layers.4.attention.self.key.weight
# r_layers.4.attention.self.key.bias
# r_layers.4.attention.self.value.weight
# r_layers.4.attention.self.value.bias
# r_layers.4.attention.output.dense.weight
# r_layers.4.attention.output.dense.bias
# r_layers.4.attention.output.LayerNorm.weight
# r_layers.4.attention.output.LayerNorm.bias
# r_layers.4.intermediate.dense.weight
# r_layers.4.intermediate.dense.bias
# r_layers.4.output.dense.weight
# r_layers.4.output.dense.bias
# r_layers.4.output.LayerNorm.weight
# r_layers.4.output.LayerNorm.bias
# r_layers.5.attention.self.query.weight
# r_layers.5.attention.self.query.bias
# r_layers.5.attention.self.key.weight
# r_layers.5.attention.self.key.bias
# r_layers.5.attention.self.value.weight
# r_layers.5.attention.self.value.bias
# r_layers.5.attention.output.dense.weight
# r_layers.5.attention.output.dense.bias
# r_layers.5.attention.output.LayerNorm.weight
# r_layers.5.attention.output.LayerNorm.bias
# r_layers.5.intermediate.dense.weight
# r_layers.5.intermediate.dense.bias
# r_layers.5.output.dense.weight
# r_layers.5.output.dense.bias
# r_layers.5.output.LayerNorm.weight
# r_layers.5.output.LayerNorm.bias
# audio_cls_token_proj_coarse.0.weight
# audio_cls_token_proj_coarse.0.bias
# audio_cls_token_proj_coarse.2.weight
# audio_cls_token_proj_coarse.2.bias
# visual_cls_token_proj_coarse.0.weight
# visual_cls_token_proj_coarse.0.bias
# visual_cls_token_proj_coarse.2.weight
# visual_cls_token_proj_coarse.2.bias
# audio_cls_token_proj_pre.0.weight
# audio_cls_token_proj_pre.0.bias
# audio_cls_token_proj_pre.2.weight
# audio_cls_token_proj_pre.2.bias
# attn_after_res.attention.self.query.weight
# attn_after_res.attention.self.query.bias
# attn_after_res.attention.self.key.weight
# attn_after_res.attention.self.key.bias
# attn_after_res.attention.self.value.weight
# attn_after_res.attention.self.value.bias
# attn_after_res.attention.output.dense.weight
# attn_after_res.attention.output.dense.bias
# attn_after_res.attention.output.LayerNorm.weight
# attn_after_res.attention.output.LayerNorm.bias
# attn_after_res.intermediate.dense.weight
# attn_after_res.intermediate.dense.bias
# attn_after_res.output.dense.weight
# attn_after_res.output.dense.bias
# attn_after_res.output.LayerNorm.weight
# attn_after_res.output.LayerNorm.bias
# attn_after_res_proj.weight
# attn_after_res_proj.bias



# # new names
# visual_cls_token
# trm13.mask_emb
# trm13.cls_token
# trm13.feature_extractor.conv_layers.0.0.weight
# trm13.feature_extractor.conv_layers.0.2.weight
# trm13.feature_extractor.conv_layers.0.2.bias
# trm13.feature_extractor.conv_layers.1.0.weight
# trm13.feature_extractor.conv_layers.2.0.weight
# trm13.feature_extractor.conv_layers.3.0.weight
# trm13.feature_extractor.conv_layers.4.0.weight
# trm13.feature_extractor.conv_layers.5.0.weight
# trm13.feature_extractor.conv_layers.6.0.weight
# trm13.post_extract_proj.weight
# trm13.post_extract_proj.bias
# trm13.quantizer.vars
# trm13.quantizer.weight_proj.weight
# trm13.quantizer.weight_proj.bias
# trm13.project_q.weight
# trm13.project_q.bias
# trm13.encoder.pos_conv.0.bias
# trm13.encoder.pos_conv.0.weight_g
# trm13.encoder.pos_conv.0.weight_v
# trm13.encoder.layers.0.self_attn.k_proj.weight
# trm13.encoder.layers.0.self_attn.k_proj.bias
# trm13.encoder.layers.0.self_attn.v_proj.weight
# trm13.encoder.layers.0.self_attn.v_proj.bias
# trm13.encoder.layers.0.self_attn.q_proj.weight
# trm13.encoder.layers.0.self_attn.q_proj.bias
# trm13.encoder.layers.0.self_attn.out_proj.weight
# trm13.encoder.layers.0.self_attn.out_proj.bias
# trm13.encoder.layers.0.self_attn_layer_norm.weight
# trm13.encoder.layers.0.self_attn_layer_norm.bias
# trm13.encoder.layers.0.fc1.weight
# trm13.encoder.layers.0.fc1.bias
# trm13.encoder.layers.0.fc2.weight
# trm13.encoder.layers.0.fc2.bias
# trm13.encoder.layers.0.final_layer_norm.weight
# trm13.encoder.layers.0.final_layer_norm.bias
# trm13.encoder.layers.1.self_attn.k_proj.weight
# trm13.encoder.layers.1.self_attn.k_proj.bias
# trm13.encoder.layers.1.self_attn.v_proj.weight
# trm13.encoder.layers.1.self_attn.v_proj.bias
# trm13.encoder.layers.1.self_attn.q_proj.weight
# trm13.encoder.layers.1.self_attn.q_proj.bias
# trm13.encoder.layers.1.self_attn.out_proj.weight
# trm13.encoder.layers.1.self_attn.out_proj.bias
# trm13.encoder.layers.1.self_attn_layer_norm.weight
# trm13.encoder.layers.1.self_attn_layer_norm.bias
# trm13.encoder.layers.1.fc1.weight
# trm13.encoder.layers.1.fc1.bias
# trm13.encoder.layers.1.fc2.weight
# trm13.encoder.layers.1.fc2.bias
# trm13.encoder.layers.1.final_layer_norm.weight
# trm13.encoder.layers.1.final_layer_norm.bias
# trm13.encoder.layers.2.self_attn.k_proj.weight
# trm13.encoder.layers.2.self_attn.k_proj.bias
# trm13.encoder.layers.2.self_attn.v_proj.weight
# trm13.encoder.layers.2.self_attn.v_proj.bias
# trm13.encoder.layers.2.self_attn.q_proj.weight
# trm13.encoder.layers.2.self_attn.q_proj.bias
# trm13.encoder.layers.2.self_attn.out_proj.weight
# trm13.encoder.layers.2.self_attn.out_proj.bias
# trm13.encoder.layers.2.self_attn_layer_norm.weight
# trm13.encoder.layers.2.self_attn_layer_norm.bias
# trm13.encoder.layers.2.fc1.weight
# trm13.encoder.layers.2.fc1.bias
# trm13.encoder.layers.2.fc2.weight
# trm13.encoder.layers.2.fc2.bias
# trm13.encoder.layers.2.final_layer_norm.weight
# trm13.encoder.layers.2.final_layer_norm.bias
# trm13.encoder.layers.3.self_attn.k_proj.weight
# trm13.encoder.layers.3.self_attn.k_proj.bias
# trm13.encoder.layers.3.self_attn.v_proj.weight
# trm13.encoder.layers.3.self_attn.v_proj.bias
# trm13.encoder.layers.3.self_attn.q_proj.weight
# trm13.encoder.layers.3.self_attn.q_proj.bias
# trm13.encoder.layers.3.self_attn.out_proj.weight
# trm13.encoder.layers.3.self_attn.out_proj.bias
# trm13.encoder.layers.3.self_attn_layer_norm.weight
# trm13.encoder.layers.3.self_attn_layer_norm.bias
# trm13.encoder.layers.3.fc1.weight
# trm13.encoder.layers.3.fc1.bias
# trm13.encoder.layers.3.fc2.weight
# trm13.encoder.layers.3.fc2.bias
# trm13.encoder.layers.3.final_layer_norm.weight
# trm13.encoder.layers.3.final_layer_norm.bias
# trm13.encoder.layers.4.self_attn.k_proj.weight
# trm13.encoder.layers.4.self_attn.k_proj.bias
# trm13.encoder.layers.4.self_attn.v_proj.weight
# trm13.encoder.layers.4.self_attn.v_proj.bias
# trm13.encoder.layers.4.self_attn.q_proj.weight
# trm13.encoder.layers.4.self_attn.q_proj.bias
# trm13.encoder.layers.4.self_attn.out_proj.weight
# trm13.encoder.layers.4.self_attn.out_proj.bias
# trm13.encoder.layers.4.self_attn_layer_norm.weight
# trm13.encoder.layers.4.self_attn_layer_norm.bias
# trm13.encoder.layers.4.fc1.weight
# trm13.encoder.layers.4.fc1.bias
# trm13.encoder.layers.4.fc2.weight
# trm13.encoder.layers.4.fc2.bias
# trm13.encoder.layers.4.final_layer_norm.weight
# trm13.encoder.layers.4.final_layer_norm.bias
# trm13.encoder.layers.5.self_attn.k_proj.weight
# trm13.encoder.layers.5.self_attn.k_proj.bias
# trm13.encoder.layers.5.self_attn.v_proj.weight
# trm13.encoder.layers.5.self_attn.v_proj.bias
# trm13.encoder.layers.5.self_attn.q_proj.weight
# trm13.encoder.layers.5.self_attn.q_proj.bias
# trm13.encoder.layers.5.self_attn.out_proj.weight
# trm13.encoder.layers.5.self_attn.out_proj.bias
# trm13.encoder.layers.5.self_attn_layer_norm.weight
# trm13.encoder.layers.5.self_attn_layer_norm.bias
# trm13.encoder.layers.5.fc1.weight
# trm13.encoder.layers.5.fc1.bias
# trm13.encoder.layers.5.fc2.weight
# trm13.encoder.layers.5.fc2.bias
# trm13.encoder.layers.5.final_layer_norm.weight
# trm13.encoder.layers.5.final_layer_norm.bias
# trm13.encoder.layers.6.self_attn.k_proj.weight
# trm13.encoder.layers.6.self_attn.k_proj.bias
# trm13.encoder.layers.6.self_attn.v_proj.weight
# trm13.encoder.layers.6.self_attn.v_proj.bias
# trm13.encoder.layers.6.self_attn.q_proj.weight
# trm13.encoder.layers.6.self_attn.q_proj.bias
# trm13.encoder.layers.6.self_attn.out_proj.weight
# trm13.encoder.layers.6.self_attn.out_proj.bias
# trm13.encoder.layers.6.self_attn_layer_norm.weight
# trm13.encoder.layers.6.self_attn_layer_norm.bias
# trm13.encoder.layers.6.fc1.weight
# trm13.encoder.layers.6.fc1.bias
# trm13.encoder.layers.6.fc2.weight
# trm13.encoder.layers.6.fc2.bias
# trm13.encoder.layers.6.final_layer_norm.weight
# trm13.encoder.layers.6.final_layer_norm.bias
# trm13.encoder.layers.7.self_attn.k_proj.weight
# trm13.encoder.layers.7.self_attn.k_proj.bias
# trm13.encoder.layers.7.self_attn.v_proj.weight
# trm13.encoder.layers.7.self_attn.v_proj.bias
# trm13.encoder.layers.7.self_attn.q_proj.weight
# trm13.encoder.layers.7.self_attn.q_proj.bias
# trm13.encoder.layers.7.self_attn.out_proj.weight
# trm13.encoder.layers.7.self_attn.out_proj.bias
# trm13.encoder.layers.7.self_attn_layer_norm.weight
# trm13.encoder.layers.7.self_attn_layer_norm.bias
# trm13.encoder.layers.7.fc1.weight
# trm13.encoder.layers.7.fc1.bias
# trm13.encoder.layers.7.fc2.weight
# trm13.encoder.layers.7.fc2.bias
# trm13.encoder.layers.7.final_layer_norm.weight
# trm13.encoder.layers.7.final_layer_norm.bias
# trm13.encoder.layers.8.self_attn.k_proj.weight
# trm13.encoder.layers.8.self_attn.k_proj.bias
# trm13.encoder.layers.8.self_attn.v_proj.weight
# trm13.encoder.layers.8.self_attn.v_proj.bias
# trm13.encoder.layers.8.self_attn.q_proj.weight
# trm13.encoder.layers.8.self_attn.q_proj.bias
# trm13.encoder.layers.8.self_attn.out_proj.weight
# trm13.encoder.layers.8.self_attn.out_proj.bias
# trm13.encoder.layers.8.self_attn_layer_norm.weight
# trm13.encoder.layers.8.self_attn_layer_norm.bias
# trm13.encoder.layers.8.fc1.weight
# trm13.encoder.layers.8.fc1.bias
# trm13.encoder.layers.8.fc2.weight
# trm13.encoder.layers.8.fc2.bias
# trm13.encoder.layers.8.final_layer_norm.weight
# trm13.encoder.layers.8.final_layer_norm.bias
# trm13.encoder.layers.9.self_attn.k_proj.weight
# trm13.encoder.layers.9.self_attn.k_proj.bias
# trm13.encoder.layers.9.self_attn.v_proj.weight
# trm13.encoder.layers.9.self_attn.v_proj.bias
# trm13.encoder.layers.9.self_attn.q_proj.weight
# trm13.encoder.layers.9.self_attn.q_proj.bias
# trm13.encoder.layers.9.self_attn.out_proj.weight
# trm13.encoder.layers.9.self_attn.out_proj.bias
# trm13.encoder.layers.9.self_attn_layer_norm.weight
# trm13.encoder.layers.9.self_attn_layer_norm.bias
# trm13.encoder.layers.9.fc1.weight
# trm13.encoder.layers.9.fc1.bias
# trm13.encoder.layers.9.fc2.weight
# trm13.encoder.layers.9.fc2.bias
# trm13.encoder.layers.9.final_layer_norm.weight
# trm13.encoder.layers.9.final_layer_norm.bias
# trm13.encoder.layers.10.self_attn.k_proj.weight
# trm13.encoder.layers.10.self_attn.k_proj.bias
# trm13.encoder.layers.10.self_attn.v_proj.weight
# trm13.encoder.layers.10.self_attn.v_proj.bias
# trm13.encoder.layers.10.self_attn.q_proj.weight
# trm13.encoder.layers.10.self_attn.q_proj.bias
# trm13.encoder.layers.10.self_attn.out_proj.weight
# trm13.encoder.layers.10.self_attn.out_proj.bias
# trm13.encoder.layers.10.self_attn_layer_norm.weight
# trm13.encoder.layers.10.self_attn_layer_norm.bias
# trm13.encoder.layers.10.fc1.weight
# trm13.encoder.layers.10.fc1.bias
# trm13.encoder.layers.10.fc2.weight
# trm13.encoder.layers.10.fc2.bias
# trm13.encoder.layers.10.final_layer_norm.weight
# trm13.encoder.layers.10.final_layer_norm.bias
# trm13.encoder.layers.11.self_attn.k_proj.weight
# trm13.encoder.layers.11.self_attn.k_proj.bias
# trm13.encoder.layers.11.self_attn.v_proj.weight
# trm13.encoder.layers.11.self_attn.v_proj.bias
# trm13.encoder.layers.11.self_attn.q_proj.weight
# trm13.encoder.layers.11.self_attn.q_proj.bias
# trm13.encoder.layers.11.self_attn.out_proj.weight
# trm13.encoder.layers.11.self_attn.out_proj.bias
# trm13.encoder.layers.11.self_attn_layer_norm.weight
# trm13.encoder.layers.11.self_attn_layer_norm.bias
# trm13.encoder.layers.11.fc1.weight
# trm13.encoder.layers.11.fc1.bias
# trm13.encoder.layers.11.fc2.weight
# trm13.encoder.layers.11.fc2.bias
# trm13.encoder.layers.11.final_layer_norm.weight
# trm13.encoder.layers.11.final_layer_norm.bias
# trm13.encoder.layer_norm.weight
# trm13.encoder.layer_norm.bias
# trm13.layer_norm.weight
# trm13.layer_norm.bias
# trm13.final_proj.weight
# trm13.final_proj.bias
# cnn2.linear.weight
# cnn2.linear.bias
# cnn2.bn1.weight
# cnn2.bn1.bias
# cnn2.layer1.0.conv1.weight
# cnn2.layer1.0.bn1.weight
# cnn2.layer1.0.bn1.bias
# cnn2.layer1.0.conv2.weight
# cnn2.layer1.0.bn2.weight
# cnn2.layer1.0.bn2.bias
# cnn2.layer1.0.downsample.0.weight
# cnn2.layer1.0.downsample.1.weight
# cnn2.layer1.0.downsample.1.bias
# cnn2.layer1.1.conv1.weight
# cnn2.layer1.1.bn1.weight
# cnn2.layer1.1.bn1.bias
# cnn2.layer1.1.conv2.weight
# cnn2.layer1.1.bn2.weight
# cnn2.layer1.1.bn2.bias
# cnn2.layer2.0.conv1.weight
# cnn2.layer2.0.bn1.weight
# cnn2.layer2.0.bn1.bias
# cnn2.layer2.0.conv2.weight
# cnn2.layer2.0.bn2.weight
# cnn2.layer2.0.bn2.bias
# cnn2.layer2.0.downsample.0.weight
# cnn2.layer2.0.downsample.1.weight
# cnn2.layer2.0.downsample.1.bias
# cnn2.layer2.1.conv1.weight
# cnn2.layer2.1.bn1.weight
# cnn2.layer2.1.bn1.bias
# cnn2.layer2.1.conv2.weight
# cnn2.layer2.1.bn2.weight
# cnn2.layer2.1.bn2.bias
# cnn2.layer3.0.conv1.weight
# cnn2.layer3.0.bn1.weight
# cnn2.layer3.0.bn1.bias
# cnn2.layer3.0.conv2.weight
# cnn2.layer3.0.bn2.weight
# cnn2.layer3.0.bn2.bias
# cnn2.layer3.0.downsample.0.weight
# cnn2.layer3.0.downsample.1.weight
# cnn2.layer3.0.downsample.1.bias
# cnn2.layer3.1.conv1.weight
# cnn2.layer3.1.bn1.weight
# cnn2.layer3.1.bn1.bias
# cnn2.layer3.1.conv2.weight
# cnn2.layer3.1.bn2.weight
# cnn2.layer3.1.bn2.bias
# cnn2.layer4.0.conv1.weight
# cnn2.layer4.0.bn1.weight
# cnn2.layer4.0.bn1.bias
# cnn2.layer4.0.conv2.weight
# cnn2.layer4.0.bn2.weight
# cnn2.layer4.0.bn2.bias
# cnn2.layer4.0.downsample.0.weight
# cnn2.layer4.0.downsample.1.weight
# cnn2.layer4.0.downsample.1.bias
# cnn2.layer4.1.conv1.weight
# cnn2.layer4.1.bn1.weight
# cnn2.layer4.1.bn1.bias
# cnn2.layer4.1.conv2.weight
# cnn2.layer4.1.bn2.weight
# cnn2.layer4.1.bn2.bias
# trm.0.attention.self.query.weight
# trm.0.attention.self.query.bias
# trm.0.attention.self.key.weight
# trm.0.attention.self.key.bias
# trm.0.attention.self.value.weight
# trm.0.attention.self.value.bias
# trm.0.attention.output.dense.weight
# trm.0.attention.output.dense.bias
# trm.0.attention.output.LayerNorm.weight
# trm.0.attention.output.LayerNorm.bias
# trm.0.intermediate.dense.weight
# trm.0.intermediate.dense.bias
# trm.0.output.dense.weight
# trm.0.output.dense.bias
# trm.0.output.LayerNorm.weight
# trm.0.output.LayerNorm.bias
# trm.1.attention.self.query.weight
# trm.1.attention.self.query.bias
# trm.1.attention.self.key.weight
# trm.1.attention.self.key.bias
# trm.1.attention.self.value.weight
# trm.1.attention.self.value.bias
# trm.1.attention.output.dense.weight
# trm.1.attention.output.dense.bias
# trm.1.attention.output.LayerNorm.weight
# trm.1.attention.output.LayerNorm.bias
# trm.1.intermediate.dense.weight
# trm.1.intermediate.dense.bias
# trm.1.output.dense.weight
# trm.1.output.dense.bias
# trm.1.output.LayerNorm.weight
# trm.1.output.LayerNorm.bias
# trm.2.attention.self.query.weight
# trm.2.attention.self.query.bias
# trm.2.attention.self.key.weight
# trm.2.attention.self.key.bias
# trm.2.attention.self.value.weight
# trm.2.attention.self.value.bias
# trm.2.attention.output.dense.weight
# trm.2.attention.output.dense.bias
# trm.2.attention.output.LayerNorm.weight
# trm.2.attention.output.LayerNorm.bias
# trm.2.intermediate.dense.weight
# trm.2.intermediate.dense.bias
# trm.2.output.dense.weight
# trm.2.output.dense.bias
# trm.2.output.LayerNorm.weight
# trm.2.output.LayerNorm.bias
# trm.3.attention.self.query.weight
# trm.3.attention.self.query.bias
# trm.3.attention.self.key.weight
# trm.3.attention.self.key.bias
# trm.3.attention.self.value.weight
# trm.3.attention.self.value.bias
# trm.3.attention.output.dense.weight
# trm.3.attention.output.dense.bias
# trm.3.attention.output.LayerNorm.weight
# trm.3.attention.output.LayerNorm.bias
# trm.3.intermediate.dense.weight
# trm.3.intermediate.dense.bias
# trm.3.output.dense.weight
# trm.3.output.dense.bias
# trm.3.output.LayerNorm.weight
# trm.3.output.LayerNorm.bias
# trm.4.attention.self.query.weight
# trm.4.attention.self.query.bias
# trm.4.attention.self.key.weight
# trm.4.attention.self.key.bias
# trm.4.attention.self.value.weight
# trm.4.attention.self.value.bias
# trm.4.attention.output.dense.weight
# trm.4.attention.output.dense.bias
# trm.4.attention.output.LayerNorm.weight
# trm.4.attention.output.LayerNorm.bias
# trm.4.intermediate.dense.weight
# trm.4.intermediate.dense.bias
# trm.4.output.dense.weight
# trm.4.output.dense.bias
# trm.4.output.LayerNorm.weight
# trm.4.output.LayerNorm.bias
# trm.5.attention.self.query.weight
# trm.5.attention.self.query.bias
# trm.5.attention.self.key.weight
# trm.5.attention.self.key.bias
# trm.5.attention.self.value.weight
# trm.5.attention.self.value.bias
# trm.5.attention.output.dense.weight
# trm.5.attention.output.dense.bias
# trm.5.attention.output.LayerNorm.weight
# trm.5.attention.output.LayerNorm.bias
# trm.5.intermediate.dense.weight
# trm.5.intermediate.dense.bias
# trm.5.output.dense.weight
# trm.5.output.dense.bias
# trm.5.output.LayerNorm.weight
# trm.5.output.LayerNorm.bias
# audio_cls_token_proj_coarse.0.weight
# audio_cls_token_proj_coarse.0.bias
# audio_cls_token_proj_coarse.2.weight
# audio_cls_token_proj_coarse.2.bias
# audio_cls_token_proj_pre.0.weight
# audio_cls_token_proj_pre.0.bias
# audio_cls_token_proj_pre.2.weight
# audio_cls_token_proj_pre.2.bias
# trm2.attention.self.query.weight
# trm2.attention.self.query.bias
# trm2.attention.self.key.weight
# trm2.attention.self.key.bias
# trm2.attention.self.value.weight
# trm2.attention.self.value.bias
# trm2.attention.output.dense.weight
# trm2.attention.output.dense.bias
# trm2.attention.output.LayerNorm.weight
# trm2.attention.output.LayerNorm.bias
# trm2.intermediate.dense.weight
# trm2.intermediate.dense.bias
# trm2.output.dense.weight
# trm2.output.dense.bias
# trm2.output.LayerNorm.weight
# trm2.output.LayerNorm.bias
# trm2_proj.weight
# trm2_proj.bias
# visn_fc.visn_fc.weight
# visn_fc.visn_fc.bias
# visn_fc.visn_layer_norm.weight
# visn_fc.visn_layer_norm.bias
# visn_fc.box_fc.weight
# visn_fc.box_fc.bias
# visn_fc.box_layer_norm.weight
# visn_fc.box_layer_norm.bias
# visual_cls_token_proj_coarse.0.weight
# visual_cls_token_proj_coarse.0.bias
# visual_cls_token_proj_coarse.2.weight
# visual_cls_token_proj_coarse.2.bias