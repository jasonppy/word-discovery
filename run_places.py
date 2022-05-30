import argparse
import os
import pickle
import time
from steps import trainer
from models import audio_encoder, dual_encoder
from datasets import places_dataset
from logging import getLogger

logger = getLogger(__name__)
logger.info("I am process %s, running on %s: starting (%s)" % (
        os.getpid(), os.uname()[1], time.asctime()))

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--resume", action="store_true", dest="resume", help="load from exp_dir if True")
parser.add_argument("--validate", action="store_true", default=False, help="temp, if call trainer_variants rather than trainer")


trainer.Trainer.add_args(parser)
audio_encoder.AudioEncoder.add_args(parser)
dual_encoder.DualEncoder.add_args(parser)
places_dataset.ImageCaptionDataset.add_args(parser)
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
        if key not in old_args:
            old_args[key] = new_args[key]
    args = argparse.Namespace(**old_args)
    args.resume = resume
else:
    print("\nexp_dir: %s" % args.exp_dir)
    with open("%s/args.pkl" % args.exp_dir, "wb") as f:
        pickle.dump(args, f)
args.places = True
logger.info(args)


if args.validate:
    my_trainer = trainer.Trainer(args)
    my_trainer.validate(my_trainer.valid_loader)
    my_trainer.validate(my_trainer.valid_loader2, unseen=True)
else:
    my_trainer = trainer.Trainer(args)
    my_trainer.train()
