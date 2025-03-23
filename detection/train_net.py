"""
Probabilistic Detectron Training Script following Detectron2 training script found at detectron2/tools.
"""
import core
import os
import sys

# This is very ugly. Essential for now but should be fixed.
sys.path.append(os.path.join(core.top_dir(), 'src', 'detr'))

# Detectron imports
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.engine import launch
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators, verify_results


# Project imports
from core.setup import setup_config, setup_arg_parser
from default_trainer import DefaultTrainer



class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        """
        Builds evaluators for post-training mAP report.
        Args:
            cfg(CfgNode): a detectron2 CfgNode
            dataset_name(str): registered dataset name

        Returns:
            detectron2 DatasetEvaluators object
        """
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluators = [COCOEvaluator(dataset_name, cfg, True, output_folder)]
        return DatasetEvaluators(evaluators)



def main(args):
    # Setup config node
    cfg = setup_config(args,
                       random_seed=args.random_seed, is_testing=False, ood=False)

    trainer = Trainer(cfg)

    if args.eval_only:
        model = trainer.build_model(cfg)
        model.eval()
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    # Explicitly handle resume with optimizer reset
    if args.resume and trainer.checkpointer.has_checkpoint():
        checkpoint_file = trainer.checkpointer.get_checkpoint_file()
        print(f"Loading checkpoint from {checkpoint_file}")
        import torch
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        print(f"Checkpoint keys: {checkpoint.keys()}")
        trainer.model.load_state_dict(checkpoint['model'])
        trainer.start_iter = checkpoint.get('iteration', 0)  # Set start_iter
        print(f"Resuming from iteration: {trainer.start_iter}")
        trainer.optimizer = Trainer.build_optimizer(cfg, trainer.model)
        trainer.scheduler = trainer.build_lr_scheduler(cfg, trainer.optimizer)
        print(f"Optimizer reset with BASE_LR: {cfg.SOLVER.BASE_LR}")
    else:
        print("Starting fresh training (no checkpoint)")
        trainer.resume_or_load(resume=False)
        trainer.start_iter = 0

    return trainer.train()  # Use default start_iter and max_iter from cfg

if __name__ == "__main__":
    # Create arg parser
    arg_parser = setup_arg_parser()

    args = arg_parser.parse_args()
    print("Command Line Args:", args)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
