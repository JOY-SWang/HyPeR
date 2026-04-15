import logging
from dataclasses import dataclass, field
from typing import Optional

import transformers
from transformers import HfArgumentParser
from trl import GRPOConfig

from trainer.grpo_trainer_deepconf_hs import GRPOTrainer, DeepConfConfig, PauseConfig
from utils.rewards import accuracy_reward, format_reward
from dataset.dataset import AudioDataset


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    config_path: Optional[str] = field(default=None, metadata={"help": "config path"})
    model_name_or_path : Optional[str] = field(default=None, metadata={"help": "model name or path"})
    out_dir: Optional[str] = field(default=None, metadata={"help": "output dir for model"})
    data_file: Optional[str] = field(default=None, metadata={"help": "train data file"})
    use_wandb: Optional[str] = field(default="false", metadata={"help": "whether use wandb to report logs"})
    
    # DeepConf specific arguments
    deepconf_enabled: Optional[bool] = field(default=True, metadata={"help": "enable DeepConf-GRPO"})
    window_size: Optional[int] = field(default=2048, metadata={"help": "sliding window size for LGC computation"})
    stride: Optional[int] = field(default=256, metadata={"help": "sliding window stride"})
    top_p_keep: Optional[float] = field(default=0.25, metadata={"help": "fraction of trajectories to keep per prompt"})
    weight_clip_min: Optional[float] = field(default=0.5, metadata={"help": "minimum weight clipping"})
    weight_clip_max: Optional[float] = field(default=1.5, metadata={"help": "maximum weight clipping"})
    standardize_over: Optional[str] = field(default="batch", metadata={"help": "standardization scope: 'batch' or 'group'"})
    
    # Pause (latent thinking) specific arguments
    pause_enabled: Optional[bool] = field(default=True, metadata={"help": "enable pause/latent thinking"})
    tau_pause_quantile: Optional[float] = field(default=0.50, metadata={"help": "entropy quantile to trigger pause"})
    tau_abort_quantile: Optional[float] = field(default=0.08, metadata={"help": "entropy quantile to abort"})
    max_pauses: Optional[int] = field(default=5, metadata={"help": "maximum number of pauses per sequence"})
    max_think_tokens: Optional[int] = field(default=128, metadata={"help": "maximum thinking tokens per pause"})
    recovery_bonus: Optional[float] = field(default=0.05, metadata={"help": "bonus for recovery after pause"})
    leak_penalty: Optional[float] = field(default=1.0, metadata={"help": "penalty for think token leakage"})

    def __post_init__(self):
        if self.config_path is None:
            raise ValueError("config path should not none")


def main():
    print("=== Starting DeepConf-GRPO Training ===")
    # print("Parsing arguments...")
    
    parser = HfArgumentParser(DataTrainingArguments)
    data_args = parser.parse_args_into_dataclasses()[0]
    
    # print("Arguments parsed successfully")
    # print(f"Model: {data_args.model_name_or_path}")
    # print(f"Data file: {data_args.data_file}")
    # print(f"Output dir: {data_args.out_dir}")
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    transformers.logging.set_verbosity_info()
    logging.info(data_args)

    # Initialize reward functions
    # print("Initializing reward functions...")
    reward_funcs_registry = {"accuracy": accuracy_reward, "format": format_reward}
    reward_funcs = [reward_funcs_registry["accuracy"], reward_funcs_registry["format"]]
    # print("Reward functions initialized")

    # Load training dataset
    print("Loading training dataset...")
    train_dataset = AudioDataset(data_args.data_file)
    print(f"Dataset loaded with {len(train_dataset)} samples")

    # Configure DeepConf-GRPO
    print("Configuring DeepConf-GRPO...")
    deepconf_config = DeepConfConfig(
        enabled=data_args.deepconf_enabled,
        window_size=data_args.window_size,
        stride=data_args.stride,
        top_p_keep=data_args.top_p_keep,
        weight_clip_min=data_args.weight_clip_min,
        weight_clip_max=data_args.weight_clip_max,
        standardize_over=data_args.standardize_over,
        truncate_below_tau=False,  # Keep default for now
        tau_quantile=0.90
    )
    print("DeepConf configuration created")
    
    # Configure Pause (latent thinking)
    print("Configuring Pause (latent thinking)...")
    pause_config = PauseConfig(
        enabled=data_args.pause_enabled,
        tau_pause_quantile=data_args.tau_pause_quantile,
        tau_abort_quantile=data_args.tau_abort_quantile,
        max_pauses=data_args.max_pauses,
        max_think_tokens=data_args.max_think_tokens,
        recovery_bonus=data_args.recovery_bonus,
        leak_penalty=data_args.leak_penalty
    )
    print("Pause configuration created")

    # Training configuration
    print("Creating training configuration...")
    training_args = GRPOConfig(
        seed=42,
        data_seed=42,
        output_dir=data_args.out_dir, 
        deepspeed=data_args.config_path, 
        max_prompt_length=512, 
        max_completion_length=1024,
        per_device_train_batch_size=1, 
        gradient_accumulation_steps=4, 
        logging_steps=1, 
        bf16=True,
        report_to="wandb" if data_args.use_wandb == "true" else [],
        gradient_checkpointing=False, 
        num_train_epochs=2, 
        max_steps=1000,
        run_name="AQA-DeepConf-GRPO", 
        save_steps=100, 
        save_only_model=True, 
        temperature=1.0,
        num_generations=4,
        beta=0.1  # KL penalty weight
    )
    print("Training configuration created")
    
    # Initialize DeepConf-GRPO trainer
    print("Initializing DeepConf-GRPO trainer...")
    trainer = GRPOTrainer(
        model=data_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        deepconf_config=deepconf_config,
        pause_config=pause_config
    )
    trainer.latent_probe_cfg.enabled = True
    trainer.latent_probe_cfg.sample_index = 0      # 看 batch 里第 0 条
    trainer.latent_probe_cfg.max_print_steps = 64  # 打印前 64 步
    print("Trainer initialized successfully")

    # Log configuration
    logging.info(f"DeepConf-GRPO Configuration:")
    logging.info(f"  DeepConf enabled: {deepconf_config.enabled}")
    logging.info(f"  Window size: {deepconf_config.window_size}")
    logging.info(f"  Stride: {deepconf_config.stride}")
    logging.info(f"  Top-p keep: {deepconf_config.top_p_keep}")
    logging.info(f"  Weight clipping: [{deepconf_config.weight_clip_min}, {deepconf_config.weight_clip_max}]")
    logging.info(f"  Standardize over: {deepconf_config.standardize_over}")
    logging.info(f"  Pause enabled: {pause_config.enabled}")
    logging.info(f"  Max pauses: {pause_config.max_pauses}")
    logging.info(f"  Max think tokens: {pause_config.max_think_tokens}")

    # Start training
    print("Starting training...")
    trainer.train()
    print("Training completed, saving model...")
    trainer.save_model(data_args.out_dir)
    
    logging.info(f"Training completed. Model saved to {data_args.out_dir}")
    print("=== DeepConf-GRPO Training Completed Successfully ===")


if __name__ == "__main__":
    main()