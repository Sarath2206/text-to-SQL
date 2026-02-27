#!/usr/bin/env python3
"""
Simplified Training Script for Google Colab (24GB GPU)

This script provides a minimal working example of RL training with enhanced rewards
that can run on Google Colab with a single 24GB GPU.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import wandb

# Add extensions to path
sys.path.insert(0, str(Path(__file__).parent))

from extensions.reward_enhanced import (
    EnhancedRewardComputer,
    EnhancedRewardResult
)
from extensions.config import load_config, ExtensionConfig


class SimplifiedTrainer:
    """
    Simplified trainer for demonstration on Colab.
    
    This is a minimal implementation that shows how to integrate
    the enhanced reward system. For production use, integrate with
    SQL-R1's full VERL training loop.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Coder-3B-Instruct",
        config_path: Optional[str] = None,
        output_dir: str = "./outputs",
        use_wandb: bool = False
    ):
        """Initialize trainer."""
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb
        
        # Load configuration
        if config_path and Path(config_path).exists():
            self.config = load_config(config_path)
        else:
            self.config = ExtensionConfig()
        
        print(f"[Trainer] Initializing with model: {model_name}")
        print(f"[Trainer] Output directory: {output_dir}")
        print(f"[Trainer] Enhanced rewards: {self.config.reward.enable_enhanced}")
        
        # Initialize model and tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Trainer] Using device: {self.device}")
        
        # Load tokenizer
        print("[Trainer] Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Load model with memory optimizations
        print("[Trainer] Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if self.config.optimization.mixed_precision == "bf16" else torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Enable gradient checkpointing if configured
        if self.config.optimization.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            print("[Trainer] Gradient checkpointing enabled")
        
        # Initialize enhanced reward computer
        self.reward_computer = EnhancedRewardComputer(
            baseline_reward_fn=self._baseline_reward,
            enable_enhanced=self.config.reward.enable_enhanced,
            schema_weight=self.config.reward.schema_weight,
            structural_select_weight=self.config.reward.structural_select_weight,
            structural_where_weight=self.config.reward.structural_where_weight,
            structural_join_weight=self.config.reward.structural_join_weight,
            syntax_weight=self.config.reward.syntax_weight
        )
        
        # Initialize wandb if requested
        if self.use_wandb:
            wandb.init(
                project="sql-r1-extension",
                config={
                    "model": model_name,
                    "enhanced_rewards": self.config.reward.enable_enhanced,
                    "batch_size": self.config.optimization.batch_size,
                    "gradient_accumulation": self.config.optimization.gradient_accumulation_steps
                }
            )
        
        print("[Trainer] Initialization complete")
    
    def _baseline_reward(
        self,
        solution_str: str,
        ground_truth: Dict[str, Any],
        **kwargs
    ) -> float:
        """
        Baseline reward function (simplified version of SQL-R1's reward).
        
        In production, this would call SQL-R1's actual reward function.
        For demonstration, we use a simplified version.
        """
        reward = 0.0
        
        # Format reward: Check for correct XML tags
        has_think = '<think>' in solution_str and '</think>' in solution_str
        has_answer = '<answer>' in solution_str and '</answer>' in solution_str
        has_sql_block = '```sql' in solution_str and '```' in solution_str
        
        if has_think and has_answer and has_sql_block:
            reward += 1.0
        
        # In production, add execution and result rewards here
        # For now, we just return format reward
        return reward
    
    def generate_response(self, prompt: str, max_length: int = 512) -> str:
        """Generate response from model."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from response
        response = response[len(prompt):].strip()
        return response
    
    def compute_reward(
        self,
        response: str,
        ground_truth: Dict[str, Any],
        schema: Optional[Dict[str, Any]] = None
    ) -> EnhancedRewardResult:
        """Compute reward for a response."""
        return self.reward_computer.compute_reward(
            solution_str=response,
            ground_truth=ground_truth,
            schema=schema
        )
    
    def train_step(
        self,
        batch: Dict[str, Any],
        step: int
    ) -> Dict[str, float]:
        """
        Single training step.
        
        In production, this would be replaced by SQL-R1's PPO training loop.
        For demonstration, we just show reward computation.
        """
        metrics = {
            'total_reward': 0.0,
            'baseline_reward': 0.0,
            'schema_reward': 0.0,
            'structural_reward': 0.0,
            'syntax_reward': 0.0
        }
        
        # Generate responses for batch
        for i, (prompt, ground_truth, schema) in enumerate(zip(
            batch['prompts'],
            batch['ground_truths'],
            batch.get('schemas', [None] * len(batch['prompts']))
        )):
            # Generate response
            response = self.generate_response(prompt)
            
            # Compute reward
            reward_result = self.compute_reward(response, ground_truth, schema)
            
            # Accumulate metrics
            metrics['total_reward'] += reward_result.total
            metrics['baseline_reward'] += reward_result.baseline_total
            metrics['schema_reward'] += reward_result.schema
            metrics['structural_reward'] += reward_result.structural
            metrics['syntax_reward'] += reward_result.syntax
            
            # Log first example
            if i == 0 and step % self.config.logging.log_interval == 0:
                print(f"\n[Step {step}] Example:")
                print(f"Prompt: {prompt[:100]}...")
                print(f"Response: {response[:200]}...")
                print(f"Reward: {reward_result.total:.3f}")
                print(f"  Baseline: {reward_result.baseline_total:.3f}")
                print(f"  Schema: {reward_result.schema:.3f}")
                print(f"  Structural: {reward_result.structural:.3f}")
                print(f"  Syntax: {reward_result.syntax:.3f}")
        
        # Average metrics
        batch_size = len(batch['prompts'])
        for key in metrics:
            metrics[key] /= batch_size
        
        return metrics
    
    def train(
        self,
        dataset,
        num_steps: int = 100,
        batch_size: int = 2
    ):
        """
        Training loop.
        
        This is a simplified demonstration. In production, use SQL-R1's
        full PPO training loop with VERL framework.
        """
        print(f"\n[Trainer] Starting training for {num_steps} steps")
        print(f"[Trainer] Batch size: {batch_size}")
        
        for step in range(num_steps):
            # Sample batch from dataset
            batch_indices = torch.randint(0, len(dataset), (batch_size,))
            batch = {
                'prompts': [dataset[i]['prompt'] for i in batch_indices],
                'ground_truths': [dataset[i]['ground_truth'] for i in batch_indices],
                'schemas': [dataset[i].get('schema') for i in batch_indices]
            }
            
            # Training step
            metrics = self.train_step(batch, step)
            
            # Log metrics
            if step % self.config.logging.log_interval == 0:
                print(f"\n[Step {step}] Metrics:")
                for key, value in metrics.items():
                    print(f"  {key}: {value:.4f}")
                
                if self.use_wandb:
                    wandb.log(metrics, step=step)
            
            # Save checkpoint
            if step % self.config.logging.save_interval == 0 and step > 0:
                checkpoint_path = self.output_dir / f"checkpoint-{step}"
                checkpoint_path.mkdir(parents=True, exist_ok=True)
                self.model.save_pretrained(checkpoint_path)
                self.tokenizer.save_pretrained(checkpoint_path)
                print(f"[Step {step}] Saved checkpoint to {checkpoint_path}")
        
        print("\n[Trainer] Training complete!")
        
        # Save final model
        final_path = self.output_dir / "final_model"
        final_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(final_path)
        self.tokenizer.save_pretrained(final_path)
        print(f"[Trainer] Saved final model to {final_path}")


def create_demo_dataset():
    """Create a small demo dataset for testing."""
    return [
        {
            'prompt': 'Convert to SQL: Show all employees in engineering department',
            'ground_truth': {
                'sql': 'SELECT * FROM employees WHERE department = "engineering"'
            },
            'schema': {
                'tables': {
                    'employees': {
                        'columns': ['id', 'name', 'department', 'salary']
                    }
                }
            }
        },
        {
            'prompt': 'Convert to SQL: Count total number of orders',
            'ground_truth': {
                'sql': 'SELECT COUNT(*) FROM orders'
            },
            'schema': {
                'tables': {
                    'orders': {
                        'columns': ['id', 'customer_id', 'total', 'date']
                    }
                }
            }
        },
        {
            'prompt': 'Convert to SQL: Get average salary by department',
            'ground_truth': {
                'sql': 'SELECT department, AVG(salary) FROM employees GROUP BY department'
            },
            'schema': {
                'tables': {
                    'employees': {
                        'columns': ['id', 'name', 'department', 'salary']
                    }
                }
            }
        }
    ]


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train SQL-R1 with enhanced rewards on Colab")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Coder-3B-Instruct",
                        help="Model name or path")
    parser.add_argument("--config", type=str, default="configs/train_24gb.yaml",
                        help="Path to config file")
    parser.add_argument("--output-dir", type=str, default="./outputs",
                        help="Output directory")
    parser.add_argument("--num-steps", type=int, default=100,
                        help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Batch size")
    parser.add_argument("--use-wandb", action="store_true",
                        help="Use Weights & Biases for logging")
    parser.add_argument("--demo", action="store_true",
                        help="Use demo dataset (for testing)")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = SimplifiedTrainer(
        model_name=args.model,
        config_path=args.config,
        output_dir=args.output_dir,
        use_wandb=args.use_wandb
    )
    
    # Load or create dataset
    if args.demo:
        print("[Main] Using demo dataset")
        dataset = create_demo_dataset()
    else:
        print("[Main] Loading dataset...")
        # In production, load actual Text-to-SQL dataset
        # For now, use demo dataset
        dataset = create_demo_dataset()
    
    # Train
    trainer.train(
        dataset=dataset,
        num_steps=args.num_steps,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
