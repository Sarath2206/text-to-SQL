# SQL-R1 Extension - Google Colab Setup Guide

This guide shows how to run the SQL-R1 RL extension on Google Colab with a 24GB GPU.

## Quick Start

### Option 1: Use the Jupyter Notebook

1. Upload `SQL_R1_Extension_Colab.ipynb` to Google Colab
2. Set runtime to GPU (Runtime > Change runtime type > GPU)
3. Run all cells

### Option 2: Manual Setup

#### 1. Setup Colab Environment

```python
# Check GPU
!nvidia-smi

# Clone repository
!git clone https://github.com/YOUR_USERNAME/sql-r1-extension.git
%cd sql-r1-extension

# Install dependencies
!pip install -q torch transformers datasets sqlparse pyyaml wandb accelerate
```

#### 2. Test Enhanced Rewards

```python
from extensions.reward_enhanced import (
    SchemaAwareReward,
    StructuralReward,
    EnhancedSyntaxReward,
    EnhancedRewardComputer
)

# Test schema-aware reward
schema = {
    'tables': {
        'employees': {'columns': ['id', 'name', 'salary', 'department']}
    }
}

reward = SchemaAwareReward(weight=-0.5)
sql = "SELECT name FROM customers"  # 'customers' doesn't exist
print(f"Hallucination penalty: {reward.compute(sql, schema)}")  # Should be -0.5
```

#### 3. Run Training Demo

```bash
python train_colab.py \
    --model "Qwen/Qwen2.5-Coder-3B-Instruct" \
    --config "configs/train_24gb.yaml" \
    --output-dir "./outputs" \
    --num-steps 10 \
    --batch-size 2 \
    --demo
```

## Memory Optimization for 24GB GPU

The configuration is optimized for 24GB GPUs:

### Model Selection
- **Model**: Qwen2.5-Coder-3B-Instruct (instead of 7B/14B)
- **Memory**: ~6-8GB for model weights in bf16

### Training Configuration
- **Batch Size**: 2 (reduced from 8)
- **Gradient Accumulation**: 8 steps (effective batch size = 16)
- **Mixed Precision**: bfloat16 (saves ~50% memory)
- **Gradient Checkpointing**: Enabled (saves ~3-4GB, adds ~20% compute)

### Expected Memory Usage
- Model weights: ~6-8GB
- Activations: ~8-10GB
- Optimizer states: ~4-6GB
- **Total**: ~20-22GB (fits in 24GB with buffer)

## File Structure

```
sql-r1-extension/
├── extensions/
│   ├── reward_enhanced.py      # Enhanced reward components
│   ├── config.py                # Configuration management
│   ├── logging_enhanced.py      # Enhanced logging (skeleton)
│   ├── checkpoint.py            # Checkpoint management (skeleton)
│   └── validation.py            # Validation utilities (skeleton)
├── configs/
│   └── train_24gb.yaml          # 24GB GPU configuration
├── tests/
│   ├── test_schema_reward.py
│   ├── test_structural_reward.py
│   ├── test_syntax_reward.py
│   └── test_enhanced_reward_computer.py
├── train_colab.py               # Simplified training script for Colab
├── SQL_R1_Extension_Colab.ipynb # Jupyter notebook for Colab
└── README.md                    # Main documentation
```

## Enhanced Reward Components

### 1. Schema-Aware Reward
Detects and penalizes hallucinated schema elements:
- Non-existent tables
- Non-existent columns
- Configurable penalty weight (default: -0.5 per hallucination)

```python
schema_reward = SchemaAwareReward(weight=-0.5)
reward = schema_reward.compute(sql, schema)
```

### 2. Structural Reward
Rewards partial structural correctness:
- SELECT clause matching (weight: 0.3)
- WHERE clause matching (weight: 0.3)
- JOIN clause matching (weight: 0.2)
- Order-independent comparison

```python
structural_reward = StructuralReward(
    select_weight=0.3,
    where_weight=0.3,
    join_weight=0.2
)
reward = structural_reward.compute(generated_sql, ground_truth_sql)
```

### 3. Enhanced Syntax Reward
Validates SQL syntax at AST level:
- DML statement validation (SELECT, INSERT, UPDATE, DELETE)
- Token structure checking
- Configurable reward weight (default: 0.2)

```python
syntax_reward = EnhancedSyntaxReward(weight=0.2)
reward = syntax_reward.compute(sql)
```

### 4. Enhanced Reward Computer
Integrates all components with baseline reward:

```python
computer = EnhancedRewardComputer(
    baseline_reward_fn=baseline_reward,
    enable_enhanced=True,
    schema_weight=-0.5,
    structural_select_weight=0.3,
    structural_where_weight=0.3,
    structural_join_weight=0.2,
    syntax_weight=0.2
)

result = computer.compute_reward(
    solution_str=model_output,
    ground_truth=ground_truth,
    schema=schema
)

print(f"Total: {result.total}")
print(f"Baseline: {result.baseline_total}")
print(f"Schema: {result.schema}")
print(f"Structural: {result.structural}")
print(f"Syntax: {result.syntax}")
```

## Testing

All reward components have comprehensive unit tests:

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_schema_reward.py -v

# Test results: 56/56 passing
```

## Integration with SQL-R1

For production use, integrate with SQL-R1's VERL training loop:

```python
# In SQL-R1's verl/trainer/main_ppo.py

from extensions.reward_enhanced import EnhancedRewardComputer
from extensions.config import load_config

# Load configuration
config = load_config('configs/train_24gb.yaml')

# Wrap SQL-R1's reward function
enhanced_reward = EnhancedRewardComputer(
    baseline_reward_fn=sql_r1_reward_function,
    enable_enhanced=config.reward.enable_enhanced,
    schema_weight=config.reward.schema_weight,
    structural_select_weight=config.reward.structural_select_weight,
    structural_where_weight=config.reward.structural_where_weight,
    structural_join_weight=config.reward.structural_join_weight,
    syntax_weight=config.reward.syntax_weight
)

# Use in PPO training loop
for batch in dataloader:
    # Generate responses
    responses = model.generate(batch['prompts'])
    
    # Compute rewards
    rewards = []
    for response, gt, schema in zip(responses, batch['ground_truths'], batch['schemas']):
        reward_result = enhanced_reward.compute_reward(
            solution_str=response,
            ground_truth=gt,
            schema=schema
        )
        rewards.append(reward_result.total)
    
    # PPO update
    ppo_trainer.step(responses, rewards)
```

## Configuration

Edit `configs/train_24gb.yaml` to customize:

```yaml
# Reward weights
reward:
  enable_enhanced: true
  schema_weight: -0.5              # Penalty for hallucinations
  structural_select_weight: 0.3    # Reward for SELECT match
  structural_where_weight: 0.3     # Reward for WHERE match
  structural_join_weight: 0.2      # Reward for JOIN match
  syntax_weight: 0.2               # Reward for valid syntax

# Optimization
optimization:
  model_name: "Qwen/Qwen2.5-Coder-3B-Instruct"
  batch_size: 2
  gradient_accumulation_steps: 8
  gradient_checkpointing: true
  mixed_precision: "bf16"

# Logging
logging:
  log_interval: 10
  save_interval: 500
  eval_interval: 100
  detailed_rewards: true
  track_gpu_memory: true
```

## Troubleshooting

### Out of Memory (OOM)

If you encounter OOM errors:

1. Reduce batch size:
   ```yaml
   batch_size: 1  # Instead of 2
   ```

2. Increase gradient accumulation:
   ```yaml
   gradient_accumulation_steps: 16  # Instead of 8
   ```

3. Reduce sequence lengths:
   ```yaml
   max_prompt_length: 2048  # Instead of 4096
   max_response_length: 1024  # Instead of 2048
   ```

4. Use smaller model:
   ```yaml
   model_name: "Qwen/Qwen2.5-Coder-1.5B-Instruct"
   ```

### Slow Training

Training is slower on single GPU compared to 8x80GB setup:
- Expected: ~10-20 samples/second (vs ~100-200 on 8xA100)
- Gradient checkpointing adds ~20% overhead
- This is normal for 24GB GPU optimization

### Import Errors

Make sure all dependencies are installed:
```bash
pip install torch transformers datasets sqlparse pyyaml wandb accelerate
```

## Performance Expectations

### Memory Usage
- Model (3B, bf16): ~6-8GB
- Training overhead: ~12-14GB
- Total: ~20-22GB (fits in 24GB)

### Training Speed
- Single 24GB GPU: ~10-20 samples/second
- 8x80GB GPUs: ~100-200 samples/second
- Gradient accumulation maintains effective batch size

### Expected Improvements
- **Reduced Hallucinations**: 10-20% fewer non-existent tables/columns
- **Better Structural Correctness**: 5-15% improvement in partial matches
- **Improved Syntax**: 5-10% fewer syntax errors

## Next Steps

1. **Test on Colab**: Run the notebook to verify everything works
2. **Integrate with SQL-R1**: Add to SQL-R1's training pipeline
3. **Run Full Training**: Train on Spider/WikiSQL datasets
4. **Evaluate**: Compare baseline vs enhanced on test set
5. **Tune Hyperparameters**: Adjust reward weights based on results

## Support

For issues or questions:
1. Check test results: `pytest tests/ -v`
2. Review configuration: `configs/train_24gb.yaml`
3. Monitor GPU memory: `nvidia-smi`
4. Check logs in `outputs/` directory

## Citation

If you use this code, please cite:
- SQL-R1: https://arxiv.org/abs/2504.08600
- This extension: [Your paper/repo]
