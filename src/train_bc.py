#!/usr/bin/env python3
"""Train behavioral cloning model from recorded demonstrations.

This script trains a neural network policy using behavioral cloning (imitation learning)
from recorded demonstration data using a simple PyTorch implementation.

Usage:
    python train_bc.py demos/recording.npz                    # Train with defaults
    python train_bc.py demos/recording.npz --epochs 100       # Custom epochs
    python train_bc.py demos/recording.npz --successful-only  # Train only on successful demos
    python train_bc.py demos/recording.npz --output model.pt  # Custom output path
"""

import argparse
import numpy as np
from pathlib import Path

from demo_utils import load_demo_data


def train_simple_pytorch(observations, actions, args):
    """Simple behavioral cloning using PyTorch directly."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    print("\nUsing simple PyTorch behavioral cloning...")

    # Define policy network
    class BCPolicy(nn.Module):
        def __init__(self, obs_dim, action_dim, hidden_dim=256):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
                nn.Tanh()  # Actions normalized to [-1, 1]
            )

        def forward(self, x):
            return self.net(x)

    obs_dim = observations.shape[1]
    action_dim = actions.shape[1]

    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    policy = BCPolicy(obs_dim, action_dim).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # Create dataloader
    dataset = TensorDataset(
        torch.FloatTensor(observations),
        torch.FloatTensor(actions)
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Dataset size: {len(dataset)}")
    print()

    for epoch in range(args.epochs):
        total_loss = 0
        num_batches = 0

        for obs_batch, act_batch in loader:
            obs_batch = obs_batch.to(device)
            act_batch = act_batch.to(device)

            optimizer.zero_grad()
            pred = policy(obs_batch)
            loss = criterion(pred, act_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:4d}/{args.epochs}, Loss: {avg_loss:.6f}")

    # Save model
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save with metadata
    save_dict = {
        'model_state_dict': policy.state_dict(),
        'obs_dim': obs_dim,
        'action_dim': action_dim,
        'hidden_dim': 256,
    }
    torch.save(save_dict, str(output_path))
    print(f"\nModel saved to {output_path}")

    # Print model summary
    total_params = sum(p.numel() for p in policy.parameters())
    print(f"Total parameters: {total_params:,}")


def main():
    parser = argparse.ArgumentParser(
        description='Train behavioral cloning model from demonstrations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s demos/recording.npz
      Train with default settings

  %(prog)s demos/recording.npz --epochs 100 --batch-size 128
      Train with custom hyperparameters

  %(prog)s demos/recording.npz --successful-only
      Train only on successful demonstrations

  %(prog)s demos/recording.npz --output models/expert_policy.pt
      Save model to custom path
        """
    )
    parser.add_argument('demo_file', help='Path to .npz demo file')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Training batch size (default: 64)')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4)')
    parser.add_argument('--output', type=str, default='models/bc_policy.pt',
                       help='Output model path (default: models/bc_policy.pt)')
    parser.add_argument('--successful-only', action='store_true',
                       help='Only train on successful episodes')

    args = parser.parse_args()

    print("="*60)
    print("Behavioral Cloning Training")
    print("="*60)

    # Load data
    observations, actions = load_demo_data(args.demo_file, args.successful_only)

    train_simple_pytorch(observations, actions, args)

    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)


if __name__ == '__main__':
    main()
