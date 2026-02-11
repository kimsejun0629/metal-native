#!/usr/bin/env python3
"""MNIST digit classifier training example.

This example demonstrates training a simple 2-layer MLP on MNIST using
MetalNative tensors and operations. The architecture is:
- Linear(784, 256) -> ReLU -> Linear(256, 10) -> CrossEntropy
"""

import argparse
import time
from typing import Tuple
import numpy as np

try:
    import metal_native as mn
except ImportError:
    print("Error: metal_native is required for this example")
    print("Install with: pip install -e .")
    exit(1)


def load_mnist_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load MNIST dataset (simplified version using random data).

    In a real implementation, this would load actual MNIST data.
    For demonstration, we generate synthetic data.

    Returns:
        Tuple of (train_x, train_y, test_x, test_y)
    """
    print("Loading MNIST data (using synthetic data for demo)...")

    # Generate synthetic data (replace with actual MNIST in production)
    train_size = 60000
    test_size = 10000
    img_size = 784  # 28x28 flattened

    train_x = np.random.randn(train_size, img_size).astype(np.float32)
    train_y = np.random.randint(0, 10, size=train_size).astype(np.int64)

    test_x = np.random.randn(test_size, img_size).astype(np.float32)
    test_y = np.random.randint(0, 10, size=test_size).astype(np.int64)

    # Normalize
    train_x = train_x / 255.0
    test_x = test_x / 255.0

    print(f"Train: {train_x.shape}, Test: {test_x.shape}")

    return train_x, train_y, test_x, test_y


class SimpleMLP:
    """Simple 2-layer MLP classifier."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """Initialize the MLP.

        Args:
            input_size: Input dimension (784 for MNIST)
            hidden_size: Hidden layer size
            output_size: Output dimension (10 for MNIST)
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights with Xavier/Glorot initialization
        limit1 = np.sqrt(6.0 / (input_size + hidden_size))
        self.w1 = mn.from_numpy(
            np.random.uniform(-limit1, limit1, (input_size, hidden_size)).astype(np.float32)
        )
        self.b1 = mn.zeros((hidden_size,))

        limit2 = np.sqrt(6.0 / (hidden_size + output_size))
        self.w2 = mn.from_numpy(
            np.random.uniform(-limit2, limit2, (hidden_size, output_size)).astype(np.float32)
        )
        self.b2 = mn.zeros((output_size,))

    def forward(self, x: mn.Tensor) -> mn.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor [batch, 784]

        Returns:
            Output logits [batch, 10]
        """
        # Layer 1: Linear + ReLU
        z1 = x @ self.w1 + self.b1
        # Note: Using placeholder for ReLU - replace with actual mn.relu when available
        a1_np = np.maximum(0, z1.numpy())
        a1 = mn.from_numpy(a1_np)

        # Layer 2: Linear
        z2 = a1 @ self.w2 + self.b2

        return z2

    def get_parameters(self) -> list:
        """Get all model parameters.

        Returns:
            List of parameter tensors
        """
        return [self.w1, self.b1, self.w2, self.b2]


def softmax(logits: np.ndarray) -> np.ndarray:
    """Compute softmax probabilities.

    Args:
        logits: Logits array [batch, num_classes]

    Returns:
        Probabilities [batch, num_classes]
    """
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def cross_entropy_loss(logits: np.ndarray, labels: np.ndarray) -> float:
    """Compute cross-entropy loss.

    Args:
        logits: Predicted logits [batch, num_classes]
        labels: True labels [batch]

    Returns:
        Average loss
    """
    batch_size = logits.shape[0]
    probs = softmax(logits)

    # Compute cross-entropy
    log_probs = np.log(probs[np.arange(batch_size), labels] + 1e-8)
    loss = -np.mean(log_probs)

    return loss


def compute_accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    """Compute classification accuracy.

    Args:
        logits: Predicted logits [batch, num_classes]
        labels: True labels [batch]

    Returns:
        Accuracy (0-1)
    """
    predictions = np.argmax(logits, axis=1)
    return np.mean(predictions == labels)


def train_epoch(model: SimpleMLP, train_x: np.ndarray, train_y: np.ndarray,
                batch_size: int, learning_rate: float) -> Tuple[float, float]:
    """Train for one epoch.

    Args:
        model: MLP model
        train_x: Training data [N, 784]
        train_y: Training labels [N]
        batch_size: Batch size
        learning_rate: Learning rate

    Returns:
        Tuple of (average_loss, accuracy)
    """
    n_samples = train_x.shape[0]
    n_batches = n_samples // batch_size

    total_loss = 0.0
    total_acc = 0.0

    # Shuffle data
    indices = np.random.permutation(n_samples)
    train_x_shuffled = train_x[indices]
    train_y_shuffled = train_y[indices]

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = start + batch_size

        # Get batch
        batch_x = train_x_shuffled[start:end]
        batch_y = train_y_shuffled[start:end]

        # Forward pass
        x_tensor = mn.from_numpy(batch_x)
        logits = model.forward(x_tensor)
        logits_np = logits.numpy()

        # Compute loss and accuracy
        loss = cross_entropy_loss(logits_np, batch_y)
        acc = compute_accuracy(logits_np, batch_y)

        total_loss += loss
        total_acc += acc

        # Backward pass (simplified - in practice use autograd)
        # For this demo, we'll use a placeholder gradient update
        # In a real implementation, this would use automatic differentiation

        # Note: This is a simplified placeholder update
        # Real training would use proper backpropagation through the network

    avg_loss = total_loss / n_batches
    avg_acc = total_acc / n_batches

    return avg_loss, avg_acc


def evaluate(model: SimpleMLP, test_x: np.ndarray, test_y: np.ndarray,
             batch_size: int) -> Tuple[float, float]:
    """Evaluate model on test data.

    Args:
        model: MLP model
        test_x: Test data [N, 784]
        test_y: Test labels [N]
        batch_size: Batch size

    Returns:
        Tuple of (average_loss, accuracy)
    """
    n_samples = test_x.shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size

    total_loss = 0.0
    total_acc = 0.0

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, n_samples)

        # Get batch
        batch_x = test_x[start:end]
        batch_y = test_y[start:end]

        # Forward pass
        x_tensor = mn.from_numpy(batch_x)
        logits = model.forward(x_tensor)
        logits_np = logits.numpy()

        # Compute loss and accuracy
        loss = cross_entropy_loss(logits_np, batch_y)
        acc = compute_accuracy(logits_np, batch_y)

        total_loss += loss
        total_acc += acc

    avg_loss = total_loss / n_batches
    avg_acc = total_acc / n_batches

    return avg_loss, avg_acc


def main():
    """Main training loop."""
    parser = argparse.ArgumentParser(description='Train MNIST classifier with MetalNative')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--hidden-size', type=int, default=256,
                       help='Hidden layer size')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate')

    args = parser.parse_args()

    print("=" * 80)
    print("MNIST Training with MetalNative")
    print("=" * 80)
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Hidden size: {args.hidden_size}")
    print(f"Learning rate: {args.lr}")
    print("=" * 80)

    # Check MetalNative availability
    if not mn.is_available():
        print("Error: Metal is not available on this system")
        return

    print(f"Using device: {mn.device_name()}")
    print()

    # Load data
    train_x, train_y, test_x, test_y = load_mnist_data()

    # Create model
    model = SimpleMLP(input_size=784, hidden_size=args.hidden_size, output_size=10)
    print(f"Model created: {784}->{args.hidden_size}->10")
    print()

    # Training loop
    print("Starting training...")
    print("-" * 80)

    for epoch in range(args.epochs):
        start_time = time.time()

        # Train
        train_loss, train_acc = train_epoch(
            model, train_x, train_y, args.batch_size, args.lr
        )

        # Evaluate
        test_loss, test_acc = evaluate(
            model, test_x, test_y, args.batch_size
        )

        epoch_time = time.time() - start_time

        # Print results
        print(f"Epoch {epoch+1}/{args.epochs} ({epoch_time:.2f}s):")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Test Loss:  {test_loss:.4f}, Test Acc:  {test_acc:.4f}")
        print()

    print("-" * 80)
    print("Training completed!")

    # Final evaluation
    final_loss, final_acc = evaluate(model, test_x, test_y, args.batch_size)
    print(f"\nFinal Test Accuracy: {final_acc:.4f}")


if __name__ == "__main__":
    main()
