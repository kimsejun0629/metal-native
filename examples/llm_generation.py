#!/usr/bin/env python3
"""Simple autoregressive text generation with a mini GPT-like model.

This example demonstrates building a small transformer-based language model
for text generation using MetalNative. The model includes:
- Token embedding
- 2 transformer blocks (attention + FFN + layer norm)
- Language modeling head
- Autoregressive sampling
"""

import argparse
import time
from typing import List, Tuple
import numpy as np

try:
    import metal_native as mn
except ImportError:
    print("Error: metal_native is required for this example")
    print("Install with: pip install -e .")
    exit(1)


class SimpleTransformerBlock:
    """Simple transformer block with self-attention and FFN."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        """Initialize transformer block.

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward dimension
        """
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.head_dim = d_model // n_heads

        # Attention weights
        self.q_proj = self._init_weight(d_model, d_model)
        self.k_proj = self._init_weight(d_model, d_model)
        self.v_proj = self._init_weight(d_model, d_model)
        self.o_proj = self._init_weight(d_model, d_model)

        # FFN weights
        self.ff_w1 = self._init_weight(d_model, d_ff)
        self.ff_w2 = self._init_weight(d_ff, d_model)

        # Layer norm parameters
        self.ln1_gamma = np.ones(d_model, dtype=np.float32)
        self.ln1_beta = np.zeros(d_model, dtype=np.float32)
        self.ln2_gamma = np.ones(d_model, dtype=np.float32)
        self.ln2_beta = np.zeros(d_model, dtype=np.float32)

    def _init_weight(self, in_dim: int, out_dim: int) -> np.ndarray:
        """Initialize weight matrix."""
        std = np.sqrt(2.0 / in_dim)
        return np.random.randn(in_dim, out_dim).astype(np.float32) * std

    def layer_norm(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """Apply layer normalization."""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return gamma * (x - mean) / np.sqrt(var + 1e-5) + beta

    def attention(self, x: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """Self-attention with causal masking.

        Args:
            x: Input [batch, seq_len, d_model]
            mask: Attention mask [seq_len, seq_len] (optional)

        Returns:
            Attention output [batch, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = x @ self.q_proj
        k = x @ self.k_proj
        v = x @ self.v_proj

        # Reshape for multi-head attention
        q = q.reshape(batch_size, seq_len, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Scaled dot-product attention
        scale = 1.0 / np.sqrt(self.head_dim)
        scores = (q @ k.transpose(0, 1, 3, 2)) * scale

        # Apply causal mask
        if mask is not None:
            scores = scores + mask

        # Softmax
        scores = scores - np.max(scores, axis=-1, keepdims=True)
        attn_weights = np.exp(scores)
        attn_weights = attn_weights / (np.sum(attn_weights, axis=-1, keepdims=True) + 1e-8)

        # Apply attention to values
        attn_out = attn_weights @ v

        # Reshape and project
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        output = attn_out @ self.o_proj

        return output

    def feed_forward(self, x: np.ndarray) -> np.ndarray:
        """Feed-forward network with GELU activation."""
        h = x @ self.ff_w1
        # GELU activation
        h = 0.5 * h * (1 + np.tanh(np.sqrt(2 / np.pi) * (h + 0.044715 * h ** 3)))
        output = h @ self.ff_w2
        return output

    def forward(self, x: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """Forward pass through transformer block."""
        # Attention sub-layer
        residual = x
        x = self.layer_norm(x, self.ln1_gamma, self.ln1_beta)
        x = residual + self.attention(x, mask)

        # FFN sub-layer
        residual = x
        x = self.layer_norm(x, self.ln2_gamma, self.ln2_beta)
        x = residual + self.feed_forward(x)

        return x


class MiniGPT:
    """Mini GPT-like model for text generation."""

    def __init__(self, vocab_size: int, d_model: int, n_heads: int, d_ff: int,
                 n_layers: int, max_seq_len: int):
        """Initialize mini GPT model.

        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward dimension
            n_layers: Number of transformer layers
            max_seq_len: Maximum sequence length
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len

        # Token embedding
        self.token_embedding = np.random.randn(vocab_size, d_model).astype(np.float32) * 0.02

        # Position embedding
        self.position_embedding = np.random.randn(max_seq_len, d_model).astype(np.float32) * 0.02

        # Transformer blocks
        self.blocks = [
            SimpleTransformerBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ]

        # Language modeling head
        self.lm_head = np.random.randn(d_model, vocab_size).astype(np.float32) * 0.02

        # Final layer norm
        self.ln_final_gamma = np.ones(d_model, dtype=np.float32)
        self.ln_final_beta = np.zeros(d_model, dtype=np.float32)

    def get_causal_mask(self, seq_len: int) -> np.ndarray:
        """Create causal attention mask.

        Args:
            seq_len: Sequence length

        Returns:
            Mask [1, 1, seq_len, seq_len] with -inf for future positions
        """
        mask = np.triu(np.full((seq_len, seq_len), -1e9, dtype=np.float32), k=1)
        return mask[None, None, :, :]

    def forward(self, input_ids: np.ndarray) -> np.ndarray:
        """Forward pass through the model.

        Args:
            input_ids: Token IDs [batch, seq_len]

        Returns:
            Logits [batch, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape

        # Embed tokens and positions
        token_emb = self.token_embedding[input_ids]  # [batch, seq_len, d_model]
        pos_emb = self.position_embedding[:seq_len]  # [seq_len, d_model]
        x = token_emb + pos_emb[None, :, :]

        # Create causal mask
        mask = self.get_causal_mask(seq_len)

        # Apply transformer blocks
        for block in self.blocks:
            x = block.forward(x, mask)

        # Final layer norm
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x = self.ln_final_gamma * (x - mean) / np.sqrt(var + 1e-5) + self.ln_final_beta

        # Language modeling head
        logits = x @ self.lm_head

        return logits


def sample_token(logits: np.ndarray, temperature: float = 1.0, top_k: int = 50) -> int:
    """Sample next token from logits.

    Args:
        logits: Logits for next token [vocab_size]
        temperature: Sampling temperature
        top_k: Number of top tokens to consider (0 = all tokens)

    Returns:
        Sampled token ID
    """
    # Apply temperature
    logits = logits / temperature

    # Top-k filtering
    if top_k > 0:
        indices_to_remove = logits < np.partition(logits, -top_k)[-top_k]
        logits[indices_to_remove] = -float('inf')

    # Softmax
    logits = logits - np.max(logits)
    probs = np.exp(logits)
    probs = probs / np.sum(probs)

    # Sample
    token = np.random.choice(len(probs), p=probs)
    return token


def generate_text(model: MiniGPT, prompt_ids: List[int], max_new_tokens: int,
                  temperature: float = 1.0, top_k: int = 50) -> List[int]:
    """Generate text autoregressively.

    Args:
        model: MiniGPT model
        prompt_ids: Prompt token IDs
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling parameter

    Returns:
        List of generated token IDs (including prompt)
    """
    generated_ids = prompt_ids.copy()

    for _ in range(max_new_tokens):
        # Prepare input (last max_seq_len tokens)
        input_ids = generated_ids[-model.max_seq_len:]
        input_ids = np.array([input_ids], dtype=np.int64)

        # Forward pass
        logits = model.forward(input_ids)

        # Get logits for last position
        next_token_logits = logits[0, -1, :]

        # Sample next token
        next_token = sample_token(next_token_logits, temperature, top_k)

        # Append to sequence
        generated_ids.append(next_token)

        # Stop at EOS token (assuming 0 is EOS)
        if next_token == 0:
            break

    return generated_ids


def create_simple_vocab() -> dict:
    """Create a simple vocabulary for demonstration.

    Returns:
        Dictionary mapping token IDs to strings
    """
    vocab = {
        0: "<EOS>",
        1: "the",
        2: "cat",
        3: "sat",
        4: "on",
        5: "mat",
        6: "dog",
        7: "ran",
        8: "in",
        9: "park",
        10: "quick",
        11: "brown",
        12: "fox",
        13: "jumped",
        14: "over",
        15: "lazy",
    }
    return vocab


def ids_to_text(ids: List[int], vocab: dict) -> str:
    """Convert token IDs to text.

    Args:
        ids: Token IDs
        vocab: Vocabulary mapping

    Returns:
        Text string
    """
    tokens = [vocab.get(idx, f"<UNK:{idx}>") for idx in ids]
    return " ".join(tokens)


def main():
    """Main generation example."""
    parser = argparse.ArgumentParser(description='Text generation with mini GPT')
    parser.add_argument('--vocab-size', type=int, default=100,
                       help='Vocabulary size')
    parser.add_argument('--d-model', type=int, default=128,
                       help='Model dimension')
    parser.add_argument('--n-heads', type=int, default=4,
                       help='Number of attention heads')
    parser.add_argument('--d-ff', type=int, default=512,
                       help='Feed-forward dimension')
    parser.add_argument('--n-layers', type=int, default=2,
                       help='Number of transformer layers')
    parser.add_argument('--max-seq-len', type=int, default=128,
                       help='Maximum sequence length')
    parser.add_argument('--max-new-tokens', type=int, default=20,
                       help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Sampling temperature')
    parser.add_argument('--top-k', type=int, default=10,
                       help='Top-k sampling')

    args = parser.parse_args()

    print("=" * 80)
    print("Autoregressive Text Generation with MetalNative")
    print("=" * 80)
    print(f"Model: d_model={args.d_model}, n_heads={args.n_heads}, n_layers={args.n_layers}")
    print(f"Vocab size: {args.vocab_size}")
    print(f"Max sequence length: {args.max_seq_len}")
    print(f"Generation: max_tokens={args.max_new_tokens}, temp={args.temperature}, top_k={args.top_k}")
    print("=" * 80)

    # Check MetalNative availability
    if not mn.is_available():
        print("Error: Metal is not available on this system")
        return

    print(f"Using device: {mn.device_name()}")
    print()

    # Create model
    print("Creating mini GPT model...")
    model = MiniGPT(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        n_layers=args.n_layers,
        max_seq_len=args.max_seq_len
    )
    print("Model created successfully")
    print()

    # Create simple vocabulary
    vocab = create_simple_vocab()

    # Create prompt
    prompt_ids = [1, 2, 3, 4, 5]  # "the cat sat on mat"
    prompt_text = ids_to_text(prompt_ids, vocab)
    print(f"Prompt: {prompt_text}")
    print()

    # Generate text
    print("Generating text...")
    start_time = time.time()

    generated_ids = generate_text(
        model, prompt_ids, args.max_new_tokens, args.temperature, args.top_k
    )

    generation_time = time.time() - start_time

    # Convert to text
    generated_text = ids_to_text(generated_ids, vocab)

    print("-" * 80)
    print("Generated text:")
    print(generated_text)
    print("-" * 80)
    print()
    print(f"Generated {len(generated_ids) - len(prompt_ids)} tokens in {generation_time:.2f}s")
    print(f"Throughput: {(len(generated_ids) - len(prompt_ids)) / generation_time:.2f} tokens/s")


if __name__ == "__main__":
    main()
