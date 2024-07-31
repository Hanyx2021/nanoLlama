"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass
from typing import Tuple,Optional
import jittor as jt
from jittor import nn

ParallelEmbedding = jt.nn.Embedding
RowParallelLinear = jt.nn.Linear
ColumnParallelLinear = jt.nn.Linear
    
@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5
    max_batch_size: int = 32
    max_seq_len: int = 2048


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = jt.ones(dim)

    def _norm(self, x):
        return x / jt.sqrt(x.pow(2).mean(-1, keepdims=True) + self.eps)

    def execute(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (jt.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = jt.arange(end)  # type: ignore
    freqs = jt.outer(t, freqs).float()  # type: ignore
    freqs_cis = jt.concat([jt.cos(freqs).unsqueeze(dim=-1), jt.sin(freqs).unsqueeze(dim=-1)], dim=-1)
    return freqs_cis


def reshape_for_broadcast(freqs_cis: jt.Var, x: jt.Var):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-2], 2)
    shape = [d if i == 1 or i == ndim - 2 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(
    xq: jt.Var,
    xk: jt.Var,
    freqs_cis: jt.Var,
) -> Tuple[jt.Var, jt.Var]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)

    xq_out = jt.concat([xq_[:, :, :, :, 0] * freqs_cis[:, :, :, :, 0] - xq_[:, :, :, :, 1] * freqs_cis[:, :, :, :, 1], 
                        xq_[:, :, :, :, 0] * freqs_cis[:, :, :, :, 1] + xq_[:, :, :, :, 1] * freqs_cis[:, :, :, :, 0]], dim=-1).flatten(3)
    xk_out = jt.concat([xk_[:, :, :, :, 0] * freqs_cis[:, :, :, :, 0] - xk_[:, :, :, :, 1] * freqs_cis[:, :, :, :, 1], 
                        xk_[:, :, :, :, 0] * freqs_cis[:, :, :, :, 1] + xk_[:, :, :, :, 1] * freqs_cis[:, :, :, :, 0]], dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_local_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False
        )
        self.wk = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False
        )
        self.wv = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False
        )
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False
        )

        self.cache_k = jt.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)
        )
        self.cache_v = jt.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)
        )

    def execute(self, x: jt.Var, start_pos: int, freqs_cis: jt.Var, mask: Optional[jt.Var]):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)

        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = values.transpose(
            1, 2
        )  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        scores = jt.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
        scores = jt.nn.softmax(scores.float(), dim=-1).type_as(xq)
        output = jt.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)
        output = output.transpose(
            1, 2
        ).contiguous().view(bsz, seqlen, -1)

        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ColumnParallelLinear(
            dim, hidden_dim, bias=False
        )
        self.w2 = RowParallelLinear(
            hidden_dim, dim, bias=False
        )
        self.w3 = ColumnParallelLinear(
            dim, hidden_dim, bias=False
        )

    def execute(self, x):
        return self.w2(jt.nn.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def execute(self, x: jt.Var, start_pos: int, freqs_cis: jt.Var, mask: Optional[jt.Var]):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = ParallelEmbedding(
            params.vocab_size, params.dim
        )

        self.layers = nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = ColumnParallelLinear(
            params.dim, params.vocab_size, bias=False
        )

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )
        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.tok_embeddings.weight.numel()
        return n_params

    def execute(self, tokens: jt.Var, start_pos: int, targets=None):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        mask = None
        if seqlen > 1:
            mask = jt.full((1,1,seqlen, seqlen), float("-inf"))
            mask = jt.misc.triu(mask, diagonal=start_pos + 1).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)

        if targets is not None:
            logits = self.output(h)
            loss = jt.nn.cross_entropy_loss(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.output(h[:, -1, :])
            loss = None
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(jt.optim.AdamW).parameters
        use_fused = fused_available # and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = jt.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    with jt.no_grad():
        def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
            """
            Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
            the sequence max_new_tokens times, feeding the predictions back into the model each time.
            Most likely you'll want to make sure to be in model.eval() mode of operation for this.
            """
            for _ in range(max_new_tokens):
                # if the sequence context is growing too long we must crop it at block_size
                idx_cond = idx if idx.size(1) <= self.params.max_seq_len else idx[:, -self.params.max_seq_len:]
                # forward the model to get the logits for the index in the sequence
                logits, _ = self(idx_cond, 0)
                # pluck the logits at the final step and scale by desired temperature
                logits = logits / temperature
                # optionally crop the logits to only the top k options
                if top_k is not None:
                    v, _ = jt.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, -1].unsqueeze(-1)] = -float('Inf')
                # apply softmax to convert logits to (normalized) probabilities
                probs = nn.softmax(logits, dim=-1)
                # sample from the distribution
                idx_next = jt.multinomial(probs, num_samples=1)
                # append sampled index to the running sequence and continue
                idx = jt.concat((idx, idx_next), dim=1)

            return idx


if __name__ == "__main__":
    jt.flags.use_acl = 1
    model_args = dict(dim = 512,n_layers = 10,n_heads = 4,vocab_size = 50304,multiple_of = 8,norm_eps = 1e-6,
                  max_batch_size = 100,max_seq_len = 256) # start with model_args from command line
    model = Transformer(ModelArgs(**model_args))
    print(model)
    model.train()
    tokens = jt.ones(2, 512)
    targets = jt.ones(2, 512)
    for i in range(10):
        logits, loss = model(tokens, 0, targets)
        optimizer = model.configure_optimizers(0.1, 3e-4, (0.9, 0.95))
        optimizer.step(loss)
        print(model.state_dict()['layers.0.attention_norm.weight'])
        print("done")