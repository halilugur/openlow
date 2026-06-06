// WGSL Shaders for WebGPU LLM Engine

export const WTE_SHADER = `
  @group(0) @binding(0) var<storage, read> tokens: array<u32>;
  @group(0) @binding(1) var<storage, read> weights: array<f32>;
  @group(0) @binding(2) var<storage, read_write> output: array<f32>;

  struct Params {
    n_embd: u32,
    vocab_size: u32,
  }
  @group(0) @binding(3) var<uniform> params: Params;

  @compute @workgroup_size(64)
  fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let token_idx = idx / params.n_embd;
    let emb_dim = idx % params.n_embd;
    
    let token = tokens[token_idx];
    if (token < params.vocab_size) {
      output[idx] = weights[token * params.n_embd + emb_dim];
    } else {
      output[idx] = 0.0;
    }
  }
`;

export const WPE_SHADER = `
  @group(0) @binding(0) var<storage, read> positions: array<u32>;
  @group(0) @binding(1) var<storage, read> weights: array<f32>;
  @group(0) @binding(2) var<storage, read_write> output: array<f32>;

  struct Params {
    n_embd: u32,
    batch_size: u32,
    seq_len: u32,
  }
  @group(0) @binding(3) var<uniform> params: Params;

  @compute @workgroup_size(64)
  fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let b = idx / (params.seq_len * params.n_embd);
    let t_emb = idx % (params.seq_len * params.n_embd);
    let t = t_emb / params.n_embd;
    let emb_dim = t_emb % params.n_embd;
    
    let pos = positions[t];
    output[idx] = weights[pos * params.n_embd + emb_dim];
  }
`;

export const ADD_SHADER = `
  @group(0) @binding(0) var<storage, read> a: array<f32>;
  @group(0) @binding(1) var<storage, read> b: array<f32>;
  @group(0) @binding(2) var<storage, read_write> output: array<f32>;

  @compute @workgroup_size(64)
  fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    output[global_id.x] = a[global_id.x] + b[global_id.x];
  }
`;

export const LAYERNORM_SHADER = `
  @group(0) @binding(0) var<storage, read> x: array<f32>;
  @group(0) @binding(1) var<storage, read> gamma: array<f32>;
  @group(0) @binding(2) var<storage, read> beta: array<f32>;
  @group(0) @binding(3) var<storage, read_write> output: array<f32>;

  struct Params {
    C: u32,
    epsilon: f32,
  }
  @group(0) @binding(4) var<uniform> params: Params;

  @compute @workgroup_size(64)
  fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row_idx = global_id.x; // row index (b * T + t)
    
    // 1. Calculate Mean
    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < params.C; i = i + 1u) {
      sum = sum + x[row_idx * params.C + i];
    }
    let mean: f32 = sum / f32(params.C);
    
    // 2. Calculate Variance
    var var_sum: f32 = 0.0;
    for (var i: u32 = 0u; i < params.C; i = i + 1u) {
      let diff = x[row_idx * params.C + i] - mean;
      var_sum = var_sum + diff * diff;
    }
    let variance: f32 = var_sum / f32(params.C);
    
    // 3. Normalize and scale/shift
    let std_inv = 1.0 / sqrt(variance + params.epsilon);
    for (var i: u32 = 0u; i < params.C; i = i + 1u) {
      let idx = row_idx * params.C + i;
      output[idx] = (x[idx] - mean) * std_inv * gamma[i] + beta[i];
    }
  }
`;

export const MATMUL_SHADER = `
  @group(0) @binding(0) var<storage, read> x: array<f32>;
  @group(0) @binding(1) var<storage, read> w: array<f32>;
  @group(0) @binding(2) var<storage, read> bias: array<f32>;
  @group(0) @binding(3) var<storage, read_write> output: array<f32>;

  struct Params {
    M: u32,
    N: u32,
    K: u32,
    has_bias: u32,
  }
  @group(0) @binding(4) var<uniform> params: Params;

  @compute @workgroup_size(16, 16)
  fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let row = id.y;
    let col = id.x;
    
    if (row >= params.M || col >= params.N) {
      return;
    }
    
    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < params.K; i = i + 1u) {
      sum = sum + x[row * params.K + i] * w[i * params.N + col];
    }
    
    if (params.has_bias == 1u) {
      sum = sum + bias[col];
    }
    
    output[row * params.N + col] = sum;
  }
`;

export const GELU_SHADER = `
  @group(0) @binding(0) var<storage, read> x: array<f32>;
  @group(0) @binding(1) var<storage, read_write> output: array<f32>;

  @compute @workgroup_size(64)
  fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let val = x[idx];
    let k = 0.7978845608 * (val + 0.044715 * val * val * val);
    output[idx] = 0.5 * val * (1.0 + tanh(k));
  }
`;

export const ATTENTION_SHADER = `
  @group(0) @binding(0) var<storage, read> q: array<f32>;
  @group(0) @binding(1) var<storage, read> k: array<f32>;
  @group(0) @binding(2) var<storage, read> v: array<f32>;
  @group(0) @binding(3) var<storage, read_write> output: array<f32>;

  struct Params {
    B: u32,
    n_head: u32,
    T: u32,
    head_size: u32,
  }
  @group(0) @binding(4) var<uniform> params: Params;

  @compute @workgroup_size(16, 16)
  fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let tq = id.x;
    let bh = id.y;
    
    let total_bh = params.B * params.n_head;
    if (tq >= params.T || bh >= total_bh) {
      return;
    }
    
    let head_size = params.head_size;
    let T = params.T;
    let scale = 1.0 / sqrt(f32(head_size));
    
    // Support sequence length up to 2048 for interactive canvas
    var scores: array<f32, 2048>;
    
    var max_score: f32 = -1e9;
    for (var tk: u32 = 0u; tk <= tq; tk = tk + 1u) {
      var sum: f32 = 0.0;
      let q_offset = (bh * T + tq) * head_size;
      let k_offset = (bh * T + tk) * head_size;
      for (var d: u32 = 0u; d < head_size; d = d + 1u) {
        sum = sum + q[q_offset + d] * k[k_offset + d];
      }
      let val = sum * scale;
      scores[tk] = val;
      if (val > max_score) {
        max_score = val;
      }
    }
    
    var exp_sum: f32 = 0.0;
    var exps: array<f32, 2048>;
    for (var tk: u32 = 0u; tk <= tq; tk = tk + 1u) {
      let e = exp(scores[tk] - max_score);
      exps[tk] = e;
      exp_sum = exp_sum + e;
    }
    
    let out_offset = (bh * T + tq) * head_size;
    for (var d: u32 = 0u; d < head_size; d = d + 1u) {
      var val_sum: f32 = 0.0;
      for (var tk: u32 = 0u; tk <= tq; tk = tk + 1u) {
        let attn_weight = exps[tk] / exp_sum;
        let v_offset = (bh * T + tk) * head_size;
        val_sum = val_sum + attn_weight * v[v_offset + d];
      }
      output[out_offset + d] = val_sum;
    }
  }
`;

export const SOFTMAX_SHADER = `
  @group(0) @binding(0) var<storage, read> logits: array<f32>;
  @group(0) @binding(1) var<storage, read_write> output: array<f32>;

  struct Params {
    B: u32,
    vocab_size: u32,
  }
  @group(0) @binding(2) var<uniform> params: Params;

  @compute @workgroup_size(64)
  fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let b = global_id.x;
    if (b >= params.B) {
      return;
    }
    
    let vocab_size = params.vocab_size;
    let offset = b * vocab_size;
    
    var max_val: f32 = -1e9;
    for (var i: u32 = 0u; i < vocab_size; i = i + 1u) {
      let val = logits[offset + i];
      if (val > max_val) {
        max_val = val;
      }
    }
    
    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < vocab_size; i = i + 1u) {
      sum = sum + exp(logits[offset + i] - max_val);
    }
    
    for (var i: u32 = 0u; i < vocab_size; i = i + 1u) {
      output[offset + i] = exp(logits[offset + i] - max_val) / sum;
    }
  }
`;
