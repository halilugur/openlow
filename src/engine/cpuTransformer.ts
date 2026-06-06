// A small but fully-correct character-level Transformer with real
// backpropagation and an Adam optimizer, running on the CPU.
//
// Unlike the visual WebGPU graph (which approximates some gradients for
// teaching purposes), this module's forward and backward passes are
// mathematically self-consistent, so it genuinely LEARNS the training text
// and can then hold a (small) conversation.
//
// Architecture: token + position embeddings -> 1 Transformer block
// (LayerNorm -> single-head causal self-attention -> residual ->
//  LayerNorm -> MLP(4x, GELU) -> residual) -> final LayerNorm -> linear head.

export interface TransformerConfig {
  nEmbd: number;     // embedding / channel dimension (C)
  blockSize: number; // max context length (T)
  mlpMult: number;   // MLP hidden expansion factor
  lr: number;        // Adam learning rate
}

const DEFAULT_CONFIG: TransformerConfig = {
  nEmbd: 64,
  blockSize: 64,
  mlpMult: 4,
  lr: 0.003,
};

// Box-Muller normal sample
function randn(): number {
  let u = 0;
  let v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

function filled(n: number, std: number): Float32Array {
  const a = new Float32Array(n);
  if (std > 0) {
    for (let i = 0; i < n; i++) a[i] = randn() * std;
  }
  return a;
}

// GELU (tanh approximation) and its derivative w.r.t. input.
const GELU_C = 0.7978845608028654; // sqrt(2/pi)
function geluFwd(x: number): number {
  const inner = GELU_C * (x + 0.044715 * x * x * x);
  return 0.5 * x * (1 + Math.tanh(inner));
}
function geluGrad(x: number): number {
  const x3 = x * x * x;
  const inner = GELU_C * (x + 0.044715 * x3);
  const t = Math.tanh(inner);
  const dInner = GELU_C * (1 + 3 * 0.044715 * x * x);
  return 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * dInner;
}

// One trainable parameter tensor with Adam state.
class Param {
  data: Float32Array;
  grad: Float32Array;
  m: Float32Array;
  v: Float32Array;
  constructor(size: number, std: number) {
    this.data = filled(size, std);
    this.grad = new Float32Array(size);
    this.m = new Float32Array(size);
    this.v = new Float32Array(size);
  }
  zeroGrad() {
    this.grad.fill(0);
  }
  step(lr: number, t: number) {
    const b1 = 0.9;
    const b2 = 0.999;
    const eps = 1e-8;
    const bc1 = 1 - Math.pow(b1, t);
    const bc2 = 1 - Math.pow(b2, t);
    const d = this.data;
    const g = this.grad;
    const m = this.m;
    const v = this.v;
    for (let i = 0; i < d.length; i++) {
      const gi = g[i];
      m[i] = b1 * m[i] + (1 - b1) * gi;
      v[i] = b2 * v[i] + (1 - b2) * gi * gi;
      const mhat = m[i] / bc1;
      const vhat = v[i] / bc2;
      d[i] -= (lr * mhat) / (Math.sqrt(vhat) + eps);
    }
  }
}

export class CharTransformer {
  cfg: TransformerConfig;
  // vocabulary
  stoi: Map<string, number> = new Map();
  itos: string[] = [];
  vocab = 0;

  // parameters
  wte!: Param; // [V, C]
  wpe!: Param; // [T, C]
  ln1g!: Param; ln1b!: Param;
  Wq!: Param; bq!: Param;
  Wk!: Param; bk!: Param;
  Wv!: Param; bv!: Param;
  Wo!: Param; bo!: Param;
  ln2g!: Param; ln2b!: Param;
  W1!: Param; b1!: Param;
  W2!: Param; b2!: Param;
  lnfg!: Param; lnfb!: Param;
  Whead!: Param; bhead!: Param;

  params: Param[] = [];
  tokens: number[] = [];
  stepCount = 0;

  constructor(cfg: Partial<TransformerConfig> = {}) {
    this.cfg = { ...DEFAULT_CONFIG, ...cfg };
  }

  // Build vocabulary + token stream from raw text, and (re)initialize weights.
  setText(text: string) {
    const chars = Array.from(new Set(text.split(''))).sort();
    this.itos = chars;
    this.stoi = new Map(chars.map((c, i) => [c, i]));
    this.vocab = chars.length;
    this.tokens = text.split('').map(c => this.stoi.get(c)!);
    this.initWeights();
  }

  encode(text: string): number[] {
    const out: number[] = [];
    for (const c of text) {
      const id = this.stoi.get(c);
      if (id !== undefined) out.push(id);
    }
    return out;
  }
  decode(ids: number[]): string {
    return ids.map(i => this.itos[i] ?? '').join('');
  }

  private initWeights() {
    const C = this.cfg.nEmbd;
    const T = this.cfg.blockSize;
    const V = this.vocab;
    const H = C * this.cfg.mlpMult;
    const s = 0.02;
    this.wte = new Param(V * C, s);
    this.wpe = new Param(T * C, s);
    this.ln1g = new Param(C, 0); this.ln1g.data.fill(1);
    this.ln1b = new Param(C, 0);
    this.Wq = new Param(C * C, s); this.bq = new Param(C, 0);
    this.Wk = new Param(C * C, s); this.bk = new Param(C, 0);
    this.Wv = new Param(C * C, s); this.bv = new Param(C, 0);
    this.Wo = new Param(C * C, s); this.bo = new Param(C, 0);
    this.ln2g = new Param(C, 0); this.ln2g.data.fill(1);
    this.ln2b = new Param(C, 0);
    this.W1 = new Param(C * H, s); this.b1 = new Param(H, 0);
    this.W2 = new Param(H * C, s); this.b2 = new Param(C, 0);
    this.lnfg = new Param(C, 0); this.lnfg.data.fill(1);
    this.lnfb = new Param(C, 0);
    this.Whead = new Param(C * V, s); this.bhead = new Param(V, 0);
    this.params = [
      this.wte, this.wpe, this.ln1g, this.ln1b,
      this.Wq, this.bq, this.Wk, this.bk, this.Wv, this.bv, this.Wo, this.bo,
      this.ln2g, this.ln2b, this.W1, this.b1, this.W2, this.b2,
      this.lnfg, this.lnfb, this.Whead, this.bhead,
    ];
    this.stepCount = 0;
  }

  // ---- low level helpers (row-major) ----
  // y[n,o] = x[n,k] @ W[k,o] + b[o]
  private linFwd(x: Float32Array, n: number, k: number, W: Float32Array, b: Float32Array | null, o: number): Float32Array {
    const y = new Float32Array(n * o);
    for (let i = 0; i < n; i++) {
      const xi = i * k;
      const yi = i * o;
      for (let j = 0; j < o; j++) {
        let sum = b ? b[j] : 0;
        for (let p = 0; p < k; p++) sum += x[xi + p] * W[p * o + j];
        y[yi + j] = sum;
      }
    }
    return y;
  }

  // LayerNorm forward over last dim C. Returns {y, xhat, rstd}
  private lnFwd(x: Float32Array, n: number, C: number, g: Float32Array, b: Float32Array) {
    const y = new Float32Array(n * C);
    const xhat = new Float32Array(n * C);
    const rstd = new Float32Array(n);
    const eps = 1e-5;
    for (let i = 0; i < n; i++) {
      const off = i * C;
      let mean = 0;
      for (let c = 0; c < C; c++) mean += x[off + c];
      mean /= C;
      let varSum = 0;
      for (let c = 0; c < C; c++) {
        const d = x[off + c] - mean;
        varSum += d * d;
      }
      const rs = 1 / Math.sqrt(varSum / C + eps);
      rstd[i] = rs;
      for (let c = 0; c < C; c++) {
        const xh = (x[off + c] - mean) * rs;
        xhat[off + c] = xh;
        y[off + c] = xh * g[c] + b[c];
      }
    }
    return { y, xhat, rstd };
  }

  // Forward pass over a token sequence. Stores activations for backward.
  // Returns logits [n, V] and the cache.
  private forward(seq: number[]) {
    const C = this.cfg.nEmbd;
    const V = this.vocab;
    const H = C * this.cfg.mlpMult;
    const n = seq.length;
    const scale = 1 / Math.sqrt(C);

    // embeddings
    const x0 = new Float32Array(n * C); // token+pos
    for (let t = 0; t < n; t++) {
      const tok = seq[t];
      const wteOff = tok * C;
      const wpeOff = t * C;
      const off = t * C;
      for (let c = 0; c < C; c++) x0[off + c] = this.wte.data[wteOff + c] + this.wpe.data[wpeOff + c];
    }

    // ln1
    const ln1 = this.lnFwd(x0, n, C, this.ln1g.data, this.ln1b.data);
    // q,k,v
    const q = this.linFwd(ln1.y, n, C, this.Wq.data, this.bq.data, C);
    const k = this.linFwd(ln1.y, n, C, this.Wk.data, this.bk.data, C);
    const v = this.linFwd(ln1.y, n, C, this.Wv.data, this.bv.data, C);

    // single-head causal attention
    const att = new Float32Array(n * n); // row-major lower triangular weights
    const attOut = new Float32Array(n * C);
    for (let i = 0; i < n; i++) {
      // scores
      let maxS = -Infinity;
      const sRow = new Float32Array(i + 1);
      for (let j = 0; j <= i; j++) {
        let dot = 0;
        const qi = i * C;
        const kj = j * C;
        for (let c = 0; c < C; c++) dot += q[qi + c] * k[kj + c];
        dot *= scale;
        sRow[j] = dot;
        if (dot > maxS) maxS = dot;
      }
      let sum = 0;
      for (let j = 0; j <= i; j++) {
        const e = Math.exp(sRow[j] - maxS);
        sRow[j] = e;
        sum += e;
      }
      for (let j = 0; j <= i; j++) {
        const a = sRow[j] / sum;
        att[i * n + j] = a;
        const vj = j * C;
        const oi = i * C;
        for (let c = 0; c < C; c++) attOut[oi + c] += a * v[vj + c];
      }
    }

    // output projection
    const proj = this.linFwd(attOut, n, C, this.Wo.data, this.bo.data, C);
    // residual 1
    const x1 = new Float32Array(n * C);
    for (let i = 0; i < n * C; i++) x1[i] = x0[i] + proj[i];

    // ln2
    const ln2 = this.lnFwd(x1, n, C, this.ln2g.data, this.ln2b.data);
    // mlp
    const fc = this.linFwd(ln2.y, n, C, this.W1.data, this.b1.data, H); // [n,H]
    const act = new Float32Array(n * H);
    for (let i = 0; i < n * H; i++) act[i] = geluFwd(fc[i]);
    const mlpOut = this.linFwd(act, n, H, this.W2.data, this.b2.data, C); // [n,C]
    // residual 2
    const x2 = new Float32Array(n * C);
    for (let i = 0; i < n * C; i++) x2[i] = x1[i] + mlpOut[i];

    // final ln
    const lnf = this.lnFwd(x2, n, C, this.lnfg.data, this.lnfb.data);
    // head
    const logits = this.linFwd(lnf.y, n, C, this.Whead.data, this.bhead.data, V);

    return { n, seq, x0, ln1, q, k, v, att, attOut, x1, ln2, fc, act, x2, lnf, logits };
  }

  // Backprop a generic linear: given dy[n,o], x[n,k], W[k,o], accumulate
  // dW, db and return dx[n,k].
  private linBwd(dy: Float32Array, x: Float32Array, n: number, k: number, o: number, W: Float32Array, dW: Float32Array, db: Float32Array | null): Float32Array {
    const dx = new Float32Array(n * k);
    for (let i = 0; i < n; i++) {
      const yi = i * o;
      const xi = i * k;
      for (let j = 0; j < o; j++) {
        const dyij = dy[yi + j];
        if (db) db[j] += dyij;
        for (let p = 0; p < k; p++) {
          dW[p * o + j] += x[xi + p] * dyij;
          dx[xi + p] += W[p * o + j] * dyij;
        }
      }
    }
    return dx;
  }

  // LayerNorm backward. dy[n,C], returns dx[n,C], accumulates dg, db.
  private lnBwd(dy: Float32Array, xhat: Float32Array, rstd: Float32Array, n: number, C: number, g: Float32Array, dg: Float32Array, db: Float32Array): Float32Array {
    const dx = new Float32Array(n * C);
    for (let i = 0; i < n; i++) {
      const off = i * C;
      let meanDxhat = 0;
      let meanDxhatXhat = 0;
      const dxhat = new Float32Array(C);
      for (let c = 0; c < C; c++) {
        const d = dy[off + c];
        dg[c] += d * xhat[off + c];
        db[c] += d;
        const dh = d * g[c];
        dxhat[c] = dh;
        meanDxhat += dh;
        meanDxhatXhat += dh * xhat[off + c];
      }
      meanDxhat /= C;
      meanDxhatXhat /= C;
      const rs = rstd[i];
      for (let c = 0; c < C; c++) {
        dx[off + c] = rs * (dxhat[c] - meanDxhat - xhat[off + c] * meanDxhatXhat);
      }
    }
    return dx;
  }

  // Compute loss + gradients for a single sequence; targets = seq shifted by 1.
  // Returns the summed cross-entropy loss and the number of predicted positions.
  private backward(cache: ReturnType<CharTransformer['forward']>, targets: number[]): { loss: number; count: number } {
    const C = this.cfg.nEmbd;
    const V = this.vocab;
    const H = C * this.cfg.mlpMult;
    const n = cache.n;
    const scale = 1 / Math.sqrt(C);

    // softmax + cross-entropy on logits -> dlogits
    const dlogits = new Float32Array(n * V);
    let loss = 0;
    let count = 0;
    for (let i = 0; i < n; i++) {
      const tgt = targets[i];
      if (tgt < 0) continue; // skip unpredicted positions
      const off = i * V;
      let maxL = -Infinity;
      for (let j = 0; j < V; j++) if (cache.logits[off + j] > maxL) maxL = cache.logits[off + j];
      let sum = 0;
      for (let j = 0; j < V; j++) sum += Math.exp(cache.logits[off + j] - maxL);
      const probTgt = Math.exp(cache.logits[off + tgt] - maxL) / sum;
      loss += -Math.log(Math.max(probTgt, 1e-12));
      count++;
      for (let j = 0; j < V; j++) {
        const p = Math.exp(cache.logits[off + j] - maxL) / sum;
        dlogits[off + j] = p;
      }
      dlogits[off + tgt] -= 1;
    }
    // normalize gradient by count (mean loss)
    if (count > 0) {
      const inv = 1 / count;
      for (let i = 0; i < dlogits.length; i++) dlogits[i] *= inv;
    }

    // head backward
    const dLnfY = this.linBwd(dlogits, cache.lnf.y, n, C, V, this.Whead.data, this.Whead.grad, this.bhead.grad);
    // final ln backward
    const dx2 = this.lnBwd(dLnfY, cache.lnf.xhat, cache.lnf.rstd, n, C, this.lnfg.data, this.lnfg.grad, this.lnfb.grad);

    // residual 2: x2 = x1 + mlpOut -> dx1 += dx2 ; dmlpOut = dx2
    const dMlpOut = dx2; // share; we'll also route to x1 at the end
    // mlp backward
    const dAct = this.linBwd(dMlpOut, cache.act, n, H, C, this.W2.data, this.W2.grad, this.b2.grad);
    const dFc = new Float32Array(n * H);
    for (let i = 0; i < n * H; i++) dFc[i] = dAct[i] * geluGrad(cache.fc[i]);
    const dLn2Y = this.linBwd(dFc, cache.ln2.y, n, C, H, this.W1.data, this.W1.grad, this.b1.grad);
    // ln2 backward -> dx1 contribution
    const dx1FromLn2 = this.lnBwd(dLn2Y, cache.ln2.xhat, cache.ln2.rstd, n, C, this.ln2g.data, this.ln2g.grad, this.ln2b.grad);

    // accumulate dx1 = dx2 (residual) + dx1FromLn2
    const dx1 = new Float32Array(n * C);
    for (let i = 0; i < n * C; i++) dx1[i] = dx2[i] + dx1FromLn2[i];

    // residual 1: x1 = x0 + proj -> dproj = dx1 ; dx0 += dx1
    const dProj = dx1;
    // output projection backward
    const dAttOut = this.linBwd(dProj, cache.attOut, n, C, C, this.Wo.data, this.Wo.grad, this.bo.grad);

    // attention backward
    const dq = new Float32Array(n * C);
    const dk = new Float32Array(n * C);
    const dv = new Float32Array(n * C);
    for (let i = 0; i < n; i++) {
      const oi = i * C;
      // da_ij and dv_j
      const da = new Float32Array(i + 1);
      for (let j = 0; j <= i; j++) {
        const a = cache.att[i * n + j];
        const vj = j * C;
        let dot = 0;
        for (let c = 0; c < C; c++) {
          dv[vj + c] += a * dAttOut[oi + c];
          dot += dAttOut[oi + c] * cache.v[vj + c];
        }
        da[j] = dot;
      }
      // softmax backward: ds_ij = a_ij*(da_ij - sum_j' a_ij'*da_ij')
      let sa = 0;
      for (let j = 0; j <= i; j++) sa += cache.att[i * n + j] * da[j];
      for (let j = 0; j <= i; j++) {
        const a = cache.att[i * n + j];
        const ds = a * (da[j] - sa) * scale;
        const qi = i * C;
        const kj = j * C;
        for (let c = 0; c < C; c++) {
          dq[qi + c] += ds * cache.k[kj + c];
          dk[kj + c] += ds * cache.q[qi + c];
        }
      }
    }

    // q,k,v linear backward -> all into dLn1Y
    const dLn1Y = new Float32Array(n * C);
    const addInto = (src: Float32Array) => { for (let i = 0; i < n * C; i++) dLn1Y[i] += src[i]; };
    addInto(this.linBwd(dq, cache.ln1.y, n, C, C, this.Wq.data, this.Wq.grad, this.bq.grad));
    addInto(this.linBwd(dk, cache.ln1.y, n, C, C, this.Wk.data, this.Wk.grad, this.bk.grad));
    addInto(this.linBwd(dv, cache.ln1.y, n, C, C, this.Wv.data, this.Wv.grad, this.bv.grad));

    // ln1 backward -> dx0 contribution
    const dx0FromLn1 = this.lnBwd(dLn1Y, cache.ln1.xhat, cache.ln1.rstd, n, C, this.ln1g.data, this.ln1g.grad, this.ln1b.grad);

    // dx0 = dx1 (residual) + dx0FromLn1
    const dx0 = new Float32Array(n * C);
    for (let i = 0; i < n * C; i++) dx0[i] = dx1[i] + dx0FromLn1[i];

    // embeddings backward
    for (let t = 0; t < n; t++) {
      const tok = cache.seq[t];
      const wteOff = tok * C;
      const wpeOff = t * C;
      const off = t * C;
      for (let c = 0; c < C; c++) {
        this.wte.grad[wteOff + c] += dx0[off + c];
        this.wpe.grad[wpeOff + c] += dx0[off + c];
      }
    }

    return { loss, count };
  }

  // Debug hook: run forward + backward on one fixed sequence (no optimizer step)
  // and return the loss plus the LM-head weight gradient. Used to verify the
  // GPU kernels produce the same math as this validated CPU reference.
  debugForwardBackward(inputs: number[], targets: number[]): { loss: number; logits: Float32Array; wheadGrad: Float32Array } {
    for (const p of this.params) p.zeroGrad();
    const cache = this.forward(inputs);
    const { loss, count } = this.backward(cache, targets);
    return {
      loss: count > 0 ? loss / count : 0,
      logits: cache.logits.slice(),
      wheadGrad: this.Whead.grad.slice(),
    };
  }

  // One optimization step over a mini-batch of random windows. Returns avg loss.
  trainStep(batch = 8): number {
    const T = this.cfg.blockSize;
    const N = this.tokens.length;
    if (N < 2) return 0;
    for (const p of this.params) p.zeroGrad();

    let totalLoss = 0;
    let totalCount = 0;
    let valid = 0;
    for (let b = 0; b < batch; b++) {
      const maxStart = Math.max(1, N - 1);
      const start = Math.floor(Math.random() * maxStart);
      const end = Math.min(start + T + 1, N);
      const window = this.tokens.slice(start, end);
      if (window.length < 2) continue;
      const inputs = window.slice(0, window.length - 1);
      const targets = window.slice(1);
      const cache = this.forward(inputs);
      const { loss, count } = this.backward(cache, targets);
      totalLoss += loss;
      totalCount += count;
      valid++;
    }
    if (valid === 0) return 0;

    // average gradients across the batch sequences
    const invBatch = 1 / valid;
    for (const p of this.params) {
      for (let i = 0; i < p.grad.length; i++) p.grad[i] *= invBatch;
    }
    // global grad clip to norm 1.0
    let sq = 0;
    for (const p of this.params) for (let i = 0; i < p.grad.length; i++) sq += p.grad[i] * p.grad[i];
    const norm = Math.sqrt(sq);
    if (norm > 1.0) {
      const sc = 1.0 / norm;
      for (const p of this.params) for (let i = 0; i < p.grad.length; i++) p.grad[i] *= sc;
    }

    this.stepCount++;
    for (const p of this.params) p.step(this.cfg.lr, this.stepCount);

    return totalCount > 0 ? totalLoss / totalCount : 0;
  }

  // Public CPU forward returning the logits [n*V] for a context (used by the
  // app's unified sampler so CPU and GPU share the same sampling code path).
  logitsForContextCPU(ctx: number[]): Float32Array {
    const cache = this.forward(ctx);
    return cache.logits.slice(0, ctx.length * this.vocab);
  }

  // Greedy / temperature sampling continuation. Generates up to maxNew tokens,
  // stopping when `stop` substring appears at the end of the generated text.
  generate(prompt: string, maxNew = 160, temperature = 0.8, topK = 12, stop = '\n'): string {
    const T = this.cfg.blockSize;
    let ids = this.encode(prompt);
    let out = '';
    for (let step = 0; step < maxNew; step++) {
      const ctx = ids.slice(Math.max(0, ids.length - T));
      const cache = this.forward(ctx);
      const V = this.vocab;
      const last = (ctx.length - 1) * V;
      // gather logits of last position
      const logits = new Float32Array(V);
      for (let j = 0; j < V; j++) logits[j] = cache.logits[last + j] / Math.max(1e-6, temperature);
      // softmax
      let maxL = -Infinity;
      for (let j = 0; j < V; j++) if (logits[j] > maxL) maxL = logits[j];
      let sum = 0;
      const probs = new Float32Array(V);
      for (let j = 0; j < V; j++) { probs[j] = Math.exp(logits[j] - maxL); sum += probs[j]; }
      for (let j = 0; j < V; j++) probs[j] /= sum;
      // top-k
      const idx = Array.from({ length: V }, (_, j) => j).sort((a, b) => probs[b] - probs[a]).slice(0, Math.min(topK, V));
      let pSum = 0;
      for (const j of idx) pSum += probs[j];
      let r = Math.random() * pSum;
      let chosen = idx[0];
      for (const j of idx) { r -= probs[j]; if (r <= 0) { chosen = j; break; } }

      const ch = this.itos[chosen];
      out += ch;
      ids.push(chosen);
      if (stop && out.endsWith(stop)) break;
    }
    return out;
  }

  // Serialize learned weights so training can persist across calls.
  exportState() {
    return {
      cfg: this.cfg,
      itos: this.itos,
      stepCount: this.stepCount,
      params: this.params.map(p => Array.from(p.data)),
    };
  }
}
