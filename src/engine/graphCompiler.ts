// Graph -> Model compiler.
//
// Turns the visual node graph the user wires on the Canvas into the concrete
// architecture configuration that the real CharTransformer trains. This is what
// makes the canvas the source of truth for the model: the number of
// Attention/MLP blocks, the embedding dimension, the context window and the MLP
// expansion are all READ FROM THE GRAPH instead of a fixed settings panel.
//
// It also validates the wiring against a valid GPT topology and, when the graph
// is incomplete or mis-wired, returns precise human-readable errors so training
// can be blocked with an explanation instead of silently "fixing" the graph.

import type { CanvasNode, CanvasEdge } from './graphTypes';

export interface GraphModelConfig {
  nEmbd: number; // embedding / channel dimension (C)
  blockSize: number; // context window length (T)
  mlpMult: number; // MLP hidden expansion factor
  nLayer: number; // number of stacked Transformer blocks
  vocabHint: number; // requested vocab size from the WTE node (informational)
}

export interface CompileResult {
  config: GraphModelConfig | null; // null when the graph is invalid
  errors: string[]; // blocking problems (training must not proceed)
  warnings: string[]; // non-blocking notes
}

const toInt = (v: unknown, fallback: number): number => {
  const n = typeof v === 'number' ? v : parseInt(String(v), 10);
  return Number.isFinite(n) ? n : fallback;
};

export function compileGraph(nodes: CanvasNode[], edges: CanvasEdge[]): CompileResult {
  const errors: string[] = [];
  const warnings: string[] = [];

  const byType = (t: CanvasNode['type']) => nodes.filter(n => n.type === t);
  // Does any edge feed `toNodeId` from a node of type `fromType`?
  const isFedByType = (toNodeId: string, fromType: CanvasNode['type']): boolean =>
    edges.some(e => {
      if (e.toNodeId !== toNodeId) return false;
      const from = nodes.find(n => n.id === e.fromNodeId);
      return from?.type === fromType;
    });

  const textInputs = byType('text_input');
  const tokenizers = byType('tokenizer');
  const wtes = byType('wte');
  const wpes = byType('wpe');
  const adds = byType('add');
  const attentions = byType('attention');
  const mlps = byType('mlp');
  const heads = byType('lm_head');

  // ---- Topology validation (input → embeddings → blocks → head) ----
  if (textInputs.length === 0) {
    errors.push('Add a Text Input node — it provides the training corpus.');
  }
  if (tokenizers.length === 0) {
    errors.push('Add a Tokenizer node connected to the Text Input.');
  } else if (textInputs.length > 0 && !tokenizers.some(t => isFedByType(t.id, 'text_input'))) {
    errors.push('Connect the Text Input node to the Tokenizer.');
  }

  if (wtes.length === 0) {
    errors.push('Add a Token Embedding (WTE) node.');
  } else if (tokenizers.length > 0 && !wtes.some(w => isFedByType(w.id, 'tokenizer'))) {
    errors.push('Connect the Tokenizer to the Token Embedding (WTE) node.');
  }
  if (wpes.length === 0) {
    errors.push('Add a Position Embedding (WPE) node.');
  } else if (tokenizers.length > 0 && !wpes.some(w => isFedByType(w.id, 'tokenizer'))) {
    errors.push('Connect the Tokenizer to the Position Embedding (WPE) node.');
  }

  // WTE + WPE must be combined by an Add node.
  if (wtes.length > 0 && wpes.length > 0) {
    const hasEmbedAdd = adds.some(a => isFedByType(a.id, 'wte') && isFedByType(a.id, 'wpe'));
    if (!hasEmbedAdd) {
      errors.push('Add a "+" (Add/Residual) node that sums the WTE and WPE outputs.');
    }
  }

  // A Transformer block = one Attention node + one MLP node. The number of
  // Attention nodes you wire is the number of stacked blocks (nLayer).
  const nLayer = attentions.length;
  if (nLayer === 0) {
    errors.push('Add at least one Causal Self-Attention node — it is the core of a Transformer block.');
  }
  if (mlps.length === 0) {
    errors.push('Add a Feed Forward (MLP) node to complete the Transformer block.');
  }
  if (nLayer > 0 && mlps.length !== nLayer) {
    errors.push(
      `Block mismatch: found ${nLayer} Attention node(s) but ${mlps.length} MLP node(s). ` +
        'Each Transformer block needs exactly one Attention and one MLP.'
    );
  }

  if (heads.length === 0) {
    errors.push('Add an LM Head node — it projects the final activations to vocabulary logits.');
  }

  // ---- Derive the embedding dimension and check it is consistent. ----
  const embedDims = new Set<number>();
  for (const w of wtes) embedDims.add(toInt(w.params.n_embd, 64));
  for (const w of wpes) embedDims.add(toInt(w.params.n_embd, 64));
  for (const a of attentions) {
    const nHead = toInt(a.params.n_head, 4);
    const headSize = toInt(a.params.head_size, 16);
    embedDims.add(nHead * headSize);
  }
  if (embedDims.size > 1) {
    errors.push(
      `Embedding dimension mismatch across nodes: ${Array.from(embedDims).sort((x, y) => x - y).join(', ')}. ` +
        'Make WTE/WPE n_embd and Attention (n_head × head_size) all equal.'
    );
  }
  const nEmbd = embedDims.size === 1 ? Array.from(embedDims)[0] : 64;

  // Context window comes from the WPE node (it owns the position table).
  const blockSize = wpes.length > 0 ? toInt(wpes[0].params.block_size, 64) : 64;

  // MLP expansion factor — all MLP nodes must agree.
  const mlpMults = new Set<number>();
  for (const m of mlps) mlpMults.add(Math.max(1, toInt(m.params.multiplier, 4)));
  if (mlpMults.size > 1) {
    errors.push(
      `MLP multiplier mismatch: ${Array.from(mlpMults).join(', ')}. Use the same expansion factor for every MLP block.`
    );
  }
  const mlpMult = mlpMults.size >= 1 ? Array.from(mlpMults)[0] : 4;

  const vocabHint = wtes.length > 0 ? toInt(wtes[0].params.vocab_size, 256) : 256;

  // Non-blocking guidance.
  if (nEmbd <= 0 || nEmbd > 512) {
    errors.push(`Embedding dimension ${nEmbd} is out of the supported range (1–512).`);
  }
  if (blockSize <= 1) {
    errors.push(`Context window (block_size = ${blockSize}) must be at least 2.`);
  }
  if (nLayer > 1) {
    warnings.push(`${nLayer}-block architecture detected — WebGPU runs single-block only, so multi-block trains on CPU.`);
  }

  if (errors.length > 0) {
    return { config: null, errors, warnings };
  }

  return {
    config: { nEmbd, blockSize, mlpMult, nLayer, vocabHint },
    errors,
    warnings,
  };
}
