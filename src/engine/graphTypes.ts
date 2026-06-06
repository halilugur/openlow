// Graph Type Definitions for Visual LLM Builder

export type NodeType =
  | 'text_input'
  | 'tokenizer'
  | 'wte'
  | 'wpe'
  | 'add'
  | 'layernorm'
  | 'attention'
  | 'mlp'
  | 'lm_head'
  | 'softmax'
  | 'sampler';

export type PortType = 'text' | 'tokens' | 'tensor';

export interface Port {
  id: string; // "nodeId-direction-name"
  name: string;
  direction: 'in' | 'out';
  type: PortType;
  shapeDescription?: string;
}

export interface CanvasNode {
  id: string;
  type: NodeType;
  label: string;
  x: number;
  y: number;
  inputs: Port[];
  outputs: Port[];
  params: Record<string, any>;
  status?: 'idle' | 'running' | 'success' | 'error';
  errorMessage?: string;
  outputShape?: number[];
  outputPreview?: string;
}

export interface CanvasEdge {
  id: string;
  fromNodeId: string;
  fromPortId: string;
  toNodeId: string;
  toPortId: string;
}

// Generate unique ID helper
export const generateId = () => Math.random().toString(36).substring(2, 9);

// Create default inputs/outputs for node types
export const createNodePorts = (type: NodeType, nodeId: string) => {
  const inputs: Port[] = [];
  const outputs: Port[] = [];

  switch (type) {
    case 'text_input':
      outputs.push({
        id: `${nodeId}-out-text`,
        name: 'text',
        direction: 'out',
        type: 'text',
        shapeDescription: 'Raw text string',
      });
      break;

    case 'tokenizer':
      inputs.push({
        id: `${nodeId}-in-text`,
        name: 'text',
        direction: 'in',
        type: 'text',
        shapeDescription: 'Input text string',
      });
      outputs.push({
        id: `${nodeId}-out-tokens`,
        name: 'tokens',
        direction: 'out',
        type: 'tokens',
        shapeDescription: '1D Token integer array [B, T]',
      });
      break;

    case 'wte':
      inputs.push({
        id: `${nodeId}-in-tokens`,
        name: 'tokens',
        direction: 'in',
        type: 'tokens',
        shapeDescription: 'Input tokens [B, T]',
      });
      outputs.push({
        id: `${nodeId}-out-tensor`,
        name: 'embeddings',
        direction: 'out',
        type: 'tensor',
        shapeDescription: 'Token embedding tensor [B, T, C]',
      });
      break;

    case 'wpe':
      inputs.push({
        id: `${nodeId}-in-tokens`,
        name: 'tokens',
        direction: 'in',
        type: 'tokens',
        shapeDescription: 'Input tokens [B, T]',
      });
      outputs.push({
        id: `${nodeId}-out-tensor`,
        name: 'positions',
        direction: 'out',
        type: 'tensor',
        shapeDescription: 'Position embedding tensor [B, T, C]',
      });
      break;

    case 'add':
      inputs.push(
        {
          id: `${nodeId}-in-a`,
          name: 'a',
          direction: 'in',
          type: 'tensor',
          shapeDescription: 'Input tensor A',
        },
        {
          id: `${nodeId}-in-b`,
          name: 'b',
          direction: 'in',
          type: 'tensor',
          shapeDescription: 'Input tensor B',
        }
      );
      outputs.push({
        id: `${nodeId}-out-tensor`,
        name: 'sum',
        direction: 'out',
        type: 'tensor',
        shapeDescription: 'Summed tensor A + B',
      });
      break;

    case 'layernorm':
      inputs.push({
        id: `${nodeId}-in-x`,
        name: 'x',
        direction: 'in',
        type: 'tensor',
        shapeDescription: 'Input tensor',
      });
      outputs.push({
        id: `${nodeId}-out-tensor`,
        name: 'normalized',
        direction: 'out',
        type: 'tensor',
        shapeDescription: 'Layer-normalized tensor',
      });
      break;

    case 'attention':
      inputs.push(
        {
          id: `${nodeId}-in-q`,
          name: 'q',
          direction: 'in',
          type: 'tensor',
          shapeDescription: 'Query tensor [B, T, C]',
        },
        {
          id: `${nodeId}-in-k`,
          name: 'k',
          direction: 'in',
          type: 'tensor',
          shapeDescription: 'Key tensor [B, T, C]',
        },
        {
          id: `${nodeId}-in-v`,
          name: 'v',
          direction: 'in',
          type: 'tensor',
          shapeDescription: 'Value tensor [B, T, C]',
        }
      );
      outputs.push({
        id: `${nodeId}-out-tensor`,
        name: 'attention',
        direction: 'out',
        type: 'tensor',
        shapeDescription: 'Self-attention output [B, T, C]',
      });
      break;

    case 'mlp':
      inputs.push({
        id: `${nodeId}-in-x`,
        name: 'x',
        direction: 'in',
        type: 'tensor',
        shapeDescription: 'Input tensor [B, T, C]',
      });
      outputs.push({
        id: `${nodeId}-out-tensor`,
        name: 'mlp_out',
        direction: 'out',
        type: 'tensor',
        shapeDescription: 'MLP Output [B, T, C]',
      });
      break;

    case 'lm_head':
      inputs.push({
        id: `${nodeId}-in-x`,
        name: 'x',
        direction: 'in',
        type: 'tensor',
        shapeDescription: 'Input tensor [B, T, C]',
      });
      outputs.push({
        id: `${nodeId}-out-tensor`,
        name: 'logits',
        direction: 'out',
        type: 'tensor',
        shapeDescription: 'Logits tensor [B, T, vocab_size]',
      });
      break;

    case 'softmax':
      inputs.push({
        id: `${nodeId}-in-logits`,
        name: 'logits',
        direction: 'in',
        type: 'tensor',
        shapeDescription: 'Input logits [B, T, vocab_size] or [B, vocab_size]',
      });
      outputs.push({
        id: `${nodeId}-out-tensor`,
        name: 'probs',
        direction: 'out',
        type: 'tensor',
        shapeDescription: 'Softmax probabilities',
      });
      break;

    case 'sampler':
      inputs.push({
        id: `${nodeId}-in-probs`,
        name: 'probs',
        direction: 'in',
        type: 'tensor',
        shapeDescription: 'Input probabilities [B, vocab_size] or [B, T, vocab_size]',
      });
      outputs.push({
        id: `${nodeId}-out-tokens`,
        name: 'next_token',
        direction: 'out',
        type: 'tokens',
        shapeDescription: 'Sampled token ID [B, 1]',
      });
      break;
  }

  return { inputs, outputs };
};

// Create a template node
export const createDefaultNode = (type: NodeType, x = 100, y = 100): CanvasNode => {
  const id = generateId();
  const { inputs, outputs } = createNodePorts(type, id);

  const labels: Record<NodeType, string> = {
    text_input: 'Text Input',
    tokenizer: 'Tokenizer (BPE)',
    wte: 'Token Embedding (WTE)',
    wpe: 'Position Embedding (WPE)',
    add: 'Add / Residual',
    layernorm: 'Layer Normalization',
    attention: 'Causal Self-Attention',
    mlp: 'Feed Forward (MLP)',
    lm_head: 'LM Head (Linear)',
    softmax: 'Softmax Classifier',
    sampler: 'Multinomial Sampler',
  };

  const defaultParams: Record<NodeType, Record<string, any>> = {
    text_input: { text: "Hello, I'm a language model," },
    tokenizer: {},
    wte: { vocab_size: 256, n_embd: 64 },
    wpe: { block_size: 256, n_embd: 64 },
    add: {},
    layernorm: { epsilon: 1e-5 },
    attention: { n_head: 4, head_size: 16 }, // C = n_head * head_size = 64
    mlp: { multiplier: 4 },
    lm_head: {},
    softmax: {},
    sampler: { temperature: 0.6, top_k: 10 },
  };

  return {
    id,
    type,
    label: labels[type],
    x,
    y,
    inputs,
    outputs,
    params: defaultParams[type],
    status: 'idle',
  };
};
