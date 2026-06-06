import React from 'react';
import type { CanvasNode } from '../../engine/graphTypes';
import {
  SlidersIcon,
  PulseIcon,
  InfoIcon,
} from '@phosphor-icons/react';

interface NodeInspectorProps {
  node: CanvasNode | null;
  onUpdateNodeParams: (nodeId: string, params: Record<string, any>) => void;
}

export const NodeInspector: React.FC<NodeInspectorProps> = ({ node, onUpdateNodeParams }) => {
  if (!node) {
    return (
      <aside className="w-[320px] bg-zinc-900 border-l border-zinc-800 flex flex-col h-full items-center justify-center text-zinc-500 p-6 select-none">
        <InfoIcon size={32} className="opacity-30 mb-2.5 text-zinc-400" />
        <span className="text-[12px] font-medium text-center leading-relaxed">
          Select a node on the canvas to configure parameters and inspect tensor outputs.
        </span>
      </aside>
    );
  }

  const handleParamChange = (key: string, val: any) => {
    onUpdateNodeParams(node.id, { ...node.params, [key]: val });
  };

  const renderParamsEditor = () => {
    switch (node.type) {
      case 'text_input':
        return (
          <div className="flex flex-col gap-2">
            <label className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider">
              Prompt Text
            </label>
            <textarea
              value={node.params.text || ''}
              onChange={(e) => handleParamChange('text', e.target.value)}
              className="w-full h-[80px] px-3 py-2 bg-zinc-950 border border-zinc-850 hover:border-zinc-750 focus:border-[var(--accent)] text-[12px] text-zinc-100 rounded-lg outline-none resize-none font-sans"
              placeholder="Type your prompt..."
            />
          </div>
        );

      case 'wte':
        return (
          <div className="flex flex-col gap-3">
            <div className="flex flex-col gap-1.5">
              <label className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider">
                Vocabulary Size
              </label>
              <input
                type="number"
                value={node.params.vocab_size || 256}
                onChange={(e) => handleParamChange('vocab_size', parseInt(e.target.value) || 256)}
                className="w-full px-3 py-2 bg-zinc-950 border border-zinc-850 focus:border-[var(--accent)] text-[12px] text-zinc-100 rounded-lg outline-none font-mono"
              />
            </div>
            <div className="flex flex-col gap-1.5">
              <label className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider">
                Embedding Size (C)
              </label>
              <input
                type="number"
                value={node.params.n_embd || 64}
                onChange={(e) => handleParamChange('n_embd', parseInt(e.target.value) || 64)}
                className="w-full px-3 py-2 bg-zinc-950 border border-zinc-850 focus:border-[var(--accent)] text-[12px] text-zinc-100 rounded-lg outline-none font-mono"
              />
            </div>
          </div>
        );

      case 'wpe':
        return (
          <div className="flex flex-col gap-3">
            <div className="flex flex-col gap-1.5">
              <label className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider">
                Context Window (Max Sequence Length)
              </label>
              <input
                type="number"
                value={node.params.block_size || 256}
                onChange={(e) => handleParamChange('block_size', parseInt(e.target.value) || 256)}
                className="w-full px-3 py-2 bg-zinc-950 border border-zinc-850 focus:border-[var(--accent)] text-[12px] text-zinc-100 rounded-lg outline-none font-mono"
              />
            </div>
            <div className="flex flex-col gap-1.5">
              <label className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider">
                Embedding Size (C)
              </label>
              <input
                type="number"
                value={node.params.n_embd || 64}
                onChange={(e) => handleParamChange('n_embd', parseInt(e.target.value) || 64)}
                className="w-full px-3 py-2 bg-zinc-950 border border-zinc-850 focus:border-[var(--accent)] text-[12px] text-zinc-100 rounded-lg outline-none font-mono"
              />
            </div>
          </div>
        );

      case 'layernorm':
        return (
          <div className="flex flex-col gap-1.5">
            <label className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider">
              Epsilon (&epsilon;)
            </label>
            <input
              type="number"
              step="0.00001"
              value={node.params.epsilon || 1e-5}
              onChange={(e) => handleParamChange('epsilon', parseFloat(e.target.value) || 1e-5)}
              className="w-full px-3 py-2 bg-zinc-950 border border-zinc-850 focus:border-[var(--accent)] text-[12px] text-zinc-100 rounded-lg outline-none font-mono"
            />
          </div>
        );

      case 'attention':
        return (
          <div className="flex flex-col gap-3">
            <div className="flex flex-col gap-1.5">
              <label className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider">
                Number of Heads
              </label>
              <input
                type="number"
                value={node.params.n_head || 4}
                onChange={(e) => handleParamChange('n_head', parseInt(e.target.value) || 4)}
                className="w-full px-3 py-2 bg-zinc-950 border border-zinc-850 focus:border-[var(--accent)] text-[12px] text-zinc-100 rounded-lg outline-none font-mono"
              />
            </div>
            <div className="flex flex-col gap-1.5">
              <label className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider">
                Head Size
              </label>
              <input
                type="number"
                value={node.params.head_size || 16}
                onChange={(e) => handleParamChange('head_size', parseInt(e.target.value) || 16)}
                className="w-full px-3 py-2 bg-zinc-950 border border-zinc-850 focus:border-[var(--accent)] text-[12px] text-zinc-100 rounded-lg outline-none font-mono"
              />
            </div>
            <div className="text-[10px] text-zinc-500 font-sans italic bg-zinc-950/40 p-2 rounded border border-zinc-850">
              Total channels (C) = n_head &times; head_size = {(node.params.n_head || 4) * (node.params.head_size || 16)}
            </div>
          </div>
        );

      case 'sampler':
        return (
          <div className="flex flex-col gap-3">
            <div className="flex flex-col gap-1.5">
              <label className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider">
                Temperature
              </label>
              <input
                type="number"
                step="0.1"
                min="0.1"
                max="2.0"
                value={node.params.temperature || 1.0}
                onChange={(e) => handleParamChange('temperature', parseFloat(e.target.value) || 1.0)}
                className="w-full px-3 py-2 bg-zinc-950 border border-zinc-850 focus:border-[var(--accent)] text-[12px] text-zinc-100 rounded-lg outline-none font-mono"
              />
            </div>
            <div className="flex flex-col gap-1.5">
              <label className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider">
                Top K
              </label>
              <input
                type="number"
                value={node.params.top_k || 50}
                onChange={(e) => handleParamChange('top_k', parseInt(e.target.value) || 50)}
                className="w-full px-3 py-2 bg-zinc-950 border border-zinc-850 focus:border-[var(--accent)] text-[12px] text-zinc-100 rounded-lg outline-none font-mono"
              />
            </div>
          </div>
        );

      default:
        return (
          <div className="text-[11px] text-zinc-500 italic">
            This module has no adjustable parameters.
          </div>
        );
    }
  };

  const getPortBadgeColor = (type: string) => {
    switch (type) {
      case 'text': return 'bg-[var(--accent)]/10 text-[var(--accent)] border-[var(--accent)]/20';
      case 'tokens': return 'bg-sky-400/10 text-sky-400 border-sky-400/20';
      case 'tensor': return 'bg-amber-500/10 text-amber-500 border-amber-500/20';
      default: return 'bg-zinc-800 text-zinc-400';
    }
  };

  return (
    <aside className="w-[320px] bg-zinc-900 border-l border-zinc-800 flex flex-col h-full overflow-y-auto select-none select-none">
      {/* Title */}
      <div className="p-4 border-b border-zinc-800 flex items-center gap-2">
        <SlidersIcon size={18} className="text-zinc-400" />
        <h2 className="text-sm font-semibold tracking-wide text-zinc-100">
          Inspector
        </h2>
      </div>

      {/* Info Group */}
      <div className="p-4 border-b border-zinc-800 bg-zinc-950/20">
        <div className="text-[11px] text-zinc-500 font-mono">NODE ID: {node.id}</div>
        <div className="text-[15px] font-bold text-zinc-100 mt-1">{node.label}</div>
        <div className="text-[11px] text-zinc-400 mt-1.5 leading-relaxed italic">
          {node.type === 'text_input' && 'Represents the raw prompt text string.'}
          {node.type === 'tokenizer' && 'Encodes a string into UTF-8 raw byte index tokens.'}
          {node.type === 'wte' && 'Looks up token vectors inside the token weight embedding matrix.'}
          {node.type === 'wpe' && 'Generates sinusoidal/learned position encodings for sequence vectors.'}
          {node.type === 'add' && 'Computes element-wise tensor addition (useful for residual sums).'}
          {node.type === 'layernorm' && 'Normalizes row vectors over channels for numerical training stability.'}
          {node.type === 'attention' && 'Executes scaled dot-product self-attention with triangular causal masking.'}
          {node.type === 'mlp' && 'Computes fully-connected linear projections with intermediate GeLU activation.'}
          {node.type === 'lm_head' && 'Linear weight projections mapping sequence back to BPE vocabulary logits.'}
          {node.type === 'softmax' && 'Applies the softmax function over logits to obtain token probability distributions.'}
          {node.type === 'sampler' && 'Selects the next token ID from probabilities using multinomial sampling.'}
        </div>
      </div>

      {/* Ports Interface */}
      <div className="p-4 border-b border-zinc-800 flex flex-col gap-3">
        <h3 className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider">
          Node Interfaces
        </h3>
        
        {/* Inputs */}
        <div className="flex flex-col gap-1.5">
          <div className="text-[10px] text-zinc-400 font-medium">Inputs:</div>
          {node.inputs.length === 0 ? (
            <div className="text-[10px] text-zinc-500 italic pl-1">None</div>
          ) : (
            node.inputs.map(input => (
              <div key={input.id} className="flex items-center justify-between p-1.5 rounded border border-zinc-850 bg-zinc-950/40">
                <span className="text-[11px] font-mono text-zinc-300 pl-1">{input.name}</span>
                <span className={`text-[9px] px-1.5 py-0.5 rounded font-mono border ${getPortBadgeColor(input.type)}`}>
                  {input.type}
                </span>
              </div>
            ))
          )}
        </div>

        {/* Outputs */}
        <div className="flex flex-col gap-1.5">
          <div className="text-[10px] text-zinc-400 font-medium">Outputs:</div>
          {node.outputs.length === 0 ? (
            <div className="text-[10px] text-zinc-500 italic pl-1">None</div>
          ) : (
            node.outputs.map(output => (
              <div key={output.id} className="flex items-center justify-between p-1.5 rounded border border-zinc-850 bg-zinc-950/40">
                <span className="text-[11px] font-mono text-zinc-300 pl-1">{output.name}</span>
                <span className={`text-[9px] px-1.5 py-0.5 rounded font-mono border ${getPortBadgeColor(output.type)}`}>
                  {output.type}
                </span>
              </div>
            ))
          )}
        </div>
      </div>

      {/* Adjust Parameters */}
      <div className="p-4 border-b border-zinc-800 flex flex-col gap-3">
        <h3 className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider">
          Parameters
        </h3>
        {renderParamsEditor()}
      </div>

      {/* GPU Activations Monitor */}
      <div className="p-4 flex-1 flex flex-col gap-3">
        <div className="flex items-center gap-1.5">
          <PulseIcon size={15} className="text-zinc-500" />
          <h3 className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider">
            GPU Activations Monitor
          </h3>
        </div>
        
        {node.outputShape ? (
          <div className="flex flex-col gap-2.5">
            <div>
              <div className="text-[10px] text-zinc-500 font-medium">TENSOR SHAPE:</div>
              <div className="text-xs font-mono text-zinc-300 mt-0.5 bg-zinc-950/60 p-2 rounded border border-zinc-850">
                [{node.outputShape.join(', ')}]
              </div>
            </div>

            <div>
              <div className="text-[10px] text-zinc-500 font-medium">ACTIVATION VALUES PREVIEW:</div>
              <div className="text-[11px] font-mono text-zinc-400 mt-1 bg-zinc-950/60 p-2.5 rounded border border-zinc-850 overflow-x-auto whitespace-pre leading-relaxed h-[150px] overflow-y-auto">
                {node.outputPreview || 'Waiting for WebGPU compilation run...'}
              </div>
            </div>
          </div>
        ) : (
          <div className="text-[11px] text-zinc-500 italic bg-zinc-950/30 p-3 rounded border border-zinc-850/50 text-center">
            Run the compilation pipeline to view GPU tensor activations.
          </div>
        )}
      </div>
    </aside>
  );
};
