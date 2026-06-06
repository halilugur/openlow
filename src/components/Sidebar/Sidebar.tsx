import React from 'react';
import type { NodeType } from '../../engine/graphTypes';
import {
  FileTextIcon,
  BracketsCurlyIcon,
  StackIcon,
  CompassIcon,
  PlusIcon,
  CompassToolIcon,
  BrainIcon,
  HashIcon,
  SlidersIcon,
  ChartPieIcon,
  LightningIcon,
} from '@phosphor-icons/react';

interface SidebarProps {
  onAddNodeClick: (type: NodeType) => void;
}

interface ToolItem {
  type: NodeType;
  label: string;
  description: string;
  icon: React.ReactNode;
}

export const Sidebar: React.FC<SidebarProps> = ({ onAddNodeClick }) => {
  const categories: { title: string; items: ToolItem[] }[] = [
    {
      title: 'Inputs & Data',
      items: [
        {
          type: 'text_input',
          label: 'Text Input',
          description: 'Input prompt text',
          icon: <FileTextIcon size={16} className="text-emerald-500" />,
        },
        {
          type: 'tokenizer',
          label: 'Tokenizer',
          description: 'Encodes text to byte tokens',
          icon: <BracketsCurlyIcon size={16} className="text-emerald-500" />,
        },
      ],
    },
    {
      title: 'Embedding Layers',
      items: [
        {
          type: 'wte',
          label: 'Token Embed (WTE)',
          description: 'Maps tokens to vectors',
          icon: <StackIcon size={16} className="text-sky-400" />,
        },
        {
          type: 'wpe',
          label: 'Pos Embed (WPE)',
          description: 'Adds positional vectors',
          icon: <CompassIcon size={16} className="text-sky-400" />,
        },
      ],
    },
    {
      title: 'Transformer Math',
      items: [
        {
          type: 'add',
          label: 'Add / Residual',
          description: 'Element-wise tensor addition',
          icon: <PlusIcon size={16} className="text-amber-500" />,
        },
        {
          type: 'layernorm',
          label: 'Layer Normalization',
          description: 'Normalizes features',
          icon: <SlidersIcon size={16} className="text-amber-500" />,
        },
        {
          type: 'attention',
          label: 'Self-Attention',
          description: 'Causal multi-head attention',
          icon: <BrainIcon size={16} className="text-amber-500" />,
        },
        {
          type: 'mlp',
          label: 'Feed Forward (MLP)',
          description: 'GELU-activated projections',
          icon: <LightningIcon size={16} className="text-amber-500" />,
        },
      ],
    },
    {
      title: 'Outputs & Samplers',
      items: [
        {
          type: 'lm_head',
          label: 'LM Head (Linear)',
          description: 'Linear layer mapping to vocab',
          icon: <HashIcon size={16} className="text-rose-400" />,
        },
        {
          type: 'softmax',
          label: 'Softmax',
          description: 'Computes class probabilities',
          icon: <ChartPieIcon size={16} className="text-rose-400" />,
        },
        {
          type: 'sampler',
          label: 'Sampler',
          description: 'Top-k / multinomial sampler',
          icon: <CompassToolIcon size={16} className="text-rose-400" />,
        },
      ],
    },
  ];

  const handleDragStart = (e: React.DragEvent, type: NodeType) => {
    e.dataTransfer.setData('nodeType', type);
    e.dataTransfer.effectAllowed = 'copy';
  };

  return (
    <aside className="w-[280px] bg-zinc-900 border-r border-zinc-800 flex flex-col h-full overflow-y-auto select-none select-none">
      {/* Title */}
      <div className="p-4 border-b border-zinc-800 flex items-center justify-between">
        <h2 className="text-sm font-semibold tracking-wide text-zinc-100 flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
          LLM BLOCKS
        </h2>
        <span className="text-[10px] text-zinc-500 font-mono">DRAG & DROP</span>
      </div>

      {/* Tool Categories */}
      <div className="flex-1 p-3 flex flex-col gap-5">
        {categories.map((cat, i) => (
          <div key={i} className="flex flex-col gap-1.5">
            <h3 className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider pl-1 mb-1">
              {cat.title}
            </h3>
            <div className="flex flex-col gap-1">
              {cat.items.map((item, idx) => (
                <div
                  key={idx}
                  draggable
                  onDragStart={(e) => handleDragStart(e, item.type)}
                  onClick={() => onAddNodeClick(item.type)}
                  className="flex items-center gap-3 p-2.5 rounded-lg border border-zinc-800/40 bg-zinc-950/20 hover:bg-zinc-800 hover:border-zinc-700 transition-all duration-150 cursor-grab active:cursor-grabbing text-left group"
                >
                  <div className="p-1.5 rounded bg-zinc-900 border border-zinc-800 group-hover:border-zinc-700">
                    {item.icon}
                  </div>
                  <div className="flex flex-col min-w-0">
                    <span className="text-[12px] font-medium text-zinc-200">
                      {item.label}
                    </span>
                    <span className="text-[10px] text-zinc-500 truncate">
                      {item.description}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
      
      {/* Footer Info */}
      <div className="p-3 border-t border-zinc-800 bg-zinc-950/30 text-[10px] text-zinc-500 font-mono text-center">
        Powered by WebGPU Shaders
      </div>
    </aside>
  );
};
