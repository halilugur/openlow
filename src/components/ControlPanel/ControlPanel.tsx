import React, { useState, useEffect, useRef } from 'react';
import {
  PlayIcon,
  SparkleIcon,
  TerminalIcon,
  CpuIcon,
  TrashIcon,
  EyeIcon,
  EyeSlashIcon,
  CopyIcon,
  BrainIcon,
  ArrowRightIcon,
  DatabaseIcon,
  SlidersIcon,
} from '@phosphor-icons/react';

interface LogMessage {
  timestamp: string;
  type: 'info' | 'success' | 'error' | 'warning' | 'gpu';
  message: string;
}

export interface TrainParams {
  nEmbd: number;
  blockSize: number;
  mlpMult: number;
  lr: number;
  steps: number;
  batch: number;
}

interface ControlPanelProps {
  onRun: () => void;
  onTrain: () => void;
  isTraining: boolean;
  onAutoGenerateToggle: (enabled: boolean) => void;
  isAutoGenerating: boolean;
  onClearCanvas: () => void;
  onLoadPreset: (name: string) => void;
  onLoadTrainingData: (name: string) => void;
  logs: LogMessage[];
  gpuAvailable: boolean;
  executionTimeMs: number | null;
  isRunning: boolean;
  generatedText: string;
  // Chat props
  chatHistory: { role: 'user' | 'assistant'; content: string }[];
  isChatResponding: boolean;
  onSendChatMessage: (message: string) => void;
  // Compute backend toggle
  computeBackend: 'cpu' | 'webgpu';
  onSetComputeBackend: (b: 'cpu' | 'webgpu') => void;
  // Training hyperparameters
  trainParams: TrainParams;
  onSetTrainParams: (p: TrainParams) => void;
}

export const ControlPanel: React.FC<ControlPanelProps> = ({
  onRun,
  onTrain,
  isTraining,
  onAutoGenerateToggle,
  isAutoGenerating,
  onClearCanvas,
  onLoadPreset,
  onLoadTrainingData,
  logs,
  gpuAvailable,
  executionTimeMs,
  isRunning,
  generatedText,
  chatHistory,
  isChatResponding,
  onSendChatMessage,
  computeBackend,
  onSetComputeBackend,
  trainParams,
  onSetTrainParams,
}) => {
  const [showConsole, setShowConsole] = useState(true);
  const [activeTab, setActiveTab] = useState<'chat' | 'raw'>('chat');
  const [chatInput, setChatInput] = useState('');
  const [showSettings, setShowSettings] = useState(false);
  const consoleEndRef = useRef<HTMLDivElement>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);
  
  const [copiedLogs, setCopiedLogs] = useState(false);
  const [copiedText, setCopiedText] = useState(false);

  const handleCopyLogs = () => {
    const text = logs.map(log => `[${log.timestamp}] ${log.message}`).join('\n');
    navigator.clipboard.writeText(text);
    setCopiedLogs(true);
    setTimeout(() => setCopiedLogs(false), 2000);
  };

  const handleCopyText = () => {
    navigator.clipboard.writeText(generatedText);
    setCopiedText(true);
    setTimeout(() => setCopiedText(false), 2000);
  };

  // Auto scroll console logs to bottom
  useEffect(() => {
    if (consoleEndRef.current) {
      consoleEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs]);

  // Auto scroll chat to bottom
  useEffect(() => {
    if (chatEndRef.current) {
      chatEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [chatHistory]);

  const getLogColor = (type: LogMessage['type']) => {
    switch (type) {
      case 'success': return 'text-emerald-400';
      case 'error': return 'text-rose-400';
      case 'warning': return 'text-amber-400';
      case 'gpu': return 'text-sky-400';
      default: return 'text-zinc-400';
    }
  };

  return (
    <div className="bg-zinc-900 border-t border-zinc-800 flex flex-col w-full shrink-0 select-none">
      {/* Dashboard Top bar (Status & Trigger) */}
      <div className="px-4 py-2 border-b border-zinc-800 flex items-center justify-between bg-zinc-950/20">
        <div className="flex items-center gap-6">
          {/* WebGPU Device Status indicator */}
          <div
            className={`flex items-center gap-1.5 h-7 px-2.5 rounded-md border font-mono text-[11px] shrink-0 ${
              gpuAvailable
                ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400'
                : 'bg-rose-500/10 border-rose-500/30 text-rose-400'
            }`}
          >
            <CpuIcon size={14} weight="bold" />
            <span className="font-bold tracking-wide whitespace-nowrap">
              WebGPU {gpuAvailable ? 'ACTIVE' : 'OFF'}
            </span>
          </div>

          {/* Time log */}
          {executionTimeMs !== null && (
            <div className="text-[11px] font-mono text-zinc-400 whitespace-nowrap shrink-0">
              Execution Time:{' '}
              <span className="text-emerald-400 font-bold">
                {executionTimeMs.toFixed(2)} ms
              </span>
            </div>
          )}

          {/* Preset templates options */}
          <div className="flex items-center gap-2 border-l border-zinc-800 pl-6 text-xs">
            <span className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider">
              Load Blueprint Preset:
            </span>
            <button
              onClick={() => onLoadPreset('embeddings')}
              disabled={isRunning}
              className="px-2.5 py-1 text-[11px] font-mono bg-zinc-950 border border-zinc-850 hover:border-zinc-700 text-zinc-300 hover:text-white rounded transition-colors cursor-pointer disabled:opacity-50"
            >
              1. Embeddings
            </button>
            <button
              onClick={() => onLoadPreset('layernorm')}
              disabled={isRunning}
              className="px-2.5 py-1 text-[11px] font-mono bg-zinc-950 border border-zinc-850 hover:border-zinc-700 text-zinc-300 hover:text-white rounded transition-colors cursor-pointer disabled:opacity-50"
            >
              2. LayerNorm
            </button>
            <button
              onClick={() => onLoadPreset('full')}
              disabled={isRunning}
              className="px-2.5 py-1 text-[11px] font-mono bg-zinc-950 border border-zinc-850 hover:border-zinc-700 text-zinc-300 hover:text-white rounded transition-colors cursor-pointer disabled:opacity-50"
            >
              3. Full GPT Layer
            </button>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {/* Reset Blueprint */}
          <button
            onClick={onClearCanvas}
            disabled={isRunning}
            className="flex items-center gap-1.5 px-3 h-8 text-[11px] font-medium bg-zinc-900 border border-zinc-800 text-rose-500 hover:text-rose-400 rounded-md transition-colors cursor-pointer disabled:opacity-50 whitespace-nowrap"
          >
            <TrashIcon size={14} />
            Reset Canvas
          </button>

          {/* Auto Generation (Iterative sampler) */}
          <button
            onClick={() => onAutoGenerateToggle(!isAutoGenerating)}
            disabled={(isRunning && !isAutoGenerating) || isTraining || isChatResponding}
            className={`flex items-center justify-center gap-1.5 px-3 text-[11px] font-semibold border rounded-md transition-all duration-150 cursor-pointer disabled:opacity-50 w-[130px] h-8 shrink-0 whitespace-nowrap ${
              isAutoGenerating
                ? 'bg-emerald-600 border-emerald-500 text-white shadow-[0_0_10px_rgba(16,185,129,0.2)]'
                : 'bg-zinc-900 border-zinc-850 hover:border-zinc-700 text-zinc-300'
            }`}
          >
            <SparkleIcon size={14} />
            {isAutoGenerating ? 'Stop Auto-Gen' : 'Loop Auto-Gen'}
          </button>

          {/* Train Model (CPU/GPU fine-tuning loop) */}
          <button
            onClick={onTrain}
            disabled={(isRunning && !isAutoGenerating) || isTraining || isAutoGenerating || isChatResponding}
            className={`flex items-center justify-center gap-1.5 px-3 text-[11px] font-semibold border rounded-md transition-all duration-150 cursor-pointer disabled:opacity-50 w-[120px] h-8 shrink-0 whitespace-nowrap ${
              isTraining
                ? 'bg-sky-600 border-sky-500 text-white shadow-[0_0_10px_rgba(56,189,248,0.2)]'
                : 'bg-zinc-900 border-zinc-850 hover:border-zinc-700 text-zinc-300'
            }`}
          >
            <BrainIcon size={14} />
            {isTraining ? 'Training...' : 'Train Model'}
          </button>

          {/* Training hyperparameter settings */}
          <div className="relative shrink-0">
            <button
              onClick={() => setShowSettings(s => !s)}
              disabled={isTraining}
              title="Training hyperparameters"
              className={`flex items-center justify-center h-8 w-8 rounded-md border transition-colors cursor-pointer disabled:opacity-50 ${
                showSettings
                  ? 'bg-zinc-800 border-zinc-600 text-white'
                  : 'bg-zinc-900 border-zinc-850 hover:border-zinc-700 text-zinc-400 hover:text-white'
              }`}
            >
              <SlidersIcon size={15} />
            </button>

            {showSettings && (
              <div className="absolute bottom-[calc(100%+8px)] right-0 z-50 w-[260px] bg-zinc-900 border border-zinc-700 rounded-lg shadow-2xl shadow-black/50 p-3">
                <div className="flex items-center justify-between mb-2.5">
                  <span className="text-[10px] font-bold uppercase tracking-wider text-zinc-300">
                    Training Hyperparameters
                  </span>
                  <button
                    onClick={() => onSetTrainParams({ nEmbd: 64, blockSize: 64, mlpMult: 4, lr: 0.003, steps: 2500, batch: 16 })}
                    className="text-[9px] font-mono text-emerald-400 hover:text-emerald-300 cursor-pointer"
                    title="Reset to defaults"
                  >
                    RESET
                  </button>
                </div>
                <div className="grid grid-cols-2 gap-2">
                  {([
                    { key: 'nEmbd', label: 'Embed dim', step: 16, min: 16, max: 256 },
                    { key: 'blockSize', label: 'Context len', step: 16, min: 8, max: 256 },
                    { key: 'mlpMult', label: 'MLP mult', step: 1, min: 1, max: 8 },
                    { key: 'lr', label: 'Learn rate', step: 0.001, min: 0.0001, max: 0.1 },
                    { key: 'steps', label: 'Steps', step: 100, min: 50, max: 20000 },
                    { key: 'batch', label: 'Batch size', step: 1, min: 1, max: 64 },
                  ] as const).map(f => (
                    <label key={f.key} className="flex flex-col gap-0.5">
                      <span className="text-[9px] font-mono text-zinc-500 uppercase tracking-wide">{f.label}</span>
                      <input
                        type="number"
                        value={trainParams[f.key]}
                        min={f.min}
                        max={f.max}
                        step={f.step}
                        onChange={(e) => {
                          const v = parseFloat(e.target.value);
                          if (!isNaN(v)) onSetTrainParams({ ...trainParams, [f.key]: v });
                        }}
                        className="w-full px-2 py-1 text-[11px] font-mono bg-zinc-950 border border-zinc-800 focus:border-emerald-600 outline-none text-zinc-100 rounded"
                      />
                    </label>
                  ))}
                </div>
                <p className="mt-2.5 text-[9px] text-zinc-500 leading-relaxed">
                  Bigger embed dim / more steps = smarter but slower. Changes apply on the next <span className="text-zinc-300 font-bold">Train Model</span> run.
                </p>
              </div>
            )}
          </div>
          {/* Compute backend toggle: CPU (always) vs WebGPU (verified) */}
          <div
            className="flex items-center h-8 rounded-md border border-zinc-800 bg-zinc-950 overflow-hidden shrink-0"
            title={gpuAvailable ? 'Choose the compute backend for real training & chat' : 'WebGPU not available in this browser'}
          >
            <span className="px-2 text-[9px] font-bold uppercase tracking-wider text-zinc-500 border-r border-zinc-800 self-stretch flex items-center">
              Backend
            </span>
            {(['cpu', 'webgpu'] as const).map(b => (
              <button
                key={b}
                onClick={() => onSetComputeBackend(b)}
                disabled={isTraining || isChatResponding || (b === 'webgpu' && !gpuAvailable)}
                className={`h-full px-2.5 text-[10px] font-mono font-bold tracking-wide transition-colors cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed ${
                  computeBackend === b
                    ? 'bg-emerald-600 text-white'
                    : 'bg-transparent text-zinc-400 hover:text-zinc-200'
                }`}
              >
                {b === 'cpu' ? 'CPU' : 'WebGPU'}
              </button>
            ))}
          </div>
          {/* Run forward computation */}
          <button
            onClick={onRun}
            disabled={(isRunning && !isAutoGenerating) || isTraining || isAutoGenerating || isChatResponding}
            className="flex items-center justify-center gap-1.5 px-3 h-8 text-[11px] font-bold bg-emerald-500 hover:bg-emerald-400 text-zinc-950 border border-emerald-400 rounded-md shadow-[0_0_15px_rgba(16,185,129,0.35)] transition-all duration-150 cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed w-[160px] shrink-0"
          >
            <PlayIcon weight="fill" size={13} />
            {isRunning ? 'Executing...' : 'Run Pipeline'}
          </button>
        </div>
      </div>

      {/* Expandable Dashboard Body */}
      {showConsole && (
        <div className="flex h-[200px] border-t border-zinc-850/50">
          {/* Monospaced Log Console */}
          <div className="flex-1 flex flex-col bg-zinc-950 p-3 overflow-y-auto border-r border-zinc-850/50 select-text selection:bg-emerald-500/20">
            <div className="flex items-center justify-between border-b border-zinc-900 pb-1.5 mb-2 shrink-0 select-none">
              <div className="flex items-center gap-1.5">
                <TerminalIcon size={14} className="text-zinc-500" />
                <span className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider">
                  GPU COMPUTE LOGGER
                </span>
              </div>
              <div className="flex items-center gap-1.5">
                {logs.length > 0 && (
                  <button
                    onClick={handleCopyLogs}
                    className="flex items-center gap-1 px-2 py-0.5 text-[9px] font-mono bg-zinc-900 border border-zinc-800 text-zinc-400 hover:text-white rounded hover:bg-zinc-800 transition-colors cursor-pointer"
                    title="Copy logs to clipboard"
                  >
                    <CopyIcon size={11} />
                    {copiedLogs ? 'COPIED!' : 'COPY'}
                  </button>
                )}
                <button
                  onClick={() => setShowConsole(false)}
                  className="flex items-center gap-1 px-2 py-0.5 text-[9px] font-mono bg-zinc-900 border border-zinc-800 text-zinc-400 hover:text-white rounded hover:bg-zinc-800 transition-colors cursor-pointer"
                  title="Hide console"
                >
                  <EyeSlashIcon size={11} />
                  HIDE
                </button>
              </div>
            </div>
            
            <div className="flex-1 overflow-y-auto flex flex-col gap-1 font-mono text-[11px]">
              {logs.length === 0 ? (
                <div className="text-zinc-600 italic">Console idle. Hit "Run Graph Pipeline" to start.</div>
              ) : (
                logs.map((log, index) => (
                  <div key={index} className="flex gap-2 leading-relaxed">
                    <span className="text-zinc-600 shrink-0 select-none">[{log.timestamp}]</span>
                    <span className={`${getLogColor(log.type)}`}>{log.message}</span>
                  </div>
                ))
              )}
              <div ref={consoleEndRef} />
            </div>
          </div>

          {/* Generated Text Outputs Display */}
          <div className="w-[420px] flex flex-col bg-zinc-950/60 p-3 select-none shrink-0 border-l border-zinc-850/50">
            <div className="flex items-center justify-between border-b border-zinc-900 pb-1.5 mb-2 shrink-0">
              <div className="flex items-center gap-3">
                <button
                  onClick={() => setActiveTab('chat')}
                  className={`text-[10px] font-bold uppercase tracking-wider pb-1.5 border-b -mb-2 transition-colors cursor-pointer ${
                    activeTab === 'chat'
                      ? 'text-emerald-400 border-emerald-500 font-bold'
                      : 'text-zinc-500 border-transparent hover:text-zinc-350'
                  }`}
                >
                  CHAT PLAYGROUND
                </button>
                <button
                  onClick={() => setActiveTab('raw')}
                  className={`text-[10px] font-bold uppercase tracking-wider pb-1.5 border-b -mb-2 transition-colors cursor-pointer ${
                    activeTab === 'raw'
                      ? 'text-emerald-400 border-emerald-500 font-bold'
                      : 'text-zinc-500 border-transparent hover:text-zinc-350'
                  }`}
                >
                  RAW GENERATOR
                </button>
              </div>
              <div className="flex items-center gap-1.5">
                <button
                  onClick={() => onLoadTrainingData('qa')}
                  disabled={isRunning || isTraining}
                  className="flex items-center gap-1 px-2 py-0.5 text-[9px] font-mono bg-zinc-900 border border-zinc-800 text-zinc-400 hover:text-white rounded hover:bg-zinc-800 transition-colors cursor-pointer disabled:opacity-50"
                  title="Load the QA training dataset into the Text Input node"
                >
                  <DatabaseIcon size={11} />
                  LOAD QA DATA
                </button>
                {activeTab === 'raw' && generatedText && (
                  <button
                    onClick={handleCopyText}
                    className="flex items-center gap-1 px-2 py-0.5 text-[9px] font-mono bg-zinc-900 border border-zinc-800 text-zinc-400 hover:text-white rounded hover:bg-zinc-800 transition-colors cursor-pointer"
                    title="Copy generated text to clipboard"
                  >
                    <CopyIcon size={11} />
                    {copiedText ? 'COPIED!' : 'COPY'}
                  </button>
                )}
              </div>
            </div>

            {activeTab === 'chat' ? (
              <div className="flex-1 flex flex-col min-h-0 bg-zinc-950/80 border border-zinc-900 rounded-lg overflow-hidden">
                {/* Chat Message History */}
                <div className="flex-1 overflow-y-auto p-3 flex flex-col gap-2.5 select-text selection:bg-emerald-500/30">
                  {chatHistory.length === 0 ? (
                    <div className="flex-1 flex flex-col items-center justify-center text-center p-4 select-none">
                      <BrainIcon size={24} className="text-zinc-700 mb-1.5" />
                      <div className="text-[10px] text-zinc-500 font-mono">
                        Chat Playground is idle.<br />Type a message below to prompt the WebGPU graph!
                      </div>
                    </div>
                  ) : (
                    chatHistory.map((msg, idx) => (
                      <div
                        key={idx}
                        className={`group flex flex-col max-w-[85%] ${
                          msg.role === 'user' ? 'ml-auto items-end' : 'mr-auto items-start'
                        }`}
                      >
                        <span className="text-[8px] font-mono text-zinc-500 mb-0.5 select-none">
                          {msg.role === 'user' ? 'USER' : 'WEBGPU MODEL'}
                        </span>
                        <div className="flex items-center gap-1.5 relative">
                          {msg.role === 'user' && msg.content && (
                            <button
                              onClick={() => navigator.clipboard.writeText(msg.content)}
                              className="opacity-0 group-hover:opacity-100 transition-opacity p-1 text-zinc-500 hover:text-zinc-300 bg-zinc-900 border border-zinc-800 rounded hover:bg-zinc-800 cursor-pointer shrink-0"
                              title="Copy message"
                            >
                              <CopyIcon size={10} />
                            </button>
                          )}
                          <div
                            className={`rounded-lg px-2.5 py-1.5 text-xs font-mono break-words leading-relaxed ${
                              msg.role === 'user'
                                ? 'bg-zinc-850 text-zinc-150 border border-zinc-700/50'
                                : 'bg-emerald-950/30 text-emerald-400 border border-emerald-500/20'
                            }`}
                          >
                            {msg.content || (
                              <span className="flex gap-1 items-center py-0.5 select-none">
                                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-bounce" />
                                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-bounce [animation-delay:0.2s]" />
                                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-bounce [animation-delay:0.4s]" />
                              </span>
                            )}
                          </div>
                          {msg.role === 'assistant' && msg.content && (
                            <button
                              onClick={() => navigator.clipboard.writeText(msg.content)}
                              className="opacity-0 group-hover:opacity-100 transition-opacity p-1 text-zinc-500 hover:text-emerald-400 bg-zinc-900 border border-zinc-800 rounded hover:bg-zinc-800 cursor-pointer shrink-0"
                              title="Copy response"
                            >
                              <CopyIcon size={10} />
                            </button>
                          )}
                        </div>
                      </div>
                    ))
                  )}
                  <div ref={chatEndRef} />
                </div>

                {/* Input Bar */}
                <form
                  onSubmit={(e) => {
                    e.preventDefault();
                    if (!chatInput.trim() || isChatResponding) return;
                    onSendChatMessage(chatInput.trim());
                    setChatInput('');
                  }}
                  className="h-[36px] border-t border-zinc-900 flex items-center bg-zinc-950 shrink-0"
                >
                  <input
                    type="text"
                    value={chatInput}
                    onChange={(e) => setChatInput(e.target.value)}
                    disabled={isChatResponding || (isRunning && !isAutoGenerating) || isTraining || isAutoGenerating}
                    placeholder={
                      isChatResponding
                        ? "Model responding..."
                        : isAutoGenerating
                        ? "Auto-Gen running..."
                        : !gpuAvailable
                        ? "Send message (Sim fallback)..."
                        : "Send message..."
                    }
                    className="flex-1 h-full px-3 text-xs bg-transparent border-none outline-none text-zinc-100 placeholder-zinc-650 disabled:cursor-not-allowed font-mono"
                  />
                  <button
                    type="submit"
                    disabled={!chatInput.trim() || isChatResponding || (isRunning && !isAutoGenerating) || isTraining || isAutoGenerating}
                    className="h-full px-3 border-l border-zinc-900 hover:bg-zinc-900 text-emerald-400 disabled:text-zinc-700 transition-colors cursor-pointer"
                  >
                    <ArrowRightIcon size={13} />
                  </button>
                </form>
              </div>
            ) : (
              /* Raw Generator Display */
              <div className="flex-1 bg-zinc-950/90 border border-zinc-900 rounded-lg p-2.5 font-mono text-xs overflow-y-auto text-emerald-400 leading-relaxed break-all select-text selection:bg-emerald-500/30">
                {generatedText || (
                  <span className="text-zinc-600 italic select-none">
                    Tokens generated by the output Sampler node will render here...
                  </span>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Slim bar to reopen the console when collapsed */}
      {!showConsole && (
        <button
          onClick={() => setShowConsole(true)}
          className="h-7 w-full flex items-center justify-center gap-1.5 text-[10px] font-mono text-zinc-500 hover:text-zinc-200 hover:bg-zinc-950/40 border-t border-zinc-850/50 transition-colors cursor-pointer"
          title="Show console & chat"
        >
          <EyeIcon size={12} />
          Show Console &amp; Chat
        </button>
      )}
    </div>
  );
};
