import React, { useState, useRef } from 'react';
import type { CanvasNode, CanvasEdge, Port, NodeType } from '../../engine/graphTypes';
import {
  ArrowsOutSimpleIcon,
  CheckCircleIcon,
  WarningIcon,
} from '@phosphor-icons/react';

interface CanvasProps {
  nodes: CanvasNode[];
  edges: CanvasEdge[];
  selectedNodeId: string | null;
  onSelectNode: (nodeId: string | null) => void;
  onUpdateNodes: (nodes: CanvasNode[]) => void;
  onUpdateEdges: (edges: CanvasEdge[]) => void;
  onAddNode: (type: NodeType, x: number, y: number) => void;
}

export const Canvas: React.FC<CanvasProps> = ({
  nodes,
  edges,
  selectedNodeId,
  onSelectNode,
  onUpdateNodes,
  onUpdateEdges,
  onAddNode,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  
  // Transform states
  const [pan, setPan] = useState({ x: 100, y: 80 });
  const [zoom, setZoom] = useState(1.0);
  const [isPanning, setIsPanning] = useState(false);
  const [panStart, setPanStart] = useState({ x: 0, y: 0 });

  // Dragging states
  const [draggingNodeId, setDraggingNodeId] = useState<string | null>(null);
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });

  // Port connection states
  const [activeDragPort, setActiveDragPort] = useState<Port | null>(null);
  const [activeDragSourceNodeId, setActiveDragSourceNodeId] = useState<string | null>(null);
  const [dragMousePos, setDragMousePos] = useState({ x: 0, y: 0 });

  // Constants for coordinate mapping (matching CSS sizes)
  const NODE_WIDTH = 220;
  const PORT_ROW_HEIGHT = 28;
  const HEADER_HEIGHT = 38;

  // Calculate port positions relative to canvas origin
  const getPortPosition = (nodeId: string, portId: string, direction: 'in' | 'out') => {
    const node = nodes.find(n => n.id === nodeId);
    if (!node) return { x: 0, y: 0 };

    const CONTAINER_PADDING_TOP = 6; // Accounts for py-1.5 container padding top offset

    if (direction === 'in') {
      const idx = node.inputs.findIndex(p => p.id === portId);
      const yOffset = HEADER_HEIGHT + CONTAINER_PADDING_TOP + idx * PORT_ROW_HEIGHT + PORT_ROW_HEIGHT / 2;
      return { x: node.x, y: node.y + yOffset };
    } else {
      const idx = node.outputs.findIndex(p => p.id === portId);
      const yOffset = HEADER_HEIGHT + CONTAINER_PADDING_TOP + idx * PORT_ROW_HEIGHT + PORT_ROW_HEIGHT / 2;
      return { x: node.x + NODE_WIDTH, y: node.y + yOffset };
    }
  };

  // Convert client screen coordinates to canvas space coordinates
  const screenToCanvas = (clientX: number, clientY: number) => {
    if (!containerRef.current) return { x: 0, y: 0 };
    const rect = containerRef.current.getBoundingClientRect();
    return {
      x: (clientX - rect.left - pan.x) / zoom,
      y: (clientY - rect.top - pan.y) / zoom,
    };
  };

  // Wheel zoom handler
  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    
    // Zoom anchor point (mouse position)
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    const zoomFactor = 1.1;
    let newZoom = zoom;
    if (e.deltaY < 0) {
      newZoom = Math.min(zoom * zoomFactor, 2.0);
    } else {
      newZoom = Math.max(zoom / zoomFactor, 0.5);
    }

    // Adjust pan to zoom relative to mouse anchor
    const dx = mouseX - pan.x;
    const dy = mouseY - pan.y;
    const newPan = {
      x: mouseX - (dx * newZoom) / zoom,
      y: mouseY - (dy * newZoom) / zoom,
    };

    setZoom(newZoom);
    setPan(newPan);
  };

  // Drag-and-drop drop handler from tools sidebar
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const type = e.dataTransfer.getData('nodeType') as NodeType;
    if (!type) return;

    const canvasPos = screenToCanvas(e.clientX, e.clientY);
    onAddNode(type, canvasPos.x - 100, canvasPos.y - 20);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  // Pointer move handler (handles node dragging, canvas panning, port connecting)
  const handlePointerMove = (e: React.PointerEvent) => {
    if (isPanning) {
      setPan({
        x: e.clientX - panStart.x,
        y: e.clientY - panStart.y,
      });
    } else if (draggingNodeId) {
      const canvasPos = screenToCanvas(e.clientX, e.clientY);
      const updated = nodes.map(n => {
        if (n.id === draggingNodeId) {
          return {
            ...n,
            x: Math.round(canvasPos.x - dragOffset.x),
            y: Math.round(canvasPos.y - dragOffset.y),
          };
        }
        return n;
      });
      onUpdateNodes(updated);
    } else if (activeDragPort) {
      const canvasPos = screenToCanvas(e.clientX, e.clientY);
      setDragMousePos(canvasPos);
    }
  };

  // Pointer up handler (ends panning, dragging, or creates connections)
  const handlePointerUp = (e: React.PointerEvent) => {
    setIsPanning(false);
    setDraggingNodeId(null);

    if (activeDragPort) {
      // Find elements underneath pointer
      const target = e.target as HTMLElement;
      const portId = target.getAttribute('data-port-id');
      const nodeId = target.getAttribute('data-node-id');
      const direction = target.getAttribute('data-port-direction');

      if (portId && nodeId && direction === 'in' && activeDragSourceNodeId !== nodeId) {
        const targetNode = nodes.find(n => n.id === nodeId);
        const targetPort = targetNode?.inputs.find(p => p.id === portId);

        // Verify type compatibility
        if (targetPort && targetPort.type === activeDragPort.type) {
          // Check if this input already has a connection, if so delete it
          let filteredEdges = edges.filter(edge => edge.toPortId !== portId);

          // Add new connection
          const newEdge: CanvasEdge = {
            id: `edge-${activeDragPort.id}-to-${portId}`,
            fromNodeId: activeDragSourceNodeId!,
            fromPortId: activeDragPort.id,
            toNodeId: nodeId,
            toPortId: portId,
          };
          onUpdateEdges([...filteredEdges, newEdge]);
        }
      }
      setActiveDragPort(null);
      setActiveDragSourceNodeId(null);
    }
  };

  const startPan = (e: React.MouseEvent) => {
    if (e.button === 1 || e.button === 0 && e.target === containerRef.current) {
      setIsPanning(true);
      setPanStart({
        x: e.clientX - pan.x,
        y: e.clientY - pan.y,
      });
    }
  };

  const deleteNode = (nodeId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (selectedNodeId === nodeId) onSelectNode(null);
    onUpdateNodes(nodes.filter(n => n.id !== nodeId));
    onUpdateEdges(edges.filter(edge => edge.fromNodeId !== nodeId && edge.toNodeId !== nodeId));
  };

  const getPortColor = (type: Port['type']) => {
    switch (type) {
      case 'text': return 'var(--accent)';
      case 'tokens': return '#38bdf8'; // Sky blue
      case 'tensor': return '#f59e0b'; // Amber
    }
  };

  const getNodeColorClass = (type: NodeType) => {
    if (type === 'text_input' || type === 'tokenizer') {
      return 'border-[var(--accent)]/30 shadow-[var(--accent-glow)]';
    }
    if (type === 'wte' || type === 'wpe') {
      return 'border-[#38bdf8]/30 shadow-[rgba(56,189,248,0.04)]';
    }
    if (type === 'lm_head' || type === 'softmax' || type === 'sampler') {
      return 'border-[#fb7185]/30 shadow-[rgba(251,113,133,0.04)]';
    }
    return 'border-zinc-800 shadow-[rgba(0,0,0,0.3)]';
  };

  return (
    <div
      ref={containerRef}
      className="relative flex-1 h-full overflow-hidden outline-none bg-zinc-950/40 select-none cursor-grab"
      style={{ cursor: isPanning ? 'grabbing' : activeDragPort ? 'crosshair' : 'grab' }}
      onWheel={handleWheel}
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      onMouseDown={startPan}
    >
      {/* Background Dot Grid */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          background: `radial-gradient(circle, var(--border-color) 1px, transparent 1px)`,
          backgroundSize: `${24 * zoom}px ${24 * zoom}px`,
          backgroundPosition: `${pan.x}px ${pan.y}px`,
        }}
      />

      {/* SVG Connections Overlay */}
      <svg
        className="absolute inset-0 w-full h-full pointer-events-none"
        style={{ zIndex: 1 }}
      >
        <g transform={`translate(${pan.x}, ${pan.y}) scale(${zoom})`}>
          {/* Active drag line */}
          {activeDragPort && activeDragSourceNodeId && (
            (() => {
              const start = getPortPosition(activeDragSourceNodeId, activeDragPort.id, 'out');
              const end = dragMousePos;
              const dx = Math.max(30, Math.abs(end.x - start.x) * 0.5);
              const path = `M ${start.x} ${start.y} C ${start.x + dx} ${start.y}, ${end.x - dx} ${end.y}, ${end.x} ${end.y}`;
              return (
                <path
                  d={path}
                  stroke={getPortColor(activeDragPort.type)}
                  strokeWidth="2.5"
                  fill="none"
                  strokeDasharray="4 4"
                  className="animate-[dash_10s_linear_infinite]"
                />
              );
            })()
          )}

          {/* Connected edges */}
          {edges.map(edge => {
            const fromNode = nodes.find(n => n.id === edge.fromNodeId);
            const toNode = nodes.find(n => n.id === edge.toNodeId);
            if (!fromNode || !toNode) return null;

            const start = getPortPosition(edge.fromNodeId, edge.fromPortId, 'out');
            const end = getPortPosition(edge.toNodeId, edge.toPortId, 'in');

            const dx = Math.max(30, Math.abs(end.x - start.x) * 0.5);
            const path = `M ${start.x} ${start.y} C ${start.x + dx} ${start.y}, ${end.x - dx} ${end.y}, ${end.x} ${end.y}`;
            const port = fromNode.outputs.find(p => p.id === edge.fromPortId);
            const color = port ? getPortColor(port.type) : 'var(--text-muted)';
            const isSelected = selectedNodeId === edge.fromNodeId || selectedNodeId === edge.toNodeId;

            return (
              <g key={edge.id} className="group pointer-events-auto cursor-pointer">
                {/* Wider invisible path for easier selection/hover */}
                <path
                  d={path}
                  stroke="transparent"
                  strokeWidth="10"
                  fill="none"
                  onClick={(e) => {
                    e.stopPropagation();
                    onUpdateEdges(edges.filter(ed => ed.id !== edge.id));
                  }}
                />
                {/* Visual path */}
                <path
                  d={path}
                  stroke={color}
                  strokeWidth={isSelected ? '3' : '2'}
                  fill="none"
                  opacity={isSelected ? 1.0 : 0.65}
                  className="transition-all duration-150 group-hover:opacity-100 group-hover:stroke-width-[3]"
                />
                {/* Animated flow overlay: dashes march from output -> input */}
                <path
                  d={path}
                  stroke={color}
                  strokeWidth={isSelected ? '3' : '2'}
                  fill="none"
                  strokeLinecap="round"
                  className="edge-flow"
                  opacity={0.9}
                />
                {/* Moving arrowhead travelling along the wire */}
                <path
                  d="M -5 -4 L 4 0 L -5 4 Z"
                  fill={color}
                  opacity={0.95}
                >
                  <animateMotion
                    dur="1.8s"
                    repeatCount="indefinite"
                    rotate="auto"
                    path={path}
                  />
                </path>
              </g>
            );
          })}
        </g>
      </svg>

      {/* Nodes Workspace */}
      <div
        className="absolute inset-0 origin-top-left pointer-events-none"
        style={{
          transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
          zIndex: 2,
        }}
      >
        {nodes.map(node => {
          const isSelected = node.id === selectedNodeId;
          const statusIcon = node.status === 'success' ? (
            <CheckCircleIcon size={14} className="text-emerald-500" />
          ) : node.status === 'error' ? (
            <WarningIcon size={14} className="text-rose-500" />
          ) : node.status === 'running' ? (
            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-ping" />
          ) : null;

          return (
            <div
              key={node.id}
              style={{
                left: node.x,
                top: node.y,
                width: NODE_WIDTH,
                zIndex: isSelected ? 50 : 10,
              }}
              className={`absolute pointer-events-auto flex flex-col rounded-lg bg-zinc-900 border text-zinc-300 font-sans shadow-md hover:border-zinc-700 transition-colors duration-150 ${
                isSelected ? 'border-[var(--accent)] ring-1 ring-[var(--accent)]/50' : 'border-zinc-800'
              } ${getNodeColorClass(node.type)}`}
              onClick={(e) => {
                e.stopPropagation();
                onSelectNode(node.id);
              }}
              onPointerDown={(e) => {
                const target = e.target as HTMLElement;
                if (target.closest('.port-circle') || target.closest('.delete-btn') || target.closest('input')) return;
                
                onSelectNode(node.id);
                setDraggingNodeId(node.id);
                
                // Track offset in canvas space
                const canvasPos = screenToCanvas(e.clientX, e.clientY);
                setDragOffset({
                  x: canvasPos.x - node.x,
                  y: canvasPos.y - node.y,
                });
              }}
            >
              {/* Node Header */}
              <div className="flex items-center justify-between px-3 h-[38px] border-b border-zinc-800 bg-zinc-950/60 rounded-t-lg">
                <div className="flex items-center gap-1.5 min-w-0">
                  {statusIcon}
                  <span className="text-[12px] font-semibold tracking-wide truncate text-zinc-200">
                    {node.label}
                  </span>
                </div>
                <button
                  onClick={(e) => deleteNode(node.id, e)}
                  className="delete-btn opacity-0 hover:bg-zinc-800 text-zinc-500 hover:text-zinc-300 rounded p-1 group-hover:opacity-100 transition-opacity duration-150"
                  style={{ display: isSelected ? 'block' : 'none' }}
                >
                  &times;
                </button>
              </div>

              {/* Node Ports Section */}
              <div className="flex flex-col py-1.5 min-h-[40px] text-[11px] font-mono">
                {/* Map inputs and outputs side by side or row-based */}
                {Array.from({ length: Math.max(node.inputs.length, node.outputs.length) }).map((_, i) => {
                  const input = node.inputs[i];
                  const output = node.outputs[i];

                  return (
                    <div key={i} className="flex justify-between items-center h-[28px] px-2.5 relative">
                      {/* Input Port */}
                      {input ? (
                        <div className="flex items-center gap-2 min-w-0">
                          <div
                            data-port-id={input.id}
                            data-node-id={node.id}
                            data-port-direction="in"
                            className="port-circle w-3 h-3 rounded-full border border-zinc-950 hover:scale-125 transition-transform duration-100 cursor-crosshair"
                            style={{
                              backgroundColor: getPortColor(input.type),
                              marginLeft: '-15px',
                              zIndex: 10,
                            }}
                            title={`${input.name}: ${input.shapeDescription}`}
                          />
                          <span className="text-zinc-400 truncate pr-1" title={input.shapeDescription}>
                            {input.name}
                          </span>
                        </div>
                      ) : <div />}

                      {/* Output Port */}
                      {output ? (
                        <div className="flex items-center gap-2 min-w-0 ml-auto">
                          <span className="text-zinc-400 truncate pl-1" title={output.shapeDescription}>
                            {output.name}
                          </span>
                          <div
                            data-port-id={output.id}
                            data-node-id={node.id}
                            data-port-direction="out"
                            className="port-circle w-3 h-3 rounded-full border border-zinc-950 hover:scale-125 transition-transform duration-100 cursor-crosshair"
                            style={{
                              backgroundColor: getPortColor(output.type),
                              marginRight: '-15px',
                              zIndex: 10,
                            }}
                            onPointerDown={(e) => {
                              e.stopPropagation();
                              setActiveDragPort(output);
                              setActiveDragSourceNodeId(node.id);
                              
                              const canvasPos = screenToCanvas(e.clientX, e.clientY);
                              setDragMousePos(canvasPos);
                            }}
                            title={`${output.name}: ${output.shapeDescription}`}
                          />
                        </div>
                      ) : <div />}
                    </div>
                  );
                })}
              </div>

              {/* Node Inspector summary info if runs successfully */}
              {node.outputShape && (
                <div className="px-3 py-1 bg-zinc-950/20 border-t border-zinc-800/50 text-[10px] text-zinc-500 font-mono text-right rounded-b-lg">
                  shape: [{node.outputShape.join(', ')}]
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Control Buttons for Canvas view (Reset Zoom / Pan) */}
      <div className="absolute bottom-4 left-4 flex gap-2" style={{ zIndex: 100 }}>
        <button
          onClick={() => {
            setPan({ x: 100, y: 80 });
            setZoom(1.0);
          }}
          className="flex items-center gap-1.5 px-2.5 py-1.5 rounded bg-zinc-900 hover:bg-zinc-800 text-zinc-300 hover:text-white border border-zinc-800 text-[11px] font-medium transition-colors cursor-pointer"
        >
          <ArrowsOutSimpleIcon size={13} />
          Reset Zoom
        </button>
      </div>
    </div>
  );
};
