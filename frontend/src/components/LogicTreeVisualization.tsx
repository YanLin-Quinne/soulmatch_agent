import React from 'react';

interface LogicTreeNode {
  id: string;
  type: 'premise' | 'conclusion';
  content: string;
  confidence: number;
  evidence: string[];
}

interface LogicTreeProps {
  tree: LogicTreeNode[];
  agentName?: string;
}

export const LogicTreeVisualization: React.FC<LogicTreeProps> = ({ tree, agentName }) => {
  return (
    <div className="logic-tree">
      <h3>{agentName || 'Agent'} Reasoning Process</h3>
      <div className="tree-nodes">
        {tree.map((node) => (
          <div key={node.id} className={`tree-node ${node.type}`}>
            <div className="node-content">{node.content}</div>
            <div className="node-confidence">Confidence: {(node.confidence * 100).toFixed(0)}%</div>
            {node.evidence.length > 0 && (
              <div className="node-evidence">
                Evidence: {node.evidence.join(', ')}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};
