import React from 'react';

interface BayesianUpdate {
  turn: number;
  mean: number;
  variance: number;
}

interface BayesianUpdateChartProps {
  feature: string;
  updates: BayesianUpdate[];
}

export const BayesianUpdateChart: React.FC<BayesianUpdateChartProps> = ({ feature, updates }) => {
  const maxTurn = Math.max(...updates.map(u => u.turn));
  const maxMean = Math.max(...updates.map(u => u.mean + Math.sqrt(u.variance)));

  return (
    <div className="bayesian-chart">
      <h3>{feature} 推断收敛过程</h3>
      <svg width="400" height="200" viewBox="0 0 400 200">
        {updates.map((u, i) => {
          const x = (u.turn / maxTurn) * 380 + 10;
          const y = 190 - (u.mean / maxMean) * 180;
          const errorBar = (Math.sqrt(u.variance) / maxMean) * 180;

          return (
            <g key={i}>
              <line x1={x} y1={y - errorBar} x2={x} y2={y + errorBar} stroke="#666" strokeWidth="1" />
              <circle cx={x} cy={y} r="3" fill="#4CAF50" />
            </g>
          );
        })}
      </svg>
      <div className="chart-info">
        最终: μ={updates[updates.length - 1]?.mean.toFixed(2)}, 
        σ²={updates[updates.length - 1]?.variance.toFixed(2)}
      </div>
    </div>
  );
};
