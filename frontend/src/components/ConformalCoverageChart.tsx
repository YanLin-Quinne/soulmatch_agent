import React from 'react';

interface ConformalCoverageProps {
  feature: string;
  predictionSet: string[];
  trueValue: string;
  coverage: number;
}

export const ConformalCoverageChart: React.FC<ConformalCoverageProps> = ({
  feature,
  predictionSet,
  trueValue,
  coverage
}) => {
  const isCorrect = predictionSet.includes(trueValue);

  return (
    <div className="conformal-coverage">
      <h3>{feature} 共形预测</h3>
      <div className="prediction-set">
        {predictionSet.map((val, i) => (
          <span
            key={i}
            className={`prediction-item ${val === trueValue ? 'correct' : ''}`}
          >
            {val}
          </span>
        ))}
      </div>
      <div className="coverage-info">
        <div>覆盖保证: {(coverage * 100).toFixed(0)}%</div>
        <div className={isCorrect ? 'success' : 'failure'}>
          {isCorrect ? '✓ 真实值在预测集内' : '✗ 真实值不在预测集内'}
        </div>
      </div>
    </div>
  );
};
