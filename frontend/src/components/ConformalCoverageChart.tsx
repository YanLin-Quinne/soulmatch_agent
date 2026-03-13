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
      <h3>{feature} Conformal Prediction</h3>
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
        <div>Coverage Guarantee: {(coverage * 100).toFixed(0)}%</div>
        <div className={isCorrect ? 'success' : 'failure'}>
          {isCorrect ? '✓ True value in prediction set' : '✗ True value not in prediction set'}
        </div>
      </div>
    </div>
  );
};
