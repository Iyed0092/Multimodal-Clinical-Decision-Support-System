import React, { useState } from "react";
import axios from "axios";
import { API_URL } from './config';

function NLPModule() {
  const [text, setText] = useState('');
  const [question, setQuestion] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleAnalyze = async () => {
    setLoading(true);

    try {
      const res = await axios.post(`${API_URL}/nlp`, { text, question });
      setResult(res.data);
    } catch {
      alert("Error analyzing text");
    }

    setLoading(false);
  };

  return (
    <div>
      <h3>Clinical Notes Analysis</h3>
      <textarea className="form-control mb-3" rows="5" placeholder="Paste clinical notes..." value={text} onChange={e => setText(e.target.value)} />
      <input type="text" className="form-control mb-3" placeholder="Ask a question (optional)" value={question} onChange={e => setQuestion(e.target.value)} />
      <button className="btn btn-info text-white" onClick={handleAnalyze} disabled={loading}>
        {loading ? "Processing..." : "Extract Insights"}
      </button>

      {result && (
        <div className="mt-4">
          <h5>Extracted Entities:</h5>
          <div className="d-flex flex-wrap gap-2">
            {result.entities.map((e, i) => (
              <span key={i} className="badge bg-secondary p-2">{e.text} <small>({e.label})</small></span>
            ))}
          </div>
          {result.answer && (
            <div className="alert alert-success mt-3">
              <strong>Answer:</strong> {result.answer.answer}<br/>
              <small>Confidence: {(result.answer.score * 100).toFixed(2)}%</small>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default NLPModule;
