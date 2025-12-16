import React, { useState } from "react";
import axios from "axios";
import { API_URL } from './config';

function XRayModule() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handlePredict = async () => {
    if (!file) return;
    setLoading(true);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await axios.post(`${API_URL}/xray`, formData);
      setResult(res.data);
    } catch {
      alert("Error processing X-Ray image");
    }

    setLoading(false);
  };

  return (
    <div>
      <h3>Pneumonia Detection</h3>
      <input type="file" className="form-control mb-3" onChange={e => setFile(e.target.files[0])} />
      <button className="btn btn-primary" onClick={handlePredict} disabled={loading}>
        {loading ? "Analyzing..." : "Analyze X-Ray"}
      </button>

      {result && (
        <div className="row mt-4">
          <div className="col-md-6">
            <h5>Prediction: <span className={result.label === 'Pneumonia' ? 'text-danger' : 'text-success'}>{result.label}</span></h5>
            <p>Confidence: {(result.confidence * 100).toFixed(2)}%</p>
          </div>
          <div className="col-md-6">
            <h5>Grad-CAM Explanation:</h5>
            <img src={`data:image/jpeg;base64,${result.heatmap_image}`} className="img-fluid rounded border" alt="Grad-CAM Heatmap" />
          </div>
        </div>
      )}
    </div>
  );
}

export default XRayModule;