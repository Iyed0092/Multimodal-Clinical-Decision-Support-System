import React, { useState } from 'react';
import axios from 'axios';
import 'bootstrap/dist/css/bootstrap.min.css';
import XRayModule from './components/X-ray';
import MRIModule from './components/MRI';
import NLPModule from './components/nlp';


const API_URL = "http://localhost:5000/api";

function App() {
  const [activeTab, setActiveTab] = useState('xray');

  return (
    <div className="container mt-5">
      <h1 className="text-center mb-4">MedDiag Platform</h1>
      
      <ul className="nav nav-tabs mb-4">
        <li className="nav-item">
          <button className={`nav-link ${activeTab === 'xray' ? 'active' : ''}`} onClick={() => setActiveTab('xray')}>X-Ray Diagnosis</button>
        </li>
        <li className="nav-item">
          <button className={`nav-link ${activeTab === 'mri' ? 'active' : ''}`} onClick={() => setActiveTab('mri')}>MRI Segmentation</button>
        </li>
        <li className="nav-item">
          <button className={`nav-link ${activeTab === 'nlp' ? 'active' : ''}`} onClick={() => setActiveTab('nlp')}>Clinical NLP</button>
        </li>
      </ul>

      <div className="card p-4 shadow-sm">
        {activeTab === 'xray' && <XRayModule />}
        {activeTab === 'mri' && <MRIModule />}
        {activeTab === 'nlp' && <NLPModule />}
      </div>
    </div>
  );
}






export default App;
