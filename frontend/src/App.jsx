import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [vitals, setVitals] = useState({
    age: '',
    bmi: '',
    systolic: '',
    diastolic: '',
    history_score: '1'
  });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    setFile(selected);
    setPreview(URL.createObjectURL(selected));
  };

  const handleAnalyze = async () => {
    if (!file) return alert("Upload image first!");
    setLoading(true);

    const formData = new FormData();
    formData.append('image', file);
    formData.append('patient_data', JSON.stringify(vitals));

    try {
      const response = await axios.post(
        'http://127.0.0.1:8000/analyze',
        formData
      );
      setResult(response.data);
    } catch (error) {
      alert("Backend connect nahi ho raha!");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: 40, fontFamily: 'Arial', maxWidth: 800, margin: 'auto' }}>
      <h1>🏥 AI Health Diagnostic System</h1>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
        
        <div style={{ border: '1px solid #ccc', padding: 20 }}>
          <h3>Patient Vitals</h3>
          {Object.keys(vitals).map((key) => (
            <div key={key} style={{ marginBottom: 10 }}>
              <label>{key}</label>
              <input
                type="number"
                value={vitals[key]}
                onChange={(e) =>
                  setVitals({ ...vitals, [key]: e.target.value })
                }
                style={{ width: '100%' }}
              />
            </div>
          ))}
        </div>

        <div style={{ border: '1px solid #ccc', padding: 20 }}>
          <h3>Upload Medical Image</h3>
          <input type="file" onChange={handleFileChange} />
          {preview && (
            <img
              src={preview}
              alt="preview"
              style={{ width: '100%', marginTop: 10 }}
            />
          )}
        </div>
      </div>

      <button
        onClick={handleAnalyze}
        disabled={loading}
        style={{
          marginTop: 20,
          width: '100%',
          padding: 15,
          backgroundColor: '#4F46E5',
          color: 'white',
          border: 'none'
        }}
      >
        {loading ? "Processing..." : "Run Diagnosis"}
      </button>

      {result && (
        <div style={{ marginTop: 20 }}>
          <h2>{result.status}</h2>
          <p>Probability: {result.probability}%</p>
          <p>{result.message}</p>
        </div>
      )}
    </div>
  );
}

export default App;
