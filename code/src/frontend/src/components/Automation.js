import React, { useState } from 'react';
import axios from 'axios';
import './Automation.css'; // Import the CSS file

const Automation = () => {
    const [cmdbId, setCmdbId] = useState('');
    const [apiResponse, setApiResponse] = useState('');
    const [loading, setLoading] = useState(false);

    const handleRunAutomation = async () => {
        setLoading(true);
        setApiResponse('');
        const normalizedCmdbId = cmdbId.trim().toUpperCase(); // Convert to uppercase
        console.log("Normalized CMDB ID:", normalizedCmdbId);
        if (!/^[A-Z0-9]+$/.test(normalizedCmdbId)) {
            setApiResponse("Error: Invalid CMDB ID. Only uppercase alphanumeric characters are allowed.");
            setLoading(false);
            return;
        }
        try {
            const response = await axios.get(`http://localhost:8001/get-details/${normalizedCmdbId}`);
            console.log("Raw API Response:", response); // Log the entire response
            console.log("Response Data:", response.data); // log response data
            setApiResponse(JSON.stringify(response.data, null, 2));
        } catch (error) {
            console.error("API Error:", error);
            if (error.response) {
                console.error("Response Data:", error.response.data);
                console.error("Response Status:", error.response.status);
                console.error("Response Headers:", error.response.headers);
            } else if (error.request) {
                console.error("Request:", error.request);
            }
            setApiResponse('Error: ${error.message}');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="automation-container">
            <h1>Run Automation</h1>
            <div className="input-area">
                <label htmlFor="cmdbId">Affect CI Number:</label>
                <input
                    type="text"
                    id="cmdbId"
                    value={cmdbId}
                    onChange={(e) => setCmdbId(e.target.value)}
                />
                <button onClick={handleRunAutomation} disabled={loading}>
                    {loading ? 'Running Automation...' : 'Run Automation'}
                </button>
            </div>
            {loading && <div className="progress-bar">Running...</div>}
            {apiResponse && (
                <div className="response-area">
                    <h2>API Response:</h2>
                    <textarea value={apiResponse} readOnly />
                </div>
            )}
        </div>
    );
};

export default Automation;