import React from 'react';
import './Table.css'; // Import the CSS file

const KnowledgeBase = ({ data }) => {
    if (!data || !data.results || !Array.isArray(data.results)) {
        return <p>No Knowledge Base data available.</p>;
    }

    const results = data.results;

    if (results.length === 0) {
        return <p>No Knowledge Base data available.</p>;
    }

    const headers = Object.keys(results[0].entity || {});

    return (
        <div>
            <table>
                <thead>
                    <tr>
                        {headers.map((header) => (
                            <th key={header}>{header}</th>
                        ))}
                    </tr>
                </thead>
                <tbody>
                    {results.map((item, index) => (
                        <tr key={index}>
                            {headers.map((header) => (
                                <td key={`${index}-${header}`}>{String(item.entity[header])}</td>
                            ))}
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
};

export default KnowledgeBase;