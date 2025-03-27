import React, { useState, useEffect } from 'react';
import './IncidentDashboard.css';
import axios from 'axios';
import RelatedIncidents from '../components/RelatedIncidents';
import RCA from '../components/RCA';
import KnowledgeBase from '../components/KnowledgeBase';
import Recommendations from '../components/Recommendations';
import EnterpriseInfo from '../components/EnterpriseInfo';

const IncidentDashboard = () => {
    const [records, setRecords] = useState([]);
    const [filteredRecords, setFilteredRecords] = useState([]);
    const [sortConfig, setSortConfig] = useState(null);
    const [filters, setFilters] = useState({});
    const [popupData, setPopupData] = useState(null);
    const [activeTab, setActiveTab] = useState('Incident Details');
    const [tabData, setTabData] = useState({}); // Store data for each tab

    useEffect(() => {
        const fetchData = async () => {
            try {
                const response = await axios.get('/api/tickets', {
                    headers: {
                        'Accept': 'application/json',
                        'Content-Type': 'application/json',
                        'Cache-Control': "no-cache",
                    },
                });

                const data = response.data;
                setRecords(data);
                setFilteredRecords(data);
            } catch (error) {
                console.error('Error fetching data:', error);
            }
        };

        fetchData();
    }, []);

    useEffect(() => {
        let result = [...records];

        Object.keys(filters).forEach((key) => {
            if (filters[key]) {
                result = result.filter((item) =>
                    String(item[key]).toLowerCase().includes(filters[key].toLowerCase())
                );
            }
        });

        if (sortConfig !== null) {
            result.sort((a, b) => {
                if (a[sortConfig.key] < b[sortConfig.key]) {
                    return sortConfig.direction === 'ascending' ? -1 : 1;
                }
                if (a[sortConfig.key] > b[sortConfig.key]) {
                    return sortConfig.direction === 'ascending' ? 1 : -1;
                }
                return 0;
            });
        }

        setFilteredRecords(result);
    }, [records, filters, sortConfig]);

    const requestSort = (key) => {
        let direction = 'ascending';
        if (sortConfig && sortConfig.key === key && sortConfig.direction === 'ascending') {
            direction = 'descending';
        }
        setSortConfig({ key, direction });
    };

    const handleFilterChange = (key, value) => {
        setFilters({ ...filters, [key]: value });
    };

    const handleLinkClick = async (incidentId) => {
        try {
            console.log("incidentId:", incidentId);
            const response = await axios.get(`http://localhost:5001/api/tickets/${incidentId}`, {
                headers: {
                    'Accept': 'application/json',
                    'Content-Type': 'application/json',
                    'Cache-Control': 'no-cache',
                },
            });
            setPopupData(response.data);
            setActiveTab('Incident Details');
        } catch (error) {
            console.error('Error fetching incident details:', error);
        }
    };

    const closePopup = () => {
        setPopupData(null);
        setTabData({}); // Clear tab data when closing popup
    };

    const fetchTabData = async (tabName, description) => {
        if (tabData[tabName]) return;
    
        // let url = '';
        // let params = {};
    
        let url = '';
        let requestData = {};
        let headers = { 'Content-Type': 'application/json' };

        switch (tabName) {
            case 'RCA':
                url = 'http://127.0.0.1:5001/api/mcp_search_rca_document';
                //params = { description: description };
               requestData = { query: description, top_k: 5 };
                break;
            case 'Known Issues':
                url = 'http://127.0.0.1:5001/api/mcp_search_known_problem';
                requestData = { query: description, top_k: 5 };
                break;
            case 'Past INC':
                url = 'http://127.0.0.1:5001/api/mcp_search_incident';
                requestData = { query: description, top_k: 5 };
                break;
            case 'Knowledge base':
                url = 'http://127.0.0.1:5001/api/mcp_search_knowledge';
                requestData = { query: description, top_k: 5 };
                break;
            case 'Workaround':
                url = 'http://127.0.0.1:5001/api/mcp_search_work_around';
                requestData = { query: description, top_k: 5 };
                break;
            default:
                return;
        }
    
        try {
            // const response = await axios.get(url, { params: params });
            const response = await axios.post(url, requestData, { headers: headers });
            setTabData({ ...tabData, [tabName]: response.data });
        } catch (error) {
            console.error(`Error fetching ${tabName} data:`, error);
        }
    };

    const renderTable = () => {
        if (filteredRecords.length === 0) {
            return <p>No records found.</p>;
        }

        const headers = Object.keys(filteredRecords[0]);

        return (
            <table>
                <thead>
                    <tr>
                        {headers.map((header) => (
                            <th key={header}>
                                {header.replace(/_/g, ' ')}
                                <button onClick={() => requestSort(header)}>
                                    {sortConfig && sortConfig.key === header
                                        ? sortConfig.direction === 'ascending'
                                            ? '▲'
                                            : '▼'
                                        : '↕'}
                                </button>
                                <input
                                    type="text"
                                    placeholder={`Filter ${header.replace(/_/g, ' ')}`}
                                    onChange={(e) => handleFilterChange(header, e.target.value)}
                                />
                            </th>
                        ))}
                    </tr>
                </thead>
                <tbody>
                    {filteredRecords.map((record, index) => (
                        <tr key={record.id} className={index % 2 === 0 ? 'even' : 'odd'}>
                            {headers.map((header, headerIndex) => (
                                <td key={`${record.incident_id}-${header}`}>
                                    {headerIndex === 0 ? (
                                        <a href="#" onClick={(e) => { e.preventDefault(); handleLinkClick(record.id); }}>
                                            {record[header]}
                                        </a>
                                    ) : (
                                        record[header]
                                    )}
                                </td>
                            ))}
                        </tr>
                    ))}
                </tbody>
            </table>
        );
    };

    const renderPopupContent = () => {
        if (!popupData) return null;

        const renderTabContent = () => {
            if (!popupData) return null;
        
            switch (activeTab) {
                case 'Incident Details':
                    return (
                        <div className="row">
                            {Object.entries(popupData).map(([key, value]) => (
                                <div key={key} className="col">
                                    <h4>{key.replace(/_/g, ' ').charAt(0).toUpperCase() + key.replace(/_/g, ' ').slice(1)}:</h4>
                                    <p>{value !== null ? String(value) : 'N/A'}</p>
                                </div>
                            ))}
                        </div>
                    );
                case 'Past INC':
                    return <RelatedIncidents data={tabData[activeTab]} />;
                case 'RCA':
                    return <RCA data={tabData[activeTab]} />;
                case 'Knowledge base':
                    return <KnowledgeBase data={tabData[activeTab]} />;
                case 'Workaround':
                    return <Recommendations data={tabData[activeTab]} />;
                case 'Known Issues':
                    return <EnterpriseInfo data={tabData[activeTab]} />;
                default:
                    return null;
            }
        };

        return (
            <div className="popup">
                <div className="popup-content">
                    <div className="tab-buttons">
                        {['Incident Details', 'Past INC', 'RCA', 'Knowledge base', 'Workaround', 'Known Issues'].map((tab) => (
                           <button
                           key={tab}
                           className={activeTab === tab ? 'active' : ''}
                           onClick={() => {
                               setActiveTab(tab);
                               if (tab !== 'Incident Details') {
                                   fetchTabData(tab, popupData.description); // Pass description if available
                               }
                           }}
                       >
                           {tab}
                       </button>
                        ))}
                    </div>
                    {renderTabContent()}
                    <button onClick={closePopup}>Close</button>
                </div>
            </div>
        );
    };

    return (
        <div className="dashboard-container">
            <h2>Incident Dashboard</h2>
            {records.length > 0 ? renderTable() : <p>Loading...</p>}
            {renderPopupContent()}
        </div>
    );
};

export default IncidentDashboard;