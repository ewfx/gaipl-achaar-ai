import React from 'react';
import Header from './components/Header';
import NavigationMenu from './components/NavigationMenu';
import IncidentDashboard from './components/IncidentDashboard';
import './DashboardScreen.css';

const DashboardScreen = () => {
  return (
    <div className="dashboard-screen">
      
      <div className="main-content">
       <IncidentDashboard />
      </div>
    </div>
  );
};

export default DashboardScreen;
