import './App.css';
import React, { useState } from "react";
import Header from "./components/Header";
import IncidentDashboard from './DashboardScreen';
import AIChatbot from './components/ChatBot';
import ContextualRecommendations from './components/Recommendations';
import EnterpriseInfo from './components/EnterpriseInfo';
import Automation from './components/Automation';
import NewBot from './components/NewBot';

const App = () => {
    const [selectedScreen, setSelectedScreen] = useState("dashboard");
    const [isChatbotOpen, setIsChatbotOpen] = useState(false); // State for chatbot visibility

    const handleNavigation = (screen) => {
        setSelectedScreen(screen);
    };
    const toggleChatbot = () => {
        setIsChatbotOpen(!isChatbotOpen);
    };
    let contentToRender;

    switch (selectedScreen) {
        case "dashboard":
            contentToRender = <IncidentDashboard />;
            break;
        //case "incidentDetails":
         //   contentToRender = <IncidentDetails />;
         //   break;
        case "chat":
            contentToRender = <AIChatbot />;
            break
        case "recommendations":
            contentToRender = <ContextualRecommendations />;
            break;
        case "enterpriseInfo":
            contentToRender = <EnterpriseInfo />;
            break;
        case "automation":
            contentToRender = <Automation />;
            break;
        case "NewChat":
            contentToRender = <NewBot />;
            break;
        default:
            contentToRender = <IncidentDashboard />;
    }

    return (
        <div>
            <Header />
            <nav className="tab-nav"> {/* Add a class for styling */}
                <ul className="tab-list"> {/* Add a class for styling */}
                    <li className={`tab-item ${selectedScreen === "dashboard" ? "active" : ""}`} onClick={() => handleNavigation("dashboard")}>Incidents</li>
                    <li className={`tab-item ${selectedScreen === "automation" ? "active" : ""}`} onClick={() => handleNavigation("automation")}>Automation</li>
                    {/* <li className={`tab-item ${selectedScreen === "knowledgeBase" ? "active" : ""}`} onClick={() => handleNavigation("knowledgeBase")}>Knowledge Base</li> */}
                    {/* <li className={`tab-item ${selectedScreen === "chat" ? "active" : ""}`} onClick={() => handleNavigation("chat")}>Chat</li> */}
                    <li className={`tab-item ${selectedScreen === "NewChat" ? "active" : ""}`} onClick={() => handleNavigation("NewChat")}>Chat</li>
                </ul>
            </nav>
            {contentToRender}
            {/* Floating Chatbot Icon */}
            <div className="floating-chatbot-icon" onClick={toggleChatbot}>
                {isChatbotOpen ? '✕' : '💬'} {/* Change icon/text based on chat state */}
            </div>

            {/* Conditional Rendering of Chatbot Component */}
            {isChatbotOpen && (
                <div className="chatbot-container">
                    <AIChatbot />
                </div>
            )}
        </div>
    );
};

export default App;