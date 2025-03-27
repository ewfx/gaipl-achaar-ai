import React from 'react';
import './Header.css';

const Header = ({ username }) => (
  <header className="header">
    <div className="logo">
      <h1>Integrated Platform Support Portal</h1>
    </div>
    <div className="user-info">
      <span>{username}</span>
      <h3 className="team-info">-Powered by AcharAI</h3>
      <button>Logout</button>
    </div>
  </header>
);

export default Header;
