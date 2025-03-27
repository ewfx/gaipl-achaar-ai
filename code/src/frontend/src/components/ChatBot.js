// import React, { useState, useRef, useEffect } from 'react';
// import './ChatBot.css';

// const ChatBot = () => {
//     const [messages, setMessages] = useState([]);
//     const [input, setInput] = useState('');
//     const chatContainerRef = useRef(null);
//     const socketRef = useRef(null);
//     const [searchId, setSearchId] = useState(null);
//     const [progressDetails, setProgressDetails] = useState({});
//     const [searchCompleted, setSearchCompleted] = useState(false); // New state variable

//     const handleInputChange = (e) => {
//         setInput(e.target.value);
//     };

//     const handleSendMessage = () => {
//         if (input.trim() !== '') {
//             const newUserMessage = { text: input, sender: 'user' };
//             setMessages((prevMessages) => [...prevMessages, newUserMessage]);
//             setInput('');
//             sendSearchRequest(input);
//             setSearchCompleted(false); // Reset search completion status
//         }
//     };

//     const sendSearchRequest = (query) => {
//         fetch('http://localhost:8000/search', {
//             method: 'POST',
//             headers: { 'Content-Type': 'application/json' },
//             body: JSON.stringify({ query: query }),
//         })
//             .then((response) => response.json())
//             .then((data) => {
//                 setSearchId(data.searchId);
//                 connectWebSocket(data.searchId);
//             });
//     };

//     const connectWebSocket = (id) => {
//         socketRef.current = new WebSocket(`ws://localhost:8000/ws/search/${id}`);

//         socketRef.current.onopen = () => {
//             console.log('WebSocket connection established');
//         };

//         socketRef.current.onmessage = (event) => {
//             const data = JSON.parse(event.data);
//             console.log('Received update:', data);
//             updateProgress(data);
//             if (data.status === 'completed') {
//                 const resultsMessages = Object.entries(data.results).map(([source, result]) => ({
//                     text: `${source}: ${result}`,
//                     sender: 'ai',
//                 }));
//                 setMessages((prevMessages) => [...prevMessages, ...resultsMessages]);
//                 setSearchCompleted(true); // Set search completion status
//             }
//         };

//         socketRef.current.onclose = () => {
//             console.log('WebSocket connection closed');
//         };

//         socketRef.current.onerror = (error) => {
//             console.error('WebSocket error:', error);
//         };
//     };

//     const updateProgress = (data) => {
//         const newProgressDetails = { ...progressDetails };
//         data.details.forEach((detail) => {
//             newProgressDetails[detail.source] = detail;
//         });
//         setProgressDetails(newProgressDetails);
//     };

//     useEffect(() => {
//         if (chatContainerRef.current) {
//             chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
//         }
//     }, [messages]);

//     return (
//         <div className="gemini-chat-container">
//             <div className="gemini-chat-content">
//                 <div className="gemini-chat-messages" ref={chatContainerRef}>
//                     {!searchCompleted && Object.entries(progressDetails).map(([source, detail]) => { // Conditionally render progress
//                         if (detail.status === 'pending') return null;
//                         return (
//                             <div key={source} className="gemini-message ai">
//                                 {detail.status === 'completed' ? (
//                                     `searched ${source}`
//                                 ) : (
//                                     `searching ${source}...`
//                                 )}
//                             </div>
//                         );
//                     })}
//                     {messages.map((message, index) => (
//                         <div key={index} className={`gemini-message ${message.sender}`}>
//                             <p>{message.text}</p>
//                         </div>
//                     ))}
//                 </div>
//                 <div className="gemini-chat-input">
//                     <input
//                         type="text"
//                         value={input}
//                         onChange={handleInputChange}
//                         placeholder="Ask me anything..."
//                     />
//                     <button onClick={handleSendMessage}>Send</button>
//                 </div>
//             </div>
//         </div>
//     );
// };

// export default ChatBot;

import React, { useState, useRef, useEffect } from 'react';
import './ChatBot.css';

const ChatBot = () => {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const chatContainerRef = useRef(null);
    const socketRef = useRef(null);
    const [searchId, setSearchId] = useState(null);
    const [progressDetails, setProgressDetails] = useState({});
    const [searchCompleted, setSearchCompleted] = useState(false);
    const completedSources = useRef(new Set()); // Track completed sources

    const handleInputChange = (e) => {
        setInput(e.target.value);
    };

    const handleSendMessage = () => {
        if (input.trim() !== '') {
            const newUserMessage = { text: input, sender: 'user' };
            setMessages((prevMessages) => [...prevMessages, newUserMessage]);
            setInput('');
            sendSearchRequest(input);
            setSearchCompleted(false);
            completedSources.current.clear(); // Reset on new search
        }
    };

    const sendSearchRequest = (query) => {
        fetch('http://localhost:8000/search', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: query }),
        })
            .then((response) => response.json())
            .then((data) => {
                setSearchId(data.searchId);
                connectWebSocket(data.searchId);
            });
    };

    const connectWebSocket = (id) => {
        socketRef.current = new WebSocket(`ws://localhost:8000/ws/search/${id}`);

        socketRef.current.onopen = () => {
            console.log('WebSocket connection established');
        };

        socketRef.current.onmessage = (event) => {
            const data = JSON.parse(event.data);
            console.log('Received update:', data);
            updateProgress(data);

            const newMessages = [];
            let allCompleted = true; // Check if all sources are completed

            data.details.forEach(detail => {
                if (detail.status === 'completed' && !completedSources.current.has(detail.source)) {
                    newMessages.push({
                        text: `🔍 ${detail.source} search completed.`,
                        sender: 'ai',
                    });
                    completedSources.current.add(detail.source);
                } else if (detail.status !== 'completed') {
                    allCompleted = false;
                }
            });

            if (data.status === 'completed') {
                const resultsMessages = Object.entries(data.results).map(([source, result]) => ({
                    text: `${source}: ${result}`,
                    sender: 'ai',
                }));
                newMessages.push(...resultsMessages);
                setSearchCompleted(true);
            }

            setMessages((prevMessages) => {
                // Remove previous search messages if all completed
                const filteredMessages = allCompleted ? prevMessages.filter(msg => !msg.text.startsWith('🔍')) : prevMessages;
                return [...filteredMessages, ...newMessages];
            });
        };

        socketRef.current.onclose = () => {
            console.log('WebSocket connection closed');
        };

        socketRef.current.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
    };

    const updateProgress = (data) => {
        const newProgressDetails = { ...progressDetails };
        data.details.forEach((detail) => {
            newProgressDetails[detail.source] = detail;
        });
        setProgressDetails(newProgressDetails);
    };

    useEffect(() => {
        if (chatContainerRef.current) {
            chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
        }
    }, [messages]);

    return (
        <div className="gemini-chat-container">
            <div className="gemini-chat-content">
                <div className="gemini-chat-messages" ref={chatContainerRef}>
                    {messages.map((message, index) => (
                        <div key={index} className={`gemini-message ${message.sender}`}>
                            <p>{message.text}</p>
                        </div>
                    ))}
                </div>
                <div className="gemini-chat-input">
                    <input
                        type="text"
                        value={input}
                        onChange={handleInputChange}
                        placeholder="Ask me anything..."
                    />
                    <button onClick={handleSendMessage}>Send</button>
                </div>
            </div>
        </div>
    );
};

export default ChatBot;