// import React, { useState, useRef, useEffect } from 'react';
// import './NewBot.css';

// const NewBot = () => {
//     const [messages, setMessages] = useState([]);
//     const [input, setInput] = useState('');
//     const chatContainerRef = useRef(null);
//     const ws = useRef(null);

//     useEffect(() => {
//         ws.current = new WebSocket("ws://localhost:5002/ws");

//         ws.current.onmessage = (event) => {
//             const msg = JSON.parse(event.data);
//             if (msg.type === "tool_event") {
//                 setMessages((prevMessages) => [...prevMessages, { text: `🛠 Tool Event: ${JSON.stringify(msg.data)}`, sender: 'ai' }]);
//             } else if (msg.type === "assistant_message") {
//                 setMessages((prevMessages) => [...prevMessages, { text: `🤖 Assistant: ${msg.data}`, sender: 'ai' }]);
//             }
//         };

//         ws.current.onclose = () => {
//             console.log('WebSocket connection closed');
//         };

//         ws.current.onerror = (error) => {
//             console.error('WebSocket error:', error);
//         };

//         return () => {
//             if (ws.current) {
//                 ws.current.close();
//             }
//         };
//     }, []);

//     const handleInputChange = (e) => {
//         setInput(e.target.value);
//     };

//     const handleSendMessage = async () => {
//         if (input.trim() !== '') {
//             setMessages((prevMessages) => [...prevMessages, { text: `🧑 You: ${input}`, sender: 'user' }]);
//             setInput('');

//             try {
//                 const response = await fetch("http://localhost:5002/mcp_chat", {
//                     method: "POST",
//                     headers: { "Content-Type": "application/json" },
//                     body: JSON.stringify({ user_input: input })
//                 });

//                 if (!response.ok) {
//                     throw new Error(`HTTP error! status: ${response.status}`);
//                 }

//                 const data = await response.json();
//                 console.log(data);
//             } catch (error) {
//                 console.error("Failed to send message:", error);
//                 setMessages((prevMessages) => [...prevMessages, { text: `Error sending message: ${error.message}`, sender: 'ai' }]);
//             }
//         }
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
//                         placeholder="Type your message"
//                     />
//                     <button onClick={handleSendMessage}>Send</button>
//                 </div>
//             </div>
//         </div>
//     );
// };

// export default NewBot;

import React, { useState, useRef, useEffect } from 'react';
import './NewBot.css';

const NewBot = () => {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const chatContainerRef = useRef(null);
    const ws = useRef(null);

    useEffect(() => {
        ws.current = new WebSocket("ws://localhost:5002/ws");

        ws.current.onmessage = (event) => {
            const msg = JSON.parse(event.data);
            if (msg.type === "tool_event") {
                setMessages((prevMessages) => [...prevMessages, { text: `🛠 Tool Event: ${JSON.stringify(msg.data)}`, sender: 'ai', isHtml: false }]);
            } else if (msg.type === "assistant_message") {
                const isHtml = /<\/?[a-z][\s\S]*>/i.test(msg.data);  // improved HTML detection
                setMessages((prevMessages) => [...prevMessages, { text: msg.data, sender: 'ai', isHtml }]);
            }
            
        };

        ws.current.onclose = () => {
            console.log('WebSocket connection closed');
        };

        ws.current.onerror = (error) => {
            console.error('WebSocket error:', error);
        };

        return () => {
            if (ws.current) {
                ws.current.close();
            }
        };
    }, []);

    const handleInputChange = (e) => {
        setInput(e.target.value);
    };

    const handleSendMessage = async () => {
        if (input.trim() !== '') {
            setMessages((prevMessages) => [...prevMessages, { text: `🧑 You: ${input}`, sender: 'user', isHtml: false }]);
            setInput('');

            try {
                const response = await fetch("http://localhost:5002/mcp_chat", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ user_input: input })
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const data = await response.json();
                console.log(data);
            } catch (error) {
                console.error("Failed to send message:", error);
                setMessages((prevMessages) => [...prevMessages, { text: `Error sending message: ${error.message}`, sender: 'ai', isHtml: false }]);
            }
        }
    };

    useEffect(() => {
        if (chatContainerRef.current) {
            chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
        }
    }, [messages]);

    const renderMessage = (message) => {
        if (message.sender === 'ai' && message.isHtml) {
            return <div dangerouslySetInnerHTML={{ __html: message.text }} />;
        } else {
            return <p>{message.text}</p>;
        }
    };

    return (
        <div className="gemini-chat-container">
            <div className="gemini-chat-content">
                <div className="gemini-chat-messages" ref={chatContainerRef}>
                    {messages.map((message, index) => (
                        <div key={index} className={`gemini-message ${message.sender}`}>
                            {renderMessage(message)}
                        </div>
                    ))}
                </div>
                <div className="gemini-chat-input">
                    <input
                        type="text"
                        value={input}
                        onChange={handleInputChange}
                        placeholder="Type your message"
                    />
                    <button onClick={handleSendMessage}>Send</button>
                </div>
            </div>
        </div>
    );
};

export default NewBot;