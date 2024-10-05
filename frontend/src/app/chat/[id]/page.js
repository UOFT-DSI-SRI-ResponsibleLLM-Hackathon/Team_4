"use client";
import React, { useState } from 'react';
import { useParams } from 'next/navigation'; // For accessing dynamic route parameters
import Box from '@mui/material/Box';
import TextField from '@mui/material/TextField';
import ArrowForwardIcon from '@mui/icons-material/ArrowForward';
import InputAdornment from '@mui/material/InputAdornment';
import Bar from '../bar';
import UserInput from '../userInput';
import AIOutput from '../aiOutput';
import axios from 'axios'; // Import axios for making API requests

export default function ChatPage() {
  const { id } = useParams(); // Get the dynamic id from the route
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([]);

  const handleType = (event) => {
    const userInput = event.target.value;
    setInput(userInput);
  };

  const handleInput = async (event) => {
    event.preventDefault();

    if (!input.trim()) return;

    const userMessage = { text: input, type: 'user' };
    setMessages((prevMessages) => [...prevMessages, userMessage]);

    try {
      // Make a POST request to the Flask backend
      const response = await axios.post('http://localhost:5000/chat', {
        prompt: input,
      });

      const aiResponse = response.data.output; // Assuming your backend returns an output field
      console.log(aiResponse);
      const aiMessage = { text: aiResponse, type: 'ai' };

      setMessages((prevMessages) => [...prevMessages, aiMessage]);
    } catch (error) {
      console.error("Error fetching AI response:", error);
      const errorMessage = { text: "Error fetching response from AI.", type: 'ai' };
      setMessages((prevMessages) => [...prevMessages, errorMessage]);
    }

    setInput('');
  };

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
      <Bar />
      <Box sx={{ flex: 1, overflowY: 'auto', padding: 2 }}>
        {messages.map((message, index) => (
          <Box key={index} sx={{ margin: '8px 0' }}>
            {message.type === 'user' ? (
              <UserInput message={message} />
            ) : (
              <AIOutput message={message} />
            )}
          </Box>
        ))}
      </Box>
      <Box
        component="form"
        noValidate
        autoComplete="off"
        onSubmit={handleInput}
        sx={{
          display: 'flex',
          justifyContent: 'center',
          padding: 2,
        }}
      >
        <TextField
          id="outlined-basic"
          label={`Message Chat`}
          variant="outlined"
          sx={{ width: '50%' }}
          value={input}
          onChange={handleType}
          InputProps={{
            endAdornment: (
              <InputAdornment position="end">
                <ArrowForwardIcon
                  onClick={handleInput}
                  sx={{ cursor: 'pointer' }}
                />
              </InputAdornment>
            ),
          }}
        />
      </Box>
    </Box>
  );
}
