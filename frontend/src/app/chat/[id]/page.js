"use client";
import React, { useState } from 'react';
import { useParams } from 'next/navigation'; // For accessing dynamic route parameters
import Box from '@mui/material/Box';
import TextField from '@mui/material/TextField';
import ArrowForwardIcon from '@mui/icons-material/ArrowForward';
import InputAdornment from '@mui/material/InputAdornment';
import Button from '@mui/material/Button'; // Import Material UI Button
import Modal from '@mui/material/Modal'; // Import Material UI Modal
import { Typography } from '@mui/material';
import Bar from '../bar';
import UserInput from '../userInput';
import AIOutput from '../aiOutput';
import axios from 'axios'; // Import axios for making API requests
import Rating from '@mui/material/Rating';
import Stack from '@mui/material/Stack';

export default function ChatPage() {
  const { id } = useParams(); // Get the dynamic id from the route
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([]);
  const [openModal, setOpenModal] = useState(false); // State for modal open/close
  const [rating, setRating] = useState(0); // State for storing the rating

  const handleRatingSubmit = async (newValue) => {
    try {
      await axios.post('http://localhost:5000/rate', {
        user_id: id,
        rating: newValue,
      });
      console.log("Rating saved successfully!");
    } catch (error) {
      console.error("Error saving rating:", error);
    }
  };

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
        user_id: id
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

  // Open the rating modal
  const handleEndConversation = () => {
    setMessages([]);
  };

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
      <Bar />
      <Box sx={{ flex: 1, overflowY: 'auto', padding: 2 }}>
        {messages.map((message, index) => (
          <Box key={index} sx={{ margin: '8px 0' }}>
            {message.type === 'user' ? (
              <Stack spacing={1}>
                <UserInput message={message} />
              </Stack>
            ) : (
              <Stack spacing={1}>
                <AIOutput message={message} />
                {/* Align the Rating component to the right */}
                <Box sx={{ display: 'flex', justifyContent: 'flex-end' }}>
                  <Rating 
                  name="size-medium" 
                  defaultValue={0}
                  onChange={(event, newValue) => {
                    setRating(newValue); 
                    handleRatingSubmit(newValue); 
                  }} 
                  />
                </Box>
              </Stack>
            )}
          </Box>
        ))}
      </Box>

      {/* Conversation Input and End Conversation Button */}
      <Box
        component="form"
        noValidate
        autoComplete="off"
        onSubmit={handleInput}
        sx={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          padding: 2,
        }}
      >
        <TextField
          id="outlined-basic"
          label="Message Chat"
          variant="outlined"
          sx={{ width: '40%', marginRight: 2 }}
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
        <Button
          variant="outlined"
          onClick={handleEndConversation}
          sx={{ height: '56px' }} // Align with the height of the text field
        >
          End Conversation
        </Button>
      </Box>
    </Box>
  );
}
