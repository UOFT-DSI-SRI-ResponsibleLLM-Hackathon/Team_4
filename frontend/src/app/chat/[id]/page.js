"use client";
import React, { useState } from 'react';
import { useParams } from 'next/navigation'; // For accessing dynamic route parameters
import Box from '@mui/material/Box';
import TextField from '@mui/material/TextField';
import ArrowForwardIcon from '@mui/icons-material/ArrowForward';
import InputAdornment from '@mui/material/InputAdornment';
import Button from '@mui/material/Button'; // Import Material UI Button
import Modal from '@mui/material/Modal'; // Import Material UI Modal
import Rating from '@mui/material/Rating'; // Import Material UI Rating
import { Typography } from '@mui/material';
import Bar from '../bar';
import UserInput from '../userInput';
import AIOutput from '../aiOutput';
import axios from 'axios'; // Import axios for making API requests

export default function ChatPage() {
  const { id } = useParams(); // Get the dynamic id from the route
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([]);
  const [openModal, setOpenModal] = useState(false); // State for modal open/close
  const [rating, setRating] = useState(0); // State for storing the rating

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
    setOpenModal(true);
  };

  // Close the rating modal
  const handleCloseModal = () => {
    setOpenModal(false);
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

      {/* Modal for Rating the Conversation */}
      <Modal
    open={openModal}
    onClose={handleCloseModal}
>
    <Box
        sx={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            width: 400,
            bgcolor: 'background.paper',
            boxShadow: 24,
            p: 4,
            borderRadius: 2,
            display: 'flex', // Use flexbox for layout
            flexDirection: 'column', // Stack children vertically
            justifyContent: 'space-between', // Space between items
            height: '200px', // Set a height for the modal
        }}
    >
        <Typography id="modal-modal-title" variant="h6" component="h2">
            We appreciate your feedback.
        </Typography>
        
        {/* Centered Box for Rating */}
        <Box 
            sx={{
                display: 'flex', // Use flexbox for centering
                justifyContent: 'center', // Center horizontally
                marginY: 2, // Add vertical margin for spacing
            }}
        >
            <Rating
                name="conversation-rating"
                value={rating}
                onChange={(event, newValue) => {
                    setRating(newValue);
                }}
            />
        </Box>

        {/* Empty Box to take up space and push button to bottom */}
        <Box sx={{ flexGrow: 1 }} />
        <Button
            onClick={handleCloseModal}
            sx={{ mt: 2, alignSelf: 'flex-end' }} // Align button to the right
            variant="outlined"
        >
            Submit
        </Button>
    </Box>
</Modal>

    </Box>
  );
}
