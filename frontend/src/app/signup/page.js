"use client";

import React, { useState } from "react";
import {
  TextField,
  Button,
  Typography,
  Container,
  Box,
  Grid,
  Avatar,
  CssBaseline,
  MenuItem,
  Select,
  InputLabel,
  FormControl,
} from "@mui/material";
import LockOutlinedIcon from "@mui/icons-material/LockOutlined";
import { useRouter } from "next/navigation";
import axios from 'axios'; // Import axios

const SignUpPage = () => {
  const router = useRouter();
  const [formData, setFormData] = useState({
    username: "",
    email: "",
    password: "",
    age: "",
    ethnicity: "",
    gender: "",
    education: "",
    employment: "",
  });

  const handleChange = (event) => {
    const { name, value } = event.target;
    setFormData({ ...formData, [name]: value });
  };

  // Function to handle navigation to the login page
  const handleLoginRedirect = () => {
    router.push("/login"); // Navigate to the login page
  };

  // Function to handle form submission
  const handleSubmit = async (event) => {
    event.preventDefault();
    try {
      const response = await axios.post("http://localhost:5000/signup", {
        username: formData.username,
        password: formData.password,
        age: formData.age,
        ethnicity: formData.ethnicity,
        gender: formData.gender,
        education: formData.education,
        employment: formData.employment,
      });

      // Assuming your Flask backend returns a message in the response
      console.log(response.data.message);
      const user_id = response.data.user_id;
      console.log(user_id);
      router.push(`/chat/${user_id}`); // Redirect after successful signup
    } catch (error) {
      console.error("Error during signup:", error.response?.data.message || error.message);
    }
  };

  return (
    <Container component="main" maxWidth="xs">
      <CssBaseline />
      <Box
        sx={{
          marginTop: 8,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
        }}
      >
        <Avatar sx={{ m: 1, bgcolor: "primary.main" }}>
          <LockOutlinedIcon />
        </Avatar>
        <Typography component="h1" variant="h5">
          Sign Up
        </Typography>
        <Box component="form" noValidate sx={{ mt: 3 }} onSubmit={handleSubmit}>
          <Grid container spacing={2}>
            <Grid item xs={12}>
              <TextField
                autoComplete="username"
                name="username"
                required
                fullWidth
                id="username"
                label="Username"
                autoFocus
                value={formData.username}
                onChange={handleChange}
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                required
                fullWidth
                id="email"
                label="Email Address"
                name="email"
                autoComplete="email"
                value={formData.email}
                onChange={handleChange}
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                required
                fullWidth
                name="password"
                label="Password"
                type="password"
                id="password"
                autoComplete="new-password"
                value={formData.password}
                onChange={handleChange}
              />
            </Grid>

            <Grid item xs={12}>
              <Typography variant="h6" gutterBottom>
                Demographic survey:
              </Typography>
            </Grid>

            <Grid item xs={12}>
              <TextField
                fullWidth
                name="age"
                label="Age"
                type="number"
                value={formData.age}
                onChange={handleChange}
              />
            </Grid>

            <Grid item xs={12}>
              <FormControl fullWidth>
                <InputLabel id="ethnicity-label">Ethnicity</InputLabel>
                <Select
                  labelId="ethnicity-label"
                  name="ethnicity"
                  label="Ethnicity"
                  value={formData.ethnicity}
                  onChange={handleChange}
                >
                  <MenuItem value="asian">Asian</MenuItem>
                  <MenuItem value="black">Black or African American</MenuItem>
                  <MenuItem value="hispanic">Hispanic or Latino</MenuItem>
                  <MenuItem value="white">White</MenuItem>
                  <MenuItem value="native-american">Native American</MenuItem>
                  <MenuItem value="pacific-islander">Pacific Islander</MenuItem>
                  <MenuItem value="middle-eastern">Middle Eastern</MenuItem>
                  <MenuItem value="south-asian">South Asian</MenuItem>
                  <MenuItem value="east-asian">East Asian</MenuItem>
                  <MenuItem value="southeast-asian">Southeast Asian</MenuItem>
                  <MenuItem value="latino">Latino</MenuItem>
                </Select>
              </FormControl>
            </Grid>

            <Grid item xs={12}>
              <FormControl fullWidth>
                <InputLabel id="gender-label">Gender</InputLabel>
                <Select
                  labelId="gender-label"
                  name="gender"
                  label="Gender"
                  value={formData.gender}
                  onChange={handleChange}
                >
                  <MenuItem value="">
                    <em>None</em>
                  </MenuItem>
                  <MenuItem value="male">Male</MenuItem>
                  <MenuItem value="female">Female</MenuItem>
                  <MenuItem value="non-binary">Non-binary</MenuItem>
                  <MenuItem value="prefer-not-to-say">
                    Prefer not to say
                  </MenuItem>
                </Select>
              </FormControl>
            </Grid>

            <Grid item xs={12}>
              <FormControl fullWidth>
                <InputLabel id="education-label">Education</InputLabel>
                <Select
                  labelId="education-label"
                  name="education"
                  label="Education"
                  value={formData.education}
                  onChange={handleChange}
                >
                  <MenuItem value="">
                    <em>None</em>
                  </MenuItem>
                  <MenuItem value="high-school">High School</MenuItem>
                  <MenuItem value="bachelor">Bachelor's Degree</MenuItem>
                  <MenuItem value="master">Master's Degree</MenuItem>
                  <MenuItem value="phd">PhD</MenuItem>
                </Select>
              </FormControl>
            </Grid>

            <Grid item xs={12}>
              <FormControl fullWidth>
                <InputLabel id="employment-label">Employment</InputLabel>
                <Select
                  labelId="employment-label"
                  name="employment"
                  label="Employment"
                  value={formData.employment}
                  onChange={handleChange}
                >
                  <MenuItem value="">
                    <em>None</em>
                  </MenuItem>
                  <MenuItem value="employed">Employed</MenuItem>
                  <MenuItem value="student">Student</MenuItem>
                  <MenuItem value="unemployed">Unemployed</MenuItem>
                  <MenuItem value="self-employed">Self-employed</MenuItem>
                </Select>
              </FormControl>
            </Grid>

            <Grid item xs={12}>
              <Button
                type="submit"
                fullWidth
                variant="contained"
                sx={{ mt: 3, mb: 2 }}
              >
                Sign Up
              </Button>

              <Button
                fullWidth
                variant="outlined"
                sx={{ mt: 2 }}
                onClick={handleLoginRedirect}
              >
                Already Signed Up? Log In.
              </Button>
            </Grid>
          </Grid>
        </Box>
      </Box>
    </Container>
  );
};

export default SignUpPage;
