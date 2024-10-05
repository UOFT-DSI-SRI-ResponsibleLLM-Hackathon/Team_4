import * as React from 'react';
import Box from '@mui/material/Box';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';

export default function UserInput({ message }) { 
  return (
    <Box 
      sx={{ minWidth: 275, mb: 2, width: '75%' }}
    > 
      <Card 
        variant="outlined" 
        sx={{ 
          mt: 1, 
          ml: 1,
          borderRadius: 2, 
          width: '100%'
        }}> 
        <CardContent>
          <Typography variant="body2">
            {message.text}
          </Typography>
        </CardContent>
      </Card>
    </Box>
  );
}
