import * as React from 'react';
import Box from '@mui/material/Box';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';

export default function AIOutput({ message }) { 
  return (
    <Box 
      sx={{ 
        display: 'flex',          
        justifyContent: 'flex-end', 
        minWidth: 275, 
        mb: 2, 
        width: '100%'            
      }}
    > 
      <Box sx={{ width: '75%' }}> 
        <Card 
          variant="outlined" 
          sx={{ 
            mt: 1, 
            mb: 1,
            mr: 2, 
            borderRadius: 2, 
            width: '100%' 
          }}
        > 
          <CardContent>
            <Typography variant="body2">
              {message.text}
            </Typography>
          </CardContent>
        </Card>
      </Box>
    </Box>
  );
}
