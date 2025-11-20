import React from 'react';
import { Box, Card, CardContent } from '@mui/material';
import { useNavigate } from 'react-router-dom';
import AuthPanel from './AuthPanel';

const LoginPage = () => {
  const navigate = useNavigate();

  return (
    <Box
      display="flex"
      justifyContent="center"
      alignItems="center"
      minHeight="90vh"
      sx={{ backgroundColor: '#0a0a0a', py: 4 }}
    >
      <Card sx={{ minWidth: 320, maxWidth: 500, width: '100%', mx: 2 }}>
        <CardContent sx={{ p: { xs: 3, sm: 4 } }}>
          <AuthPanel onAuthSuccess={() => navigate('/')} />
        </CardContent>
      </Card>
    </Box>
  );
};

export default LoginPage;
