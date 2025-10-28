import React, { useState, useEffect } from 'react';
import { 
  Card, 
  CardContent, 
  Typography, 
  Radio, 
  RadioGroup, 
  FormControlLabel, 
  Button, 
  Box, 
  Avatar, 
  Grid,
  Paper,
  Divider,
  Container,
  Stack
} from '@mui/material';
import { useNavigate } from 'react-router-dom';
import { 
  Logout, 
  Person, 
  LocalHospital,
  Security,
  CheckCircle,
  MedicalServices,
  Psychology,
  Assessment
} from '@mui/icons-material';

const LandingPage = () => {
  const [dataType, setDataType] = useState('MRI');
  const [userInfo, setUserInfo] = useState(null);
  const navigate = useNavigate();

  const checkLoginStatus = () => {
    const isLoggedIn = localStorage.getItem('userLoggedIn');
    const userEmail = localStorage.getItem('userEmail');
    const userName = localStorage.getItem('userName');
    
    if (isLoggedIn && userEmail) {
      setUserInfo({
        email: userEmail,
        name: userName || 'User'
      });
    } else {
      setUserInfo(null);
    }
  };

  useEffect(() => {
    checkLoginStatus();
    
    // Listen for storage changes (when user logs in/out from other components)
    window.addEventListener('storage', checkLoginStatus);
    
    // Listen for custom login state changes (for same-tab updates)
    window.addEventListener('loginStateChanged', checkLoginStatus);
    
    return () => {
      window.removeEventListener('storage', checkLoginStatus);
      window.removeEventListener('loginStateChanged', checkLoginStatus);
    };
  }, []);

  const handleLogout = () => {
    localStorage.removeItem('userLoggedIn');
    localStorage.removeItem('userEmail');
    localStorage.removeItem('userName');
    setUserInfo(null);
    
    // Dispatch custom event to notify other components
    window.dispatchEvent(new Event('loginStateChanged'));
    
    navigate('/login');
  };

  const handleProceed = () => {
    if (!userInfo) {
      navigate('/login');
    } else {
      navigate('/upload', { state: { dataType } });
    }
  };

  if (!userInfo) {
    // User not logged in - show refined hero layout
    const heroFeatures = [
      { icon: <CheckCircle sx={{ color: 'success.main' }} />, label: 'FDA-Compliant Processing' },
      { icon: <Security sx={{ color: 'success.main' }} />, label: 'Secure Platform' },
      { icon: <Psychology sx={{ color: 'success.main' }} />, label: 'AI-Powered Analysis' },
      { icon: <Assessment sx={{ color: 'success.main' }} />, label: 'Clinical Integration' },
    ];

    return (
      <Box
        sx={{
          minHeight: { xs: 'auto', md: 'calc(100vh - 70px)' },
          height: { md: 'calc(100vh - 70px)' },
          bgcolor: '#0a0a0a',
          display: 'flex',
          alignItems: 'center',
          overflow: 'hidden',
          py: { xs: 4, md: 0 }
        }}
      >
        <Container
          maxWidth="xl"
          sx={{
            flexGrow: 1,
            height: { md: '100%' },
            px: { xs: 3, sm: 6, md: 10 },
            display: 'flex',
            alignItems: 'center'
          }}
        >
          <Grid container spacing={{ xs: 4, md: 6 }} alignItems="stretch">
            {/* Left - Value Proposition */}
            <Grid item xs={12} md={8} sx={{ display: 'flex' }}>
              <Box sx={{ width: '100%', pr: { md: 4 }, display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                <Stack direction="row" spacing={2} alignItems="center" sx={{ mb: 4 }}>
                  <Box
                    sx={{
                      width: 56,
                      height: 56,
                      borderRadius: '50%',
                      bgcolor: 'rgba(0,255,255,0.12)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center'
                    }}
                  >
                    <Psychology sx={{ fontSize: '2rem', color: '#00ffff' }} />
                  </Box>
                  <Box>
                    <Typography variant="h3" sx={{ fontWeight: 800, color: '#e0e0e0', letterSpacing: 0.5 }}>
                      EpiPrecision
                    </Typography>
                    <Typography variant="subtitle1" color="text.secondary" sx={{ fontWeight: 600 }}>
                      Advanced Medical Imaging Platform
                    </Typography>
                  </Box>
                </Stack>

                <Typography variant="h4" sx={{ mb: 3, fontWeight: 800, color: '#e0e0e0', lineHeight: 1.3 }}>
                  Precision Epilepsy Analysis Through AI-Powered Neuroimaging
                </Typography>

                <Typography variant="body1" sx={{ mb: 4, color: 'text.secondary', lineHeight: 1.8 }}>
                  State-of-the-art independent component analysis to identify seizure onset zones with clinical-grade accuracy and reliability.
                </Typography>

                <Grid container spacing={2} sx={{ mb: 4 }}>
                  {heroFeatures.map((feature, index) => (
                    <Grid item xs={12} sm={6} key={index}>
                      <Stack direction="row" spacing={1.5} alignItems="center">
                        {feature.icon}
                        <Typography variant="body1" sx={{ fontWeight: 600 }}>
                          {feature.label}
                        </Typography>
                      </Stack>
                    </Grid>
                  ))}
                </Grid>

                <Divider sx={{ borderColor: '#222222', maxWidth: 420, my: 3 }} />
                <Typography variant="caption" color="text.secondary">
                  Trusted by clinicians and researchers for secure, compliant neuroimaging analysis.
                </Typography>
              </Box>
            </Grid>

            {/* Right - Login Card */}
            <Grid item xs={12} md={4} sx={{ display: 'flex' }}>
              <Paper
                elevation={0}
                sx={{
                  p: { xs: 4, md: 5 },
                  borderRadius: 4,
                  background: '#121212',
                  border: '1px solid rgba(255,255,255,0.08)',
                  boxShadow: '0 20px 45px rgba(0, 0, 0, 0.45)',
                  width: '100%',
                  height: '100%',
                  display: 'flex',
                  flexDirection: 'column'
                }}
              >
                <Stack spacing={3} alignItems="center" sx={{ flexGrow: 1 }}>
                  <Avatar
                    sx={{
                      width: 64,
                      height: 64,
                      bgcolor: 'rgba(0,255,255,0.12)',
                      color: '#e0e0e0',
                    }}
                  >
                    <LocalHospital sx={{ fontSize: '1.75rem' }} />
                  </Avatar>
                  <Box sx={{ textAlign: 'center' }}>
                    <Typography variant="h6" sx={{ fontWeight: 700, color: '#e0e0e0' }}>
                      Secure Access Portal
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Sign in to access advanced neuroimaging analysis tools.
                    </Typography>
                  </Box>
                  <Button
                    variant="outlined"
                    size="large"
                    fullWidth
                    onClick={() => navigate('/login')}
                    startIcon={<Person />}
                    sx={{
                      py: 1.6,
                      fontSize: '1.05rem',
                      fontWeight: 700,
                      borderRadius: 2,
                      borderWidth: 2,
                      borderColor: 'rgba(0,255,255,0.35)',
                      color: '#e0e0e0',
                      '&:hover': {
                        borderColor: '#00ffff',
                        boxShadow: '0 0 12px rgba(0,255,255,0.45)',
                      }
                    }}
                  >
                    Sign In / Create Account
                  </Button>
                  <Divider sx={{ width: '100%' }}>
                    <Typography variant="caption" color="text.secondary">
                      Secure & Compliant
                    </Typography>
                  </Divider>
                  <Typography variant="caption" color="text.secondary" sx={{ textAlign: 'center', lineHeight: 1.6 }}>
                    The platform complies with FDA guidelines and medical data protection standards. Your patient data is encrypted and secure.
                  </Typography>
                </Stack>
              </Paper>
            </Grid>
          </Grid>
        </Container>
      </Box>
    );
  }

  // User is logged in - show main dashboard interface
  return (
    <Box
      sx={{
        minHeight: 'calc(100vh - 70px)',
        bgcolor: '#0a0a0a',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        py: { xs: 4, md: 6 },
        boxSizing: 'border-box'
      }}
    >
      {/* User Welcome Section */}
      <Box sx={{ width: '100%', maxWidth: 1000, mb: 4 }}>
        <Card sx={{ background: '#1a1a1a', color: '#e0e0e0', borderRadius: 3, boxShadow: 4, border: '1px solid #333333' }}>
          <CardContent sx={{ p: 3 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 2 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 3 }}>
                <Avatar 
                  sx={{ 
                    width: 48, 
                    height: 48, 
                    bgcolor: '#2a2a2a',
                    fontSize: '1.2rem',
                    fontWeight: 600,
                    color: '#e0e0e0',
                    border: '2px solid #333333'
                  }}
                >
                  {userInfo.name.charAt(0).toUpperCase()}
                </Avatar>
                <Box>
                  <Typography variant="h5" sx={{ fontWeight: 700, mb: 0.5, color: '#e0e0e0' }}>
                    Welcome back, {userInfo.name}
                  </Typography>
                  <Typography variant="body1" sx={{ color: 'rgba(224, 224, 224, 0.95)', mb: 0.5 }}>
                    {userInfo.email}
                  </Typography>
                </Box>
              </Box>
              <Button
                variant="outlined"
                size="small"
                startIcon={<Logout />}
                onClick={handleLogout}
                sx={{ 
                  fontWeight: 600,
                  px: 2,
                  py: 0.5,
                  minWidth: 0,
                }}
              >
                Sign Out
              </Button>
            </Box>
          </CardContent>
        </Card>
      </Box>

      {/* Main Analysis Section - Centered, wide, and with whitespace */}
      <Box sx={{ width: '100%', maxWidth: 1000, mx: 'auto' }}>
        <Card sx={{ width: '100%', p: 3, boxShadow: 6, borderRadius: 4, bgcolor: '#1a1a1a' }}>
          <CardContent sx={{ p: { xs: 2, md: 4 } }}>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
              <MedicalServices sx={{ fontSize: '1.5rem', color: '#e0e0e0', mr: 1.5 }} />
              <Typography variant="h6" sx={{ fontWeight: 600 }}>
                Neuroimaging Analysis
              </Typography>
            </Box>

            <Typography variant="body2" color="text.secondary" sx={{ mb: 2, lineHeight: 1.5 }}>
              Select your data type to begin advanced independent component analysis 
              for precise seizure onset zone identification.
            </Typography>

            <Typography variant="subtitle1" sx={{ mb: 2, fontWeight: 600 }}>
              Select Data Type
            </Typography>
            
            <RadioGroup
              value={dataType}
              onChange={e => setDataType(e.target.value)}
              sx={{ mb: 2 }}
            >
              <Grid container spacing={2}>
                <Grid item xs={12}>
              <Paper 
                className={dataType === 'MRI' ? 'neon-selected' : ''}
                sx={{ 
                  p: 1.5, 
                  mb: 1.5, 
                  border: dataType === 'MRI' ? '2px solid #00ffff' : '1px solid #333333',
                  borderRadius: 2,
                  bgcolor: dataType === 'MRI' ? '#2a2a2a' : 'transparent',
                  color: dataType === 'MRI' ? '#00ffff' : '#e0e0e0',
                  boxShadow: dataType === 'MRI' ? '0 0 8px 2px #00ffff55' : 'none',
                  transition: 'border-color 0.2s, box-shadow 0.2s, color 0.2s',
                }}
              >
                <FormControlLabel 
                  value="MRI" 
                  control={<Radio />} 
                  label={
                    <Box>
                      <Typography variant="body1" sx={{ fontWeight: 600 }}>
                        MRI (Magnetic Resonance Imaging)
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        High-resolution structural and functional brain imaging
                      </Typography>
                    </Box>
                  }
                />
              </Paper>
                </Grid>
                <Grid item xs={12}>
              <Paper 
                className={dataType === 'EEG' ? 'neon-selected' : ''}
                sx={{ 
                  p: 1.5, 
                  mb: 1.5, 
                  border: dataType === 'EEG' ? '2px solid #00ffff' : '1px solid #333333',
                  borderRadius: 2,
                  bgcolor: dataType === 'EEG' ? '#2a2a2a' : 'transparent',
                  color: dataType === 'EEG' ? '#00ffff' : '#e0e0e0',
                  boxShadow: dataType === 'EEG' ? '0 0 8px 2px #00ffff55' : 'none',
                  transition: 'border-color 0.2s, box-shadow 0.2s, color 0.2s',
                }}
              >
                <FormControlLabel 
                  value="EEG" 
                  control={<Radio />} 
                  label={
                    <Box>
                      <Typography variant="body1" sx={{ fontWeight: 600 }}>
                        EEG (Electroencephalography)
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Electrical activity monitoring for seizure detection
                      </Typography>
                    </Box>
                  }
                />
              </Paper>
                </Grid>
                <Grid item xs={12}>
              <Paper 
                className={dataType === 'PET' ? 'neon-selected' : ''}
                sx={{ 
                  p: 1.5, 
                  mb: 1.5, 
                  border: dataType === 'PET' ? '2px solid #00ffff' : '1px solid #333333',
                  borderRadius: 2,
                  bgcolor: dataType === 'PET' ? '#2a2a2a' : 'transparent',
                  color: dataType === 'PET' ? '#00ffff' : '#e0e0e0',
                  boxShadow: dataType === 'PET' ? '0 0 8px 2px #00ffff55' : 'none',
                  transition: 'border-color 0.2s, box-shadow 0.2s, color 0.2s',
                }}
              >
                <FormControlLabel 
                  value="PET" 
                  control={<Radio />} 
                  label={
                    <Box>
                      <Typography variant="body1" sx={{ fontWeight: 600 }}>
                        PET (Positron Emission Tomography)
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Metabolic brain imaging for functional analysis
                      </Typography>
                    </Box>
                  }
                />
              </Paper>
                </Grid>
              </Grid>
            </RadioGroup>

            <Button
              variant="contained"
              size="medium"
              fullWidth
              onClick={handleProceed}
              sx={{ 
                py: 1.5,
                fontSize: '1rem',
                fontWeight: 600,
                borderRadius: 2,
                mt: 2,
                backgroundColor: '#2a2a2a',
                color: '#e0e0e0',
                '&:hover': { backgroundColor: '#333333' }
              }}
            >
              Begin {dataType} Analysis Workflow
            </Button>
          </CardContent>
        </Card>
      </Box>
    </Box>
  );
};

export default LandingPage; 
