import React, { useState, useEffect } from 'react';
import { keyframes } from '@emotion/react';
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
import { useNavigate, useLocation } from 'react-router-dom';
import { 
  Logout, 
  Security,
  CheckCircle,
  MedicalServices,
  Psychology,
  Assessment,
  KeyboardArrowDown
} from '@mui/icons-material';
import AuthPanel from './AuthPanel';

const auroraDrift = keyframes`
  0% { transform: translate(-10%, -10%) scale(1); opacity: 0.35; }
  50% { transform: translate(10%, 5%) scale(1.1); opacity: 0.6; }
  100% { transform: translate(-10%, -10%) scale(1); opacity: 0.35; }
`;

const gridSlide = keyframes`
  0% { background-position: 0 0, 0 0; opacity: 0.28; }
  50% { background-position: -120px 160px, 0 0; opacity: 0.4; }
  100% { background-position: -240px 320px, 0 0; opacity: 0.28; }
`;

const LandingPage = () => {
  const [dataType, setDataType] = useState('MRI');
  const [userInfo, setUserInfo] = useState(null);
  const navigate = useNavigate();
  const location = useLocation();
  const sectionPadding = { xs: 3, sm: 6, md: 10 };
  const scrollToSection = (sectionId) => {
    const element = document.getElementById(sectionId);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  };

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

  useEffect(() => {
    if (location.state?.scrollTo) {
      const targetId = location.state.scrollTo;
      const element = document.getElementById(targetId);
      if (element) {
        setTimeout(() => {
          element.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 50);
      }
      navigate(location.pathname, { replace: true, state: {} });
    }
  }, [location, navigate]);

  const handleLogout = () => {
    localStorage.removeItem('userLoggedIn');
    localStorage.removeItem('userEmail');
    localStorage.removeItem('userName');
    setUserInfo(null);
    
    // Dispatch custom event to notify other components
    window.dispatchEvent(new Event('loginStateChanged'));
    
    navigate('/');
  };

  const handleProceed = () => {
    if (!userInfo) {
      scrollToSection('login');
    } else {
      navigate('/upload', { state: { dataType } });
    }
  };

  if (!userInfo) {
    // User not logged in - show hero plus scrollable story sections
    const heroFeatures = [
      { icon: <CheckCircle sx={{ color: 'success.main' }} />, label: 'FDA-Compliant Processing' },
      { icon: <Security sx={{ color: 'success.main' }} />, label: 'Secure Platform' },
      { icon: <Psychology sx={{ color: 'success.main' }} />, label: 'AI-Powered Analysis' },
      { icon: <Assessment sx={{ color: 'success.main' }} />, label: 'Clinical Integration' }
    ];

    const evaluationBarriers = [
      { title: 'Weeks Waiting in EMU', detail: 'Two weeks of sleep-deprived monitoring before surgery is even discussed.' },
      { title: 'High Cost Burden', detail: '$150K+ spent per patient, often before a decision is made.' },
      { title: 'Limited Access', detail: 'Too few Level 4 centers force families to travel and disrupt their lives.' },
      { title: 'Inconclusive Answers', detail: 'Even after invasive monitoring, many patients still lack clear SOZ answers.' }
    ];

    const solutionHighlights = [
      {
        title: 'Clinically Validated',
        detail: 'DeepXS0Z matched gold-standard localization in 352 patients (ages 3 months–62 years) with 91% precision and 89% accuracy.'
      },
      {
        title: 'Efficiency Boost',
        detail: 'Delivers a 7× reduction in neurosurgical review effort versus manual workflows.'
      },
      {
        title: 'Cost Advantage',
        detail: 'A resting-state fMRI scan replaces weeks of costly EEG/SEEG admissions.'
      }
    ];

    const impactAudiences = [
      {
        title: 'Patients',
        bullets: [
          'Faster answers with lower risk and burden',
          'Real hope for seizure freedom through early localization'
        ]
      },
      {
        title: 'Providers',
        bullets: [
          'Frees EMU beds and staff with streamlined evaluations',
          'Increases surgical throughput with clearer candidates'
        ]
      },
      {
        title: 'Payors',
        bullets: [
          'Saves tens of thousands per patient',
          'Replaces prolonged hospitalizations with a non-invasive pathway'
        ]
      }
    ];

    const workflowSteps = [
      { label: 'Capture resting-state MRI', detail: 'Patients complete a simple resting-state fMRI scan, with or without sedation.' },
      { label: 'Independent Component Analysis', detail: 'DeepXS0Z separates neural signals from noise and resting networks.' },
      { label: 'Deep Learning Sorting', detail: 'AI prioritizes 15–20 candidate ICs most likely to reflect seizure onset zones.' },
      { label: 'Expert Review & Report', detail: 'Clinicians validate, generate the SOZ map, and hand off for surgical planning.' }
    ];

    return (
      <Box
        sx={{
          position: 'relative',
          minHeight: '100vh',
          bgcolor: '#050505',
          color: '#e0e0e0',
          overflow: 'hidden'
        }}
      >
        <Box
          aria-hidden
          sx={{
            position: 'fixed',
            inset: 0,
            pointerEvents: 'none',
            background:
              'radial-gradient(circle at 15% 25%, rgba(0,255,255,0.2), transparent 45%), radial-gradient(circle at 80% 0%, rgba(0,128,255,0.15), transparent 40%), radial-gradient(circle at 70% 80%, rgba(0,255,170,0.15), transparent 45%)',
            filter: 'blur(60px)',
            animation: `${auroraDrift} 30s ease-in-out infinite`,
            zIndex: 0
          }}
        />
        <Box
          aria-hidden
          sx={{
            position: 'fixed',
            inset: 0,
            pointerEvents: 'none',
            backgroundImage:
              'linear-gradient(rgba(255,255,255,0.08) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.05) 1px, transparent 1px)',
            backgroundSize: '180px 180px',
            mixBlendMode: 'screen',
            animation: `${gridSlide} 35s linear infinite`,
            zIndex: 0
          }}
        />
        <Box sx={{ position: 'relative', zIndex: 1 }}>
          <Box
            component="section"
            id="home"
            sx={{
            minHeight: { xs: 'auto', md: 'calc(100vh - 70px)' },
            display: 'flex',
            alignItems: 'center',
            py: { xs: 6, md: 8 },
            position: 'relative',
            overflow: 'hidden',
            borderBottom: '1px solid rgba(255,255,255,0.05)'
          }}
        >
          {/* Animated background layer */}
          <Box
            sx={{
              position: 'absolute',
              inset: 0,
              background:
                'radial-gradient(circle at 20% 20%, rgba(0,255,255,0.18), transparent 55%)',
              animation: 'pulseGlow 12s ease-in-out infinite',
              opacity: 0.5,
              pointerEvents: 'none'
            }}
          />
          {/* Floating particles overlay */}
          <Box
            sx={{
              position: 'absolute',
              inset: 0,
              backgroundImage: 'url("data:image/svg+xml,%3Csvg width=\'160\' height=\'160\' xmlns=\'http://www.w3.org/2000/svg\'%3E%3Cg fill=\'%2300FFFF33\' fill-opacity=\'0.15\'%3E%3Ccircle cx=\'10\' cy=\'10\' r=\'1.5\'/%3E%3Ccircle cx=\'70\' cy=\'40\' r=\'1\'/%3E%3Ccircle cx=\'140\' cy=\'90\' r=\'1.25\'/%3E%3Ccircle cx=\'40\' cy=\'120\' r=\'1\'/%3E%3C/g%3E%3C/svg%3E")',
              backgroundSize: '160px 160px',
              animation: 'drift 30s linear infinite',
              opacity: 0.4,
              pointerEvents: 'none'
            }}
          />
          <Container
            maxWidth="xl"
            sx={{
              flexGrow: 1,
              px: sectionPadding
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
                  <Typography variant="h2" sx={{ fontWeight: 800, color: '#e0e0e0', letterSpacing: 0.5 }}>
                        EpiPrecision
                      </Typography>
                      <Typography variant="subtitle1" color="text.secondary" sx={{ fontWeight: 600 }}>
                        Advanced Medical Imaging Platform
                      </Typography>
                    </Box>
                  </Stack>

                  <Typography variant="h3" sx={{ mb: 3, fontWeight: 800, color: '#e0e0e0', lineHeight: 1.3 }}>
                    Precision Epilepsy Analysis Through AI-Powered Neuroimaging
                  </Typography>

                <Typography variant="h6" sx={{ mb: 4, color: 'text.secondary', lineHeight: 1.8, fontWeight: 400 }}>
                  DeepXS0Z decodes seizures with AI-powered maps, rewiring hope with surgical precision while keeping patient journeys non-invasive.
                </Typography>

                <Grid container spacing={2} sx={{ mb: 4 }}>
                  {heroFeatures.map((feature) => (
                    <Grid item xs={12} sm={6} key={feature.label}>
                      <Stack direction="row" spacing={1.5} alignItems="center">
                        {feature.icon}
                        <Typography variant="h6" sx={{ fontWeight: 600 }}>
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
                    p: { xs: 3, md: 4 },
                    borderRadius: 4,
                    background: '#111',
                    border: '1px solid rgba(255,255,255,0.08)',
                    width: '100%',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: 2.5
                  }}
                >
                  <Stack spacing={2} alignItems="center">
                    <Avatar
                      sx={{
                        width: 56,
                        height: 56,
                        bgcolor: 'rgba(0,255,255,0.12)',
                        color: '#00ffff',
                        border: '2px solid rgba(0,255,255,0.3)'
                      }}
                    >
                      <Psychology sx={{ fontSize: '1.6rem' }} />
                    </Avatar>
                    <Box sx={{ textAlign: 'center' }}>
                      <Typography variant="h6" sx={{ fontWeight: 700, color: '#e0e0e0' }}>
                        Secure Access
                      </Typography>
                      <Typography variant="body1" color="text.secondary">
                        Sign in to continue to the clinician workspace.
                      </Typography>
                    </Box>
                  </Stack>
                  <Box id="login" sx={{ width: '100%' }}>
                    <AuthPanel showTitle={false} />
                  </Box>
                </Paper>
              </Grid>
            </Grid>
          </Container>
          <Box
            role="button"
            tabIndex={0}
            onClick={() => scrollToSection('missed-opportunity')}
            onKeyDown={(event) => {
              if (event.key === 'Enter' || event.key === ' ') {
                event.preventDefault();
                scrollToSection('missed-opportunity');
              }
            }}
            sx={{
              position: 'absolute',
              bottom: 24,
              left: '50%',
              transform: 'translateX(-50%)',
              display: { xs: 'none', sm: 'flex' },
              flexDirection: 'column',
              alignItems: 'center',
              gap: 0.35,
              cursor: 'pointer',
              color: 'rgba(255,255,255,0.75)',
              textTransform: 'uppercase',
              fontSize: '0.65rem',
              opacity: 0.85,
              transition: 'opacity 0.2s ease',
              '&:hover': { opacity: 1 },
              '&:focus-visible': {
                outline: '2px solid #00ffff',
                outlineOffset: 4,
                opacity: 1
              }
            }}
          >
            <Typography variant="caption" sx={{ fontWeight: 600 }}>
              Scroll Down
            </Typography>
            <KeyboardArrowDown sx={{ fontSize: 24 }} />
          </Box>
        </Box>

          <Box component="section" id="missed-opportunity" sx={{ borderTop: '1px solid #111', borderBottom: '1px solid #111', py: { xs: 6, md: 8 } }}>
            <Container maxWidth="xl" sx={{ px: sectionPadding }}>
              <Grid container spacing={4} alignItems="stretch">
              <Grid item xs={12} md={5}>
                <Typography variant="overline" sx={{ color: '#00ffff', letterSpacing: 2 }}>
                  The Missed Opportunity
                </Typography>
                <Typography variant="h3" sx={{ fontWeight: 800, mt: 1, mb: 2, lineHeight: 1.3 }}>
                  Over 1M people live with drug-resistant epilepsy, yet fewer than 4,000 surgeries happen each year.
                </Typography>
                <Typography variant="h5" color="text.secondary" sx={{ fontWeight: 500 }}>
                  That means more than 996,000 people are waiting because pre-surgical evaluation remains the real barrier.
                </Typography>
              </Grid>
              <Grid item xs={12} md={7}>
                <Grid container spacing={2}>
                  {evaluationBarriers.map((item) => (
                    <Grid item xs={12} sm={6} key={item.title}>
                      <Paper
                        sx={{
                          height: '100%',
                          p: 2.5,
                          borderRadius: 3,
                          bgcolor: '#0f0f0f',
                          border: '1px solid rgba(255,255,255,0.06)'
                        }}
                      >
                        <Typography variant="h6" sx={{ fontWeight: 700, mb: 1 }}>
                          {item.title}
                        </Typography>
                        <Typography variant="body1" color="text.secondary">
                          {item.detail}
                        </Typography>
                      </Paper>
                    </Grid>
                  ))}
                </Grid>
              </Grid>
            </Grid>
          </Container>
        </Box>

          <Container component="section" id="about" maxWidth="xl" sx={{ px: sectionPadding, py: { xs: 6, md: 8 } }}>
          <Typography variant="overline" sx={{ color: '#00ffff', letterSpacing: 2, fontSize: '0.95rem' }}>
            A Smarter Path Forward
          </Typography>
          <Typography variant="h3" sx={{ fontWeight: 800, my: 2 }}>
            DeepXS0Z combines AI with resting-state fMRI to non-invasively localize the seizure onset zone.
          </Typography>
          <Grid container spacing={3}>
            {solutionHighlights.map((item) => (
              <Grid item xs={12} md={4} key={item.title}>
                <Paper
                  sx={{
                    height: '100%',
                    p: 3,
                    borderRadius: 3,
                    bgcolor: '#0f0f0f',
                    border: '1px solid rgba(255,255,255,0.08)'
                  }}
                >
                  <Typography variant="h6" sx={{ fontWeight: 700, mb: 1.5 }}>
                    {item.title}
                  </Typography>
                  <Typography variant="body1" color="text.secondary">
                    {item.detail}
                  </Typography>
                </Paper>
              </Grid>
            ))}
          </Grid>
        </Container>

          <Container component="section" maxWidth="xl" sx={{ px: sectionPadding, py: { xs: 6, md: 8 } }}>
            <Typography variant="overline" sx={{ color: '#00ffff', letterSpacing: 2, fontSize: '0.95rem' }}>
              How It Works
          </Typography>
          <Grid container spacing={4} sx={{ mt: 1 }}>
            <Grid item xs={12} md={6}>
              <Typography variant="h4" sx={{ fontWeight: 700, mb: 2, lineHeight: 1.3 }}>
                From scan to SOZ map with four streamlined steps.
              </Typography>
              <Typography variant="h6" color="text.secondary" sx={{ fontWeight: 400 }}>
                DeepXS0Z merges deep learning with expert review, so what once required weeks of invasive tests now begins with a widely available MRI.
              </Typography>
            </Grid>
            <Grid item xs={12} md={6}>
              <Stack spacing={3}>
                {workflowSteps.map((step, index) => (
                  <Box
                    key={step.label}
                    sx={{
                      display: 'flex',
                      gap: 2,
                      alignItems: 'flex-start'
                    }}
                  >
                    <Box
                      sx={{
                        width: 48,
                        height: 48,
                        borderRadius: '50%',
                        border: '1px solid rgba(0,255,255,0.4)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        fontWeight: 700
                      }}
                    >
                      {index + 1}
                    </Box>
                    <Box>
                      <Typography variant="h6" sx={{ fontWeight: 700 }}>
                        {step.label}
                      </Typography>
                      <Typography variant="body1" color="text.secondary">
                        {step.detail}
                      </Typography>
                    </Box>
                  </Box>
                ))}
              </Stack>
            </Grid>
            </Grid>
          </Container>

          <Box component="section" sx={{ bgcolor: '#070707', py: { xs: 6, md: 8 } }}>
            <Container maxWidth="xl" sx={{ px: sectionPadding }}>
              <Typography variant="overline" sx={{ color: '#00ffff', letterSpacing: 2, fontSize: '0.95rem' }}>
                Impact Across the Continuum
              </Typography>
              <Grid container spacing={3} sx={{ mt: 2 }}>
                {impactAudiences.map((audience) => (
                  <Grid item xs={12} md={4} key={audience.title}>
                    <Paper
                      sx={{
                        height: '100%',
                        p: 3,
                        borderRadius: 3,
                        bgcolor: '#101010',
                        border: '1px solid rgba(255,255,255,0.06)'
                      }}
                    >
                      <Typography variant="h5" sx={{ fontWeight: 700, mb: 2 }}>
                        {audience.title}
                      </Typography>
                      <Stack spacing={1}>
                        {audience.bullets.map((bullet) => (
                          <Typography key={bullet} variant="body1" color="text.secondary">
                            • {bullet}
                          </Typography>
                        ))}
                      </Stack>
                    </Paper>
                  </Grid>
                ))}
              </Grid>
            </Container>
          </Box>

          <Box component="section" id="contact" sx={{ bgcolor: '#070707', py: { xs: 6, md: 8 } }}>
            <Container maxWidth="xl" sx={{ px: sectionPadding }}>
              <Box sx={{ maxWidth: 720, mx: 'auto' }}>
              <Paper
                sx={{
                  p: { xs: 4, md: 5 },
                  borderRadius: 4,
                  bgcolor: '#101010',
                  border: '1px solid rgba(255,255,255,0.08)'
                }}
              >
              <Typography variant="overline" sx={{ color: '#00ffff', letterSpacing: 2, fontSize: '0.95rem' }}>
                Contact Us
              </Typography>
              <Typography variant="h4" sx={{ fontWeight: 800, mt: 1, mb: 2 }}>
                info@epiprecision.tech | www.epiprecision.tech
              </Typography>
              <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
                4539 N 22nd St, Phoenix, Arizona, 85016, US
              </Typography>
              <Button
                variant="contained"
                size="large"
                fullWidth
                onClick={() => (window.location.href = 'mailto:info@epiprecision.tech')}
                sx={{
                  bgcolor: '#00aaaa',
                  color: '#0a0a0a',
                  fontWeight: 700,
                  '&:hover': { bgcolor: '#00c2c2' }
                }}
              >
                Start a Conversation
              </Button>
              </Paper>
            </Box>
          </Container>
          </Box>
        </Box>
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

              <Typography variant="body1" color="text.secondary" sx={{ mb: 2, lineHeight: 1.5 }}>
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
