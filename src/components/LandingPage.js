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

  const dataTypeOptions = [
    {
      value: 'MRI',
      title: 'MRI (Magnetic Resonance Imaging)',
      description: 'High-resolution structural and functional brain imaging',
      helper: 'Best for structural localization',
      icon: MedicalServices
    },
    {
      value: 'EEG',
      title: 'EEG (Electroencephalography)',
      description: 'Electrical activity monitoring for seizure detection',
      helper: 'Captures neural oscillations',
      icon: Psychology
    },
    {
      value: 'PET',
      title: 'PET (Positron Emission Tomography)',
      description: 'Metabolic brain imaging for functional analysis',
      helper: 'Highlights metabolic asymmetries',
      icon: Assessment
    }
  ];

  const getModalityInsights = (type) => {
    switch (type) {
      case 'EEG':
        return [
          'Ideal for acute monitoring of seizure onset patterns.',
          'Upload ICA time-course exports and IC overlays for AI triage.',
          'Recommended bundle: 1–2 GB CSV/MAT datasets.'
        ];
      case 'PET':
        return [
          'Useful when MRI is inconclusive or metabolic correlation is required.',
          'Bundle NIfTI, DICOM, or fused PET/MRI exports for best alignment.',
          'Typical study: 50–150 slices; ensure consistent voxel sizes.'
        ];
      default:
        return [
          'Preferred modality for resting-state SOZ localization.',
          'Upload 100–200 IC threshold images plus supporting report files.',
          'Pair with Workspace-<case>V4.mat for reproducible analysis.'
        ];
    }
  };

  const selectedOption = dataTypeOptions.find((option) => option.value === dataType) || dataTypeOptions[0];
  const highlightCards = [
    {
      label: 'Active Modality',
      value: selectedOption?.value || dataType,
      helper: selectedOption?.helper
    },
    {
      label: 'Workflow Stage',
      value: 'Data Selection',
      helper: 'Choose a dataset to unlock upload + processing'
    },
    {
      label: 'Session Status',
      value: 'Clinician Secure',
      helper: 'Encrypted workspace active'
    }
  ];

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
        position: 'relative',
        minHeight: 'calc(100vh - 70px)',
        width: '100%',
        overflow: 'hidden',
        bgcolor: '#040404',
        py: { xs: 4, md: 6 },
        px: { xs: 2, md: 4 }
      }}
    >
      <Box
        aria-hidden
        sx={{
          position: 'absolute',
          inset: 0,
          background: 'radial-gradient(circle at 20% 20%, rgba(0,255,255,0.25), transparent 45%), radial-gradient(circle at 80% 0%, rgba(0,128,255,0.18), transparent 40%)',
          filter: 'blur(90px)',
          opacity: 0.8
        }}
      />
      <Box
        aria-hidden
        sx={{
          position: 'absolute',
          inset: 0,
          backgroundImage: 'linear-gradient(120deg, rgba(255,255,255,0.05) 0%, transparent 40%, transparent 60%, rgba(255,255,255,0.05) 100%)',
          opacity: 0.2
        }}
      />

      <Box sx={{ position: 'relative', zIndex: 1, width: '100%', maxWidth: 1200, mx: 'auto' }}>
        <Grid container spacing={4}>
          <Grid item xs={12}>
            <Card
              sx={{
                background: 'linear-gradient(135deg, rgba(10,10,10,0.95) 0%, rgba(20,20,20,0.85) 100%)',
                color: '#e0e0e0',
                border: '1px solid rgba(255,255,255,0.08)',
                borderRadius: 4,
                boxShadow: '0 20px 80px rgba(0,0,0,0.55)'
              }}
            >
              <CardContent sx={{ p: { xs: 3, md: 4 } }}>
                <Box sx={{ display: 'flex', flexDirection: { xs: 'column', md: 'row' }, justifyContent: 'space-between', gap: 3 }}>
                  <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
                    <Avatar
                      sx={{
                        width: 60,
                        height: 60,
                        bgcolor: '#1f1f1f',
                        fontSize: '1.2rem',
                        fontWeight: 600,
                        border: '2px solid rgba(255,255,255,0.15)'
                      }}
                    >
                      {userInfo.name.charAt(0).toUpperCase()}
                    </Avatar>
                    <Box>
                      <Typography variant="overline" sx={{ color: 'rgba(255,255,255,0.65)', letterSpacing: 2 }}>
                        Clinician Workspace
                      </Typography>
                      <Typography variant="h4" sx={{ fontWeight: 700 }}>
                        Welcome back, {userInfo.name}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        {userInfo.email}
                      </Typography>
                    </Box>
                  </Box>
                  <Box sx={{ display: 'flex', flexDirection: { xs: 'column', sm: 'row' }, gap: 1, alignItems: { sm: 'center' } }}>
                    <Button
                      variant="outlined"
                      startIcon={<Logout />}
                      onClick={handleLogout}
                      sx={{
                        borderColor: '#4fc3f7',
                        color: '#4fc3f7',
                        '&:hover': { borderColor: '#81d4fa', color: '#81d4fa' }
                      }}
                    >
                      Sign Out
                    </Button>
                  </Box>
                </Box>
                <Grid container spacing={2} sx={{ mt: 3 }}>
                  {highlightCards.map((card) => (
                    <Grid item xs={12} md={4} key={card.label}>
                      <Paper
                        elevation={0}
                        sx={{
                          p: 2,
                          borderRadius: 3,
                          bgcolor: 'rgba(255,255,255,0.04)',
                          border: '1px solid rgba(255,255,255,0.08)'
                        }}
                      >
                        <Typography variant="caption" sx={{ letterSpacing: 1, textTransform: 'uppercase', color: 'rgba(255,255,255,0.65)' }}>
                          {card.label}
                        </Typography>
                        <Typography variant="h5" sx={{ fontWeight: 700, color: '#e0e0e0' }}>
                          {card.value}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          {card.helper}
                        </Typography>
                      </Paper>
                    </Grid>
                  ))}
                </Grid>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12}>
            <Card
              sx={{
                background: '#111',
                border: '1px solid #222',
                borderRadius: 4,
                boxShadow: '0 20px 60px rgba(0,0,0,0.35)'
              }}
            >
              <CardContent sx={{ p: { xs: 3, md: 4 } }}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2, gap: 1.5 }}>
                  <MedicalServices sx={{ fontSize: '1.5rem', color: '#4fc3f7' }} />
                  <Box>
                    <Typography variant="h6" sx={{ fontWeight: 700, color: '#e0e0e0' }}>
                      Neuroimaging Analysis
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Select a modality and continue to upload for automated triage.
                    </Typography>
                  </Box>
                </Box>

                <Grid container spacing={3}>
                  <Grid item xs={12} md={7}>
                    <RadioGroup value={dataType} onChange={(e) => setDataType(e.target.value)}>
                      <Grid container spacing={2}>
                        {dataTypeOptions.map((option) => {
                          const OptionIcon = option.icon;
                          const isActive = dataType === option.value;
                          return (
                            <Grid item xs={12} key={option.value}>
                              <Paper
                                sx={{
                                  p: 2,
                                  borderRadius: 3,
                                  border: isActive ? '2px solid #00ffff' : '1px solid #333333',
                                  bgcolor: isActive ? '#1c1c1c' : 'transparent',
                                  boxShadow: isActive ? '0 0 12px 3px #00ffff44' : 'none',
                                  transition: 'all 0.2s ease'
                                }}
                              >
                                <FormControlLabel
                                  value={option.value}
                                  control={<Radio />}
                                  label={
                                    <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 2 }}>
                                      <Box
                                        sx={{
                                          width: 44,
                                          height: 44,
                                          borderRadius: '50%',
                                          bgcolor: 'rgba(255,255,255,0.05)',
                                          display: 'flex',
                                          alignItems: 'center',
                                          justifyContent: 'center',
                                          border: '1px solid rgba(255,255,255,0.1)'
                                        }}
                                      >
                                        {OptionIcon && <OptionIcon sx={{ color: '#4fc3f7' }} />}
                                      </Box>
                                      <Box>
                                        <Typography variant="body1" sx={{ fontWeight: 600 }}>
                                          {option.title}
                                        </Typography>
                                        <Typography variant="body2" color="text.secondary">
                                          {option.description}
                                        </Typography>
                                      </Box>
                                    </Box>
                                  }
                                />
                              </Paper>
                            </Grid>
                          );
                        })}
                      </Grid>
                    </RadioGroup>
                  </Grid>
                  <Grid item xs={12} md={5}>
                    <Paper
                      sx={{
                        p: 3,
                        borderRadius: 3,
                        bgcolor: '#181818',
                        border: '1px solid #2a2a2a',
                        height: '100%'
                      }}
                    >
                      <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 1 }}>
                        {selectedOption?.title}
                      </Typography>
                      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                        {selectedOption?.description}
                      </Typography>
                      <Divider sx={{ mb: 2, borderColor: '#333' }} />
                      <Typography variant="caption" sx={{ textTransform: 'uppercase', letterSpacing: 1, color: 'rgba(255,255,255,0.6)' }}>
                        Modality Insights
                      </Typography>
                      <Box component="ul" sx={{ mt: 1.5, pl: 3, color: '#b0b0b0' }}>
                        {getModalityInsights(dataType).map((point) => (
                          <Typography component="li" key={point} variant="body2" sx={{ mb: 1, lineHeight: 1.5 }}>
                            {point}
                          </Typography>
                        ))}
                      </Box>
                    </Paper>
                  </Grid>
                </Grid>

                <Button
                  variant="contained"
                  size="large"
                  fullWidth
                  onClick={handleProceed}
                  sx={{
                    mt: 4,
                    py: 1.5,
                    fontWeight: 700,
                    borderRadius: 3,
                    background: 'linear-gradient(90deg, #00bcd4, #00acc1)',
                    '&:hover': { background: 'linear-gradient(90deg, #00acc1, #0097a7)' }
                  }}
                >
                  Begin {dataType} Analysis Workflow
                </Button>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </Box>
    </Box>
  );
};

export default LandingPage; 
