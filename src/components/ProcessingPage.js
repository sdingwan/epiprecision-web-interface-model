import React, { useState, useEffect, useRef } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Button,
  Box,
  Stepper,
  Step,
  StepLabel,
  CircularProgress,
  Paper,
  Avatar,
  Alert,
  Container
} from '@mui/material';
import { useNavigate } from 'react-router-dom';
import { useFiles } from '../App';
import {
  Science,
  FolderSpecial,
  CloudUpload,
  Assessment
} from '@mui/icons-material';
import { styled } from '@mui/material/styles';
import StepConnector, { stepConnectorClasses } from '@mui/material/StepConnector';

const steps = [
  { label: 'Upload Data', icon: CloudUpload },
  { label: 'AI Analysis', icon: Science },
  { label: 'Categorizing Results', icon: FolderSpecial },
  { label: 'Review & Download', icon: Assessment }
];

const CyanConnector = styled(StepConnector)(({ theme }) => ({
  [`&.${stepConnectorClasses.alternativeLabel}`]: {
    top: 22,
  },
  [`&.${stepConnectorClasses.active}`]: {
    [`& .${stepConnectorClasses.line}`]: {
      backgroundColor: '#00ffff',
    },
  },
  [`&.${stepConnectorClasses.completed}`]: {
    [`& .${stepConnectorClasses.line}`]: {
      backgroundColor: '#00ffff',
    },
  },
  [`& .${stepConnectorClasses.line}`]: {
    height: 3,
    border: 0,
    backgroundColor: '#b0b0b0',
    borderRadius: 1,
    transition: 'background-color 0.3s ease',
  },
}));

const ProcessingPage = () => {
  const [activeStep, setActiveStep] = useState(0);
  const [loading, setLoading] = useState(false);
  const analysisStartedRef = useRef(false);
  const navigate = useNavigate();
  const { 
    uploadedFiles, 
    distributeFiles,
    analysisError,
    analysisSummary,
    isAnalyzing
  } = useFiles();
  
  // Debug: log uploaded files
  console.log('ProcessingPage - uploadedFiles:', uploadedFiles);

  // Auto-advance steps
  useEffect(() => {
    if (activeStep === 0) {
      setLoading(true);
      const t = setTimeout(() => setActiveStep(1), 1000);
      return () => clearTimeout(t);
    }
    if (activeStep === 1) {
      setLoading(true);
      const t = setTimeout(() => setActiveStep(2), 2000);
      return () => clearTimeout(t);
    }
    if (activeStep === 2 && !analysisStartedRef.current) {
      analysisStartedRef.current = true;
      const runAnalysis = async () => {
        setLoading(true);
        try {
          await distributeFiles();
          setActiveStep(3);
        } catch (error) {
          // Keep the user on the analysis step for retry
          console.error('Analysis failed:', error);
        } finally {
          setLoading(false);
        }
      };
      runAnalysis();
    }
    if (activeStep === 3) {
      setLoading(false);
    }
  }, [activeStep, distributeFiles]);

  useEffect(() => {
    if (analysisError) {
      // Allow retry when an error occurs
      analysisStartedRef.current = false;
    }
  }, [analysisError]);

  const handleRetry = () => {
    analysisStartedRef.current = false;
    setActiveStep(2);
  };

  return (
    <Box
      sx={{
        position: 'relative',
        minHeight: '100vh',
        bgcolor: '#020202',
        py: { xs: 4, md: 6 },
        px: { xs: 2, md: 4 },
        overflow: 'hidden'
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
          backgroundImage: 'linear-gradient(120deg, rgba(255,255,255,0.05) 0%, transparent 50%, rgba(255,255,255,0.04) 100%)',
          opacity: 0.25
        }}
      />

      <Container maxWidth="md" sx={{ position: 'relative', zIndex: 1 }}>
        <Card
          sx={{
            borderRadius: 4,
            background: 'linear-gradient(180deg, rgba(15,15,15,0.95) 0%, rgba(11,11,11,0.9) 100%)',
            border: '1px solid rgba(255,255,255,0.08)',
            boxShadow: '0 25px 80px rgba(0,0,0,0.65)'
          }}
        >
          <CardContent sx={{ p: { xs: 3, md: 4 } }}>
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 3, flexWrap: 'wrap', gap: 2 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                <Avatar sx={{ bgcolor: '#1c1c1c', width: 54, height: 54, border: '2px solid rgba(255,255,255,0.1)' }}>
                  <Assessment />
                </Avatar>
                <Box>
                  <Typography variant="overline" sx={{ color: 'rgba(255,255,255,0.7)', letterSpacing: 2 }}>
                    AI Processing Workflow
                  </Typography>
                  <Typography variant="h5" sx={{ fontWeight: 700, color: '#fff' }}>
                    {steps[activeStep].label}
                  </Typography>
                </Box>
              </Box>
              {uploadedFiles.length > 0 && (
                <Paper
                  elevation={0}
                  sx={{
                    px: 3,
                    py: 1.5,
                    borderRadius: 3,
                    bgcolor: 'rgba(255,255,255,0.04)',
                    border: '1px solid rgba(255,255,255,0.08)'
                  }}
                >
                  <Typography variant="caption" sx={{ textTransform: 'uppercase', letterSpacing: 1, color: 'rgba(255,255,255,0.6)' }}>
                    Files in queue
                  </Typography>
                  <Typography variant="h6" sx={{ color: '#00ffff', fontWeight: 700 }}>
                    {uploadedFiles.length}
                  </Typography>
                </Paper>
              )}
            </Box>

            <Stepper activeStep={activeStep} alternativeLabel connector={<CyanConnector />} sx={{ mb: 4 }}>
              {steps.map((step, idx) => {
                const StepIcon = step.icon;
                return (
                  <Step key={step.label} completed={activeStep > idx}>
                    <StepLabel
                      icon={<StepIcon sx={{ color: activeStep >= idx ? '#00ffff' : '#555555' }} />}
                      sx={{
                        '& .MuiStepLabel-label': {
                          color: activeStep >= idx ? '#00ffff' : '#b0b0b0',
                          fontWeight: activeStep >= idx ? 600 : 400
                        }
                      }}
                    >
                      {step.label}
                    </StepLabel>
                  </Step>
                );
              })}
            </Stepper>

            <Box sx={{ minHeight: 200, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', textAlign: 'center', gap: 2 }}>
              {activeStep < 3 && (
                <>
                  <Typography variant="h6" sx={{ color: '#e0e0e0', fontWeight: 600 }}>
                    {activeStep === 0 && 'Preparing your dataset for AI triage.'}
                    {activeStep === 1 && 'AI analysis is interpreting each component.'}
                    {activeStep === 2 && 'Categorizing ICs into clinician review buckets.'}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {loading || isAnalyzing ? 'This typically takes a few moments. Feel free to monitor progress.' : 'Waiting to start analysis…'}
                  </Typography>
                  {(loading || isAnalyzing) && <CircularProgress sx={{ color: '#00ffff' }} />}
                </>
              )}

              {activeStep === 3 && (
                <>
                  <Typography variant="h6" sx={{ color: '#e0e0e0', fontWeight: 700 }}>
                    Analysis complete. Ready for clinician review.
                  </Typography>
                  {analysisSummary && (
                    <Paper
                      elevation={0}
                      sx={{
                        p: 2.5,
                        borderRadius: 3,
                        bgcolor: 'rgba(255,255,255,0.04)',
                        border: '1px solid rgba(255,255,255,0.08)'
                      }}
                    >
                      <Typography variant="body1" sx={{ fontWeight: 600, color: '#fff' }}>
                        Patient Status: {analysisSummary.patientIsSoz ? 'SOZ Detected' : 'No SOZ Detected'}
                      </Typography>
                      <Typography variant="caption" sx={{ color: 'rgba(255,255,255,0.7)' }}>
                        {analysisSummary.sozIcs?.length ? `SOZ ICs: ${analysisSummary.sozIcs.join(', ')}` : 'No SOZ components flagged.'}
                      </Typography>
                      <Typography variant="caption" sx={{ display: 'block', mt: 1, color: 'rgba(255,255,255,0.5)' }}>
                        {analysisSummary.totalComponents ?? 'All'} ICs processed
                      </Typography>
                    </Paper>
                  )}
                  <Button
                    variant="contained"
                    size="large"
                    onClick={() => navigate('/results')}
                    sx={{
                      mt: 2,
                      px: 4,
                      py: 1.2,
                      borderRadius: 3,
                      fontWeight: 700,
                      background: 'linear-gradient(90deg, #00bcd4, #0097a7)',
                      '&:hover': { background: 'linear-gradient(90deg, #0097a7, #007c91)' }
                    }}
                  >
                    Open Results Workspace
                  </Button>
                </>
              )}
            </Box>

            {analysisError && (
              <Alert
                severity="error"
                sx={{ mt: 3 }}
                action={
                  <Button color="inherit" size="small" onClick={handleRetry}>
                    Retry
                  </Button>
                }
              >
                {analysisError}
              </Alert>
            )}
          </CardContent>
        </Card>
      </Container>
    </Box>
  );
};

export default ProcessingPage; 
