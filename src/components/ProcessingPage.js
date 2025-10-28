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
  Alert
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
  { label: 'Upload Data', icon: <CloudUpload sx={{ color: '#e0e0e0' }} /> },
  { label: 'AI Analysis in Progress', icon: <Science sx={{ color: '#e0e0e0' }} /> },
  { label: 'Results Categorized', icon: <FolderSpecial sx={{ color: '#e0e0e0' }} /> },
  { label: 'Review & Download', icon: <Assessment sx={{ color: '#e0e0e0' }} /> }
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
    <Box sx={{ minHeight: '80vh', p: 3, display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
      <Card sx={{ minWidth: 400, maxWidth: 600, mx: 'auto', boxShadow: 4, borderRadius: 3 }}>
        {/* Gradient Header */}
        <Box sx={{
          background: '#1a1a1a',
          border: '1px solid #333333',
          color: '#e0e0e0',
          borderTopLeftRadius: 12,
          borderTopRightRadius: 12,
          p: 3,
          display: 'flex',
          alignItems: 'center',
          gap: 2
        }}>
          <Avatar sx={{ bgcolor: '#2a2a2a', color: '#e0e0e0', width: 48, height: 48, boxShadow: 2 }}>
            <Assessment fontSize="large" />
          </Avatar>
          <Box>
            <Typography variant="h5" sx={{ fontWeight: 700, color: '#e0e0e0' }}>
              {steps[activeStep].label}
            </Typography>
            <Typography variant="body2" sx={{ color: 'rgba(224,224,224,0.9)' }}>
              AI-powered Workflow
            </Typography>
          </Box>
        </Box>
        <CardContent sx={{ p: 4 }}>
          {/* File count info */}
          {uploadedFiles.length > 0 && (
            <Paper sx={{ mb: 3, p: 2, bgcolor: '#1a1a1a', borderRadius: 2, textAlign: 'center', border: '1px solid #333333' }}>
              <Typography variant="body2" sx={{ color: '#00ffff', fontWeight: 600 }}>
                Processing <b>{uploadedFiles.length}</b> uploaded {uploadedFiles.length === 1 ? 'file' : 'files'}
              </Typography>
            </Paper>
          )}

          {/* Stepper */}
          <Stepper activeStep={activeStep} alternativeLabel connector={<CyanConnector />} sx={{ mb: 4 }}>
            {steps.map((step, idx) => (
              <Step key={step.label} completed={activeStep > idx}>
                <StepLabel 
                  icon={step.icon}
                  sx={{
                    color: activeStep >= idx ? '#00ffff' : '#b0b0b0', // icon and label cyan
                    '& .MuiStepLabel-label': {
                      color: activeStep >= idx ? '#00ffff' : '#b0b0b0',
                      fontWeight: activeStep >= idx ? 600 : 400,
                    },
                  }}
                >
                  {step.label}
                </StepLabel>
              </Step>
            ))}
          </Stepper>

          {/* Step Content */}
          <Box sx={{ minHeight: 120, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
            {activeStep === 0 && (
              <Typography sx={{ mb: 2, color: '#00ffff', fontWeight: 600 }}>Ready to begin AI-powered analysis.</Typography>
            )}
            {activeStep === 1 && (
              <Typography sx={{ mb: 2, color: '#00ffff', fontWeight: 600 }}>AI Analysis in Progress…</Typography>
            )}
            {activeStep === 2 && (
              <Typography sx={{ mb: 2, color: '#00ffff', fontWeight: 600 }}>Results being categorized by AI.</Typography>
            )}
            {activeStep === 3 && (
              <>
                <Typography sx={{ mb: 2 }}>You can now review and download your results.</Typography>
                {analysisSummary && (
                  <Paper sx={{ p: 2, mb: 2, bgcolor: '#1a1a1a', border: '1px solid #333333', borderRadius: 2 }}>
                    <Typography variant="body2" sx={{ color: '#e0e0e0', textAlign: 'center' }}>
                      Patient Status: <strong>{analysisSummary.patientIsSoz ? 'SOZ Detected' : 'No SOZ Detected'}</strong>
                    </Typography>
                    <Typography variant="caption" display="block" sx={{ color: 'text.secondary', textAlign: 'center', mt: 1 }}>
                      SOZ ICs: {analysisSummary.sozIcs && analysisSummary.sozIcs.length > 0 ? analysisSummary.sozIcs.join(', ') : 'None'}
                    </Typography>
                  </Paper>
                )}
                {analysisSummary && (
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                    {analysisSummary.totalComponents ?? 'All'} ICs analyzed and categorized. Noise/ SOZ folders reflect AI results.
                  </Typography>
                )}
                <Button 
                  variant="contained" 
                  fullWidth 
                  size="large" 
                  sx={{ mt: 1, backgroundColor: '#2a2a2a', color: '#e0e0e0', '&:hover': { backgroundColor: '#333333' } }} 
                  onClick={() => navigate('/results')}
                >
                  Review
                </Button>
              </>
            )}
            {(loading || isAnalyzing) && (
              <Box display="flex" justifyContent="center" mt={2}>
                <CircularProgress />
              </Box>
            )}
          </Box>

          {analysisError && (
            <Alert 
              severity="error" 
              sx={{ mt: 2 }}
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
    </Box>
  );
};

export default ProcessingPage; 
