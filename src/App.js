import React, { createContext, useState, useContext, useCallback } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider, CssBaseline, Box } from '@mui/material';
import theme from './theme';
import Navbar from './components/Navbar';
import LandingPage from './components/LandingPage';
import LoginPage from './components/LoginPage';
import UploadPage from './components/UploadPage';
import ProcessingPage from './components/ProcessingPage';
import ResultsPage from './components/ResultsPage';

// Create context for sharing files between components
const FileContext = createContext();

const arrayBufferToBase64 = (buffer) => {
  let binary = '';
  const bytes = new Uint8Array(buffer);
  const chunkSize = 0x8000;

  for (let i = 0; i < bytes.length; i += chunkSize) {
    const chunk = bytes.subarray(i, i + chunkSize);
    binary += String.fromCharCode.apply(null, chunk);
  }

  return window.btoa(binary);
};

export const useFiles = () => {
  const context = useContext(FileContext);
  if (!context) {
    throw new Error('useFiles must be used within a FileProvider');
  }
  return context;
};

const FileProvider = ({ children }) => {
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [folderData, setFolderData] = useState({
    rsn: [],
    noise: [],
    soz: []
  });
  const [processingComplete, setProcessingComplete] = useState(false);
  const [analysisResults, setAnalysisResults] = useState(null);
  const [analysisSummary, setAnalysisSummary] = useState(null);
  const [analysisError, setAnalysisError] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const clearFiles = () => {
    setUploadedFiles([]);
    setFolderData({ rsn: [], noise: [], soz: [] });
    setProcessingComplete(false);
    setAnalysisResults(null);
    setAnalysisSummary(null);
    setAnalysisError(null);
  };

  const getICNumber = (filename = '') => {
    const match = filename.match(/IC_(\d+)/);
    return match ? parseInt(match[1], 10) : null;
  };

  const uploadCaseToServer = useCallback(async () => {
    const filePayload = await Promise.all(
      uploadedFiles.map(async (file) => {
        if (!file.originalFile) {
          return null;
        }
        const arrayBuffer = await file.originalFile.arrayBuffer();
        const base64Content = arrayBufferToBase64(arrayBuffer);
        return {
          name: file.name,
          relativePath: file.relativePath || file.name,
          content: base64Content
        };
      })
    );

    const validFiles = filePayload.filter(Boolean);
    if (validFiles.length === 0) {
      throw new Error('No files available to upload.');
    }

    const response = await fetch('/api/upload-case', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ files: validFiles })
    });

    if (!response.ok) {
      let errorMessage = 'Failed to upload case files.';
      try {
        const errorBody = await response.json();
        if (errorBody?.error) {
          errorMessage = errorBody.error;
        }
      } catch {
        // ignore parse errors
      }
      throw new Error(errorMessage);
    }

    return response.json();
  }, [uploadedFiles]);

  const distributeFiles = useCallback(async () => {
    if (!uploadedFiles || uploadedFiles.length === 0) {
      console.warn("No files to distribute");
      setFolderData({ rsn: [], noise: [], soz: [] });
      setProcessingComplete(true);
      setAnalysisResults([]);
      setAnalysisSummary({
        totalComponents: 0,
        sozCount: 0,
        patientIsSoz: false,
        sozIcs: []
      });
      return { results: [], summary: null };
    }
    
    setIsAnalyzing(true);
    setAnalysisError(null);

    try {
      await uploadCaseToServer();

      const response = await fetch('/api/run-analysis', {
        method: 'POST'
      });

      if (!response.ok) {
        let errorMessage = 'Failed to run analysis.';
        try {
          const errorBody = await response.json();
          if (errorBody?.error) {
            errorMessage = errorBody.error;
          }
        } catch (errorResponse) {
          // Ignore JSON parse errors
        }
        throw new Error(errorMessage);
      }

      const data = await response.json();
      setAnalysisResults(data.results || []);
      setAnalysisSummary(data.summary || null);

      const resultsMap = new Map(
        (data.results || []).map((item) => [Number(item.ic), item])
      );

      const categorized = { rsn: [], noise: [], soz: [] };
      const publicUrl = process.env.PUBLIC_URL || '';
      const usedIcSet = new Set();

      uploadedFiles.forEach((file) => {
        if (!file.name?.toLowerCase().includes('_thresh')) {
          return;
        }
        const icNumber = getICNumber(file.name);
        const analysis = icNumber != null ? resultsMap.get(icNumber) : null;

        if (!analysis) {
          return;
        }

        if (usedIcSet.has(icNumber)) {
          return;
        }
        usedIcSet.add(icNumber);

        let aiCategory = 'rsn';
        let aiExplanation = 'Deep learning label suggests resting state network.';

        if (analysis.isSoz) {
          aiCategory = 'soz';
          aiExplanation =
            analysis.reason || 'Pipeline flagged this component as SOZ.';
        } else if (analysis.dlLabel === 0 || analysis.klPrediction === 3) {
          aiCategory = 'noise';
          aiExplanation =
            analysis.reason || 'Pipeline labelled this component as noise.';
        } else {
          aiCategory = 'rsn';
          aiExplanation =
            analysis.reason ||
            'Pipeline labelled this component as resting-state / non-SOZ.';
        }

        const fileWithAI = {
          ...file,
          aiCategory,
          aiExplanation,
          aiHeatmap: `${publicUrl}/AIHeatmap.png`,
          icNumber,
          analysisDetails: analysis || null
        };

        if (!categorized[aiCategory]) {
          categorized[aiCategory] = [];
        }
        categorized[aiCategory].push(fileWithAI);
      });

      setFolderData({
        rsn: categorized.rsn || [],
        noise: categorized.noise || [],
        soz: categorized.soz || []
      });
      setProcessingComplete(true);
      return data;
    } catch (error) {
      console.error('Error running analysis pipeline:', error);
      setAnalysisError(error.message);
      setProcessingComplete(false);
      throw error;
    } finally {
      setIsAnalyzing(false);
    }
  }, [uploadedFiles, uploadCaseToServer]);

  return (
    <FileContext.Provider value={{
      uploadedFiles,
      setUploadedFiles,
      folderData,
      setFolderData,
      processingComplete,
      setProcessingComplete,
      clearFiles,
      distributeFiles,
      analysisResults,
      analysisSummary,
      analysisError,
      isAnalyzing
    }}>
      {children}
    </FileContext.Provider>
  );
};

function App() {
  // For production deployment, use root path since we're deploying to demo.epiprecision.tech
  const basename = process.env.NODE_ENV === 'production' 
    ? '/' 
    : '/';

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <FileProvider>
        <Router basename={basename}>
          <Navbar />
          <Box component="main" sx={{ width: '100%' }}>
            <Routes>
              <Route path="/" element={<LandingPage />} />
              <Route path="/login" element={<LoginPage />} />
              <Route path="/upload" element={<UploadPage />} />
              <Route path="/processing" element={<ProcessingPage />} />
              <Route path="/results" element={<ResultsPage />} />
            </Routes>
          </Box>
        </Router>
      </FileProvider>
    </ThemeProvider>
  );
}

export default App; 
