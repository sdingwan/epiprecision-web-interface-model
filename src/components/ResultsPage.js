import React, { useState, useMemo } from 'react';
import { 
  Card, 
  CardContent, 
  Typography, 
  Box, 
  Grid, 
  Badge, 
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  IconButton,
  Chip,
  MenuItem,
  Select,
  Paper,
  Avatar,
  Container,
  Alert,
  TextField,
  DialogActions
} from '@mui/material';
import { 
  Close, 
  InsertDriveFile,
  CheckCircleOutline,
  HighlightOff,
  Download,
  Assessment,
  KeyboardArrowDown,
  PsychologyAlt,
  ReportProblemRounded,
  WarningAmber
} from '@mui/icons-material';
import { useFiles } from '../App';
import jsPDF from 'jspdf';
import 'jspdf-autotable';

const ResultsPage = () => {
  const { 
    folderData, 
    processingComplete, 
    uploadedFiles, 
    setFolderData,
    analysisSummary 
  } = useFiles();
  
  // Debug: log data
  console.log('ResultsPage - uploadedFiles:', uploadedFiles);
  console.log('ResultsPage - folderData:', folderData);
  console.log('ResultsPage - processingComplete:', processingComplete);
  const [selectedFolderId, setSelectedFolderId] = useState(null);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [imagePreviewUrl, setImagePreviewUrl] = useState(null);
  const [shareDialogOpen, setShareDialogOpen] = useState(false);
  const [shareEmail, setShareEmail] = useState('');
  const [shareMessage, setShareMessage] = useState('');
  const [shareError, setShareError] = useState('');

  const folders = [
    {
      id: 'rsn',
      name: 'RSN',
      color: '#e3f2fd',
      borderColor: '#64b5f6',
      description: 'Resting-state networks and non-pathologic components',
      files: folderData?.rsn || [],
      badgeColor: 'info',
      Icon: PsychologyAlt,
      iconColor: '#1976d2'
    },
    {
      id: 'noise',
      name: 'Noise', 
      color: '#fffde7',
      borderColor: '#fbc02d',
      description: 'Small clusters on white matter and brain periphery',
      files: folderData?.noise || [],
      badgeColor: 'warning',
      Icon: ReportProblemRounded,
      iconColor: '#f57f17'
    },
    {
      id: 'soz',
      name: 'SOZ',
      color: '#ffebee',
      borderColor: '#e53935',
      description: 'Large cluster on both grey and white matter (Seizure Onset Zone)',
      files: folderData?.soz || [],
      badgeColor: 'error',
      Icon: WarningAmber,
      iconColor: '#c62828'
    }
  ];

  const handleFolderClick = (folder) => {
    setSelectedFolderId(folder.id);
    setDialogOpen(true);
  };

  const handleCloseDialog = () => {
    setDialogOpen(false);
    setSelectedFolderId(null);
  };

  const createImagePreview = (file) => {
    try {
      if (file && file.type && file.type.startsWith('image/')) {
        // Use the blob URL if available, otherwise try to create one from original file
        return file.blobUrl || (file.originalFile ? URL.createObjectURL(file.originalFile) : null);
      }
    } catch (error) {
      console.error("Error creating image preview:", error);
    }
    return null;
  };

  const getTotalFileCount = () => {
    if (!folderData) return 0;
    return (
      (Array.isArray(folderData.rsn) ? folderData.rsn.length : 0) + 
      (Array.isArray(folderData.noise) ? folderData.noise.length : 0) + 
      (Array.isArray(folderData.soz) ? folderData.soz.length : 0)
    );
  };

  // Handle approval and explanation changes
  const handleApprovalChange = (folderId, fileId, value) => {
    setFolderData(prev => {
      if (!prev || !prev[folderId] || !Array.isArray(prev[folderId])) {
        return prev;
      }
      return {
        ...prev,
        [folderId]: prev[folderId].map(f => f.id === fileId ? { ...f, clinicianApproval: value } : f)
      };
    });
  };
  
  const handleExplanationChange = (folderId, fileId, value) => {
    setFolderData(prev => {
      if (!prev || !prev[folderId] || !Array.isArray(prev[folderId])) {
        return prev;
      }
      return {
        ...prev,
        [folderId]: prev[folderId].map(f => f.id === fileId ? { ...f, clinicianExplanation: value } : f)
      };
    });
  };

  // Get the latest folder object from folderData using selectedFolderId
  const selectedFolder = selectedFolderId
    ? folders.find(f => f.id === selectedFolderId)
    : null;

  // Sort files by IC number in ascending order
  const sortFilesByICNumber = (fileList) => {
    return [...fileList].sort((a, b) => {
      // Extract IC number from filename (e.g., "IC_51_thresh.png" -> 51)
      const getICNumber = (filename) => {
        const match = filename.match(/IC_(\d+)/);
        return match ? parseInt(match[1], 10) : 0;
      };
      
      const aNumber = getICNumber(a.name);
      const bNumber = getICNumber(b.name);
      
      return aNumber - bNumber;
    });
  };

  // PDF generation
  const handleDownloadPDF = async () => {
    const doc = new jsPDF();
    let y = 20;
    const pageHeight = 297; // A4 height in mm
    const pageWidth = 210; // A4 width in mm
    const margin = 20;
    const contentWidth = pageWidth - (margin * 2);

    // Helper function to check if we need a new page
    const checkNewPage = (requiredSpace = 40) => {
      if (y + requiredSpace > pageHeight - margin) {
        doc.addPage();
        y = margin;
        return true;
      }
      return false;
    };

    // Header
    doc.setFontSize(22);
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(25, 118, 210); // Primary blue
    doc.text('EpiPrecision AI Analysis Report', margin, y);
    y += 15;

    // Date and summary
    doc.setFontSize(11);
    doc.setFont('helvetica', 'normal');
    doc.setTextColor(100, 100, 100);
    doc.text(`Generated: ${new Date().toLocaleString()}`, margin, y);
    y += 8;
    doc.text(`Total Files Analyzed: ${uploadedFiles.length}`, margin, y);
    y += 8;
    doc.text(`Categories: ${folders.filter(f => f.files.length > 0).length} active`, margin, y);
    y += 20;

    // Draw separator line
    doc.setDrawColor(200, 200, 200);
    doc.line(margin, y - 5, pageWidth - margin, y - 5);
    y += 5;

    // Process each folder category in priority order: SOZ, Noise, RSN
    const pdfCategoryOrder = ['soz', 'noise', 'rsn'];
    
    for (const categoryId of pdfCategoryOrder) {
      const folder = folders.find(f => f.id === categoryId);
      if (!folder || !folder.files || folder.files.length === 0) continue;

      checkNewPage(30);

      // Category header
      doc.setFontSize(16);
      doc.setFont('helvetica', 'bold');
      doc.setTextColor(60, 60, 60);
      doc.text(`${folder.name} Category`, margin, y);
      y += 8;

      doc.setFontSize(10);
      doc.setFont('helvetica', 'normal');
      doc.setTextColor(120, 120, 120);
      doc.text(`${folder.description} • ${folder.files.length} file${folder.files.length !== 1 ? 's' : ''}`, margin, y);
      y += 15;

      // Process each file in this category
      const sortedFiles = sortFilesByICNumber(folder.files);
      for (let i = 0; i < sortedFiles.length; i++) {
        const file = sortedFiles[i];
        
        checkNewPage(60); // Ensure enough space for file entry

        // File entry background
        doc.setFillColor(248, 249, 250);
        doc.rect(margin, y - 5, contentWidth, 75, 'F'); // Increased height from 50 to 75 to accommodate AI Analysis text below heatmap
        
        // Original Brain Image (left side)
        const brainImgX = margin + 5;
        const brainImgY = y;
        const brainImgSize = 30;

        // Add original brain image if available
        if (file.blobUrl) {
          try {
            const brainImgData = await getImageDataUrl(file.blobUrl);
            doc.addImage(brainImgData, 'JPEG', brainImgX, brainImgY, brainImgSize, brainImgSize);
          } catch (e) {
            // Draw placeholder if image fails
            doc.setDrawColor(200, 200, 200);
            doc.rect(brainImgX, brainImgY, brainImgSize, brainImgSize);
            doc.setFontSize(8);
            doc.setTextColor(150, 150, 150);
            doc.text('No Image', brainImgX + 8, brainImgY + 18);
          }
        } else {
          // Draw placeholder
          doc.setDrawColor(200, 200, 200);
          doc.rect(brainImgX, brainImgY, brainImgSize, brainImgSize);
          doc.setFontSize(8);
          doc.setTextColor(150, 150, 150);
          doc.text('No Image', brainImgX + 8, brainImgY + 18);
        }

        // AI Heatmap Image (right side)
        const heatmapImgX = margin + (contentWidth * 0.7);
        const heatmapImgY = y;
        const heatmapImgSize = 30;

        // Add AI heatmap if available
        if (file.aiHeatmap) {
          try {
            const heatmapImgData = await getImageDataUrl(file.aiHeatmap);
            doc.addImage(heatmapImgData, 'JPEG', heatmapImgX, heatmapImgY, heatmapImgSize, heatmapImgSize);
          } catch (e) {
            // Draw placeholder if heatmap fails
            doc.setDrawColor(200, 200, 200);
            doc.rect(heatmapImgX, heatmapImgY, heatmapImgSize, heatmapImgSize);
            doc.setFontSize(8);
            doc.setTextColor(150, 150, 150);
            doc.text('No Heatmap', heatmapImgX + 2, heatmapImgY + 18);
          }
        } else {
          // Draw placeholder
          doc.setDrawColor(200, 200, 200);
          doc.rect(heatmapImgX, heatmapImgY, heatmapImgSize, heatmapImgSize);
          doc.setFontSize(8);
          doc.setTextColor(150, 150, 150);
          doc.text('No Heatmap', heatmapImgX + 2, heatmapImgY + 18);
        }

        // File information area (center, between images)
        const textX = brainImgX + brainImgSize + 10;
        let textY = y + 5;

        // File name
        doc.setFontSize(11);
        doc.setFont('helvetica', 'bold');
        doc.setTextColor(40, 40, 40);
        const fileName = file.name.length > 35 ? file.name.substring(0, 35) + '...' : file.name;
        doc.text(`File: ${fileName}`, textX, textY);
        textY += 6;

        // File details
        doc.setFontSize(9);
        doc.setFont('helvetica', 'normal');
        doc.setTextColor(80, 80, 80);
        doc.text(`Type: ${file.dataType || 'Unknown'}`, textX, textY);
        textY += 5;
        doc.text(`Category: ${folder.name}`, textX, textY);
        textY += 5;
        doc.text(`Status: ${file.clinicianApproval || 'Pending Review'}`, textX, textY);
        textY += 8;

        // Clinical note (if available)
        if (file.clinicalNote) {
          doc.setFont('helvetica', 'bold');
          doc.setTextColor(60, 60, 60);
          doc.text('Clinical Note:', textX, textY);
          textY += 5;
          
          doc.setFont('helvetica', 'normal');
          doc.setTextColor(40, 40, 40);
          const clinicalNote = file.clinicalNote.length > 50 ? file.clinicalNote.substring(0, 50) + '...' : file.clinicalNote;
          doc.text(clinicalNote, textX, textY);
        }

        // Explanation area (below the heatmap image)
        const explanationX = heatmapImgX;
        let explanationY = heatmapImgY + heatmapImgSize + 8;
        
        doc.setFontSize(9);
        doc.setFont('helvetica', 'bold');
        doc.setTextColor(60, 60, 60);
        doc.text('Explanation:', explanationX, explanationY);
        explanationY += 5;

        doc.setFont('helvetica', 'normal');
        doc.setTextColor(80, 80, 80);
        const explanation = file.clinicianExplanation !== undefined 
          ? file.clinicianExplanation 
          : (file.aiExplanation || 'No explanation available');
        
        // Split explanation into multiple lines if needed
        const maxWidth = (contentWidth * 0.25);
        const explanationLines = doc.splitTextToSize(explanation, maxWidth);
        doc.text(explanationLines.slice(0, 3), explanationX, explanationY); // Max 3 lines

        y += 70; // Move to next file position (increased to accommodate AI Analysis text below heatmap)
      }

      y += 10; // Extra space between categories
    }

    // Footer on last page
    const totalPages = doc.internal.getNumberOfPages();
    for (let i = 1; i <= totalPages; i++) {
      doc.setPage(i);
      doc.setFontSize(8);
      doc.setTextColor(150, 150, 150);
      doc.text(`EpiPrecision Medical Imaging Platform - Page ${i} of ${totalPages}`, margin, pageHeight - 10);
      doc.text('Confidential Medical Report', pageWidth - margin - 40, pageHeight - 10);
    }

    // Save the PDF
    doc.save('EpiPrecision_Analysis_Report.pdf');
  };

  // Helper to get image data URL from blob URL or static path
  const getImageDataUrl = (url) => {
    return new Promise((resolve, reject) => {
      const img = new window.Image();
      img.crossOrigin = 'Anonymous';
      img.onload = function () {
        const canvas = document.createElement('canvas');
        canvas.width = img.width;
        canvas.height = img.height;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0);
        resolve(canvas.toDataURL('image/jpeg'));
      };
      img.onerror = reject;
      img.src = url;
    });
  };

  const patientStatus = analysisSummary
    ? analysisSummary.patientIsSoz
      ? { label: 'SOZ Detected', color: 'error', helper: 'Patient flagged for seizure onset review' }
      : { label: 'Clear', color: 'success', helper: 'No SOZ components identified' }
    : { label: 'Pending Analysis', color: 'default', helper: 'Run analysis to determine status' };

  const summaryMetrics = [
    {
      label: 'Files Uploaded',
      value: uploadedFiles.length,
      helper: 'Clinician curated dataset'
    },
    {
      label: 'ICs Analyzed',
      value: analysisSummary?.totalComponents ?? getTotalFileCount(),
      helper: 'Independent components processed'
    },
    {
      label: 'SOZ Components',
      value: analysisSummary?.sozCount ?? 0,
      helper: patientStatus.helper
    }
  ];

  const shareSummary = useMemo(() => {
    const totalReviewed = analysisSummary?.totalComponents ?? getTotalFileCount();
    const soz = analysisSummary?.sozCount ?? 0;
    const statusText = analysisSummary
      ? analysisSummary.patientIsSoz
        ? 'SOZ detected'
        : 'No SOZ detected'
      : 'Analysis pending';
    return `Files processed: ${uploadedFiles.length}\nICs analyzed: ${totalReviewed}\nSOZ components: ${soz}\nPatient status: ${statusText}`;
  }, [analysisSummary, uploadedFiles.length, folderData]);

  const defaultShareMessage = useMemo(() => (
    `Hi colleague,\n\nI'm sharing the latest EpiPrecision analysis results for review.\n\n${shareSummary}\n\nPlease see the attached PDF or open the clinician portal for full detail.\n\nThanks.`
  ), [shareSummary]);

  const openShareDialog = () => {
    setShareEmail('');
    setShareError('');
    setShareMessage(defaultShareMessage);
    setShareDialogOpen(true);
  };

  const handleShareSend = () => {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(shareEmail)) {
      setShareError('Enter a valid email address.');
      return;
    }
    const subject = 'EpiPrecision Analysis Results';
    const body = `${shareMessage}\n\nSummary:\n${shareSummary}`;
    const mailtoLink = `mailto:${encodeURIComponent(shareEmail)}?subject=${encodeURIComponent(subject)}&body=${encodeURIComponent(body)}`;
    window.open(mailtoLink, '_blank');
    setShareDialogOpen(false);
  };

  return (
    <Box
      sx={{
        minHeight: 'calc(100vh - 70px)',
        display: 'flex',
        alignItems: 'stretch',
        bgcolor: '#0a0a0a',
        py: 0,
        width: '100%',
        boxSizing: 'border-box'
      }}
    >
      <Container
        maxWidth="lg"
        sx={{
          py: { xs: 3, md: 4 },
          display: 'flex',
          flexDirection: 'column',
          flexGrow: 1,
          boxSizing: 'border-box'
        }}
      >
      {/* Header Section */}
        <Box sx={{ 
          background: '#1a1a1a',
          border: '1px solid #333333',
          borderRadius: 3,
          p: { xs: 2, md: 3 },
          mb: 3,
          color: '#e0e0e0'
        }}>
          <Box 
            sx={{ 
              display: 'flex', 
              justifyContent: 'space-between', 
              flexWrap: 'wrap', 
              gap: 2,
              alignItems: 'center',
              mb: 2
            }}
          >
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <Avatar sx={{ bgcolor: '#2a2a2a', color: '#e0e0e0', width: 48, height: 48 }}>
                <Assessment fontSize="large" />
              </Avatar>
              <Box>
                <Typography variant="h5" sx={{ fontWeight: 700, mb: 0.5, color: '#e0e0e0' }}>
                  Analysis Results
                </Typography>
                <Typography variant="subtitle1" sx={{ opacity: 0.9, color: '#e0e0e0' }}>
                  AI-powered categorization complete
                </Typography>
              </Box>
            </Box>
            {processingComplete && (
              <Box sx={{ display: 'flex', gap: 1 }}>
                <Button 
                  variant="outlined" 
                  sx={{ 
                    borderColor: '#4fc3f7',
                    color: '#4fc3f7',
                    '&:hover': { borderColor: '#81d4fa', color: '#81d4fa' }
                  }}
                  onClick={openShareDialog}
                >
                  Share via Email
                </Button>
                <Button 
                  variant="contained" 
                  sx={{ 
                    bgcolor: '#2a2a2a', 
                    color: '#e0e0e0',
                    '&:hover': { bgcolor: '#333333' }
                  }}
                  startIcon={<Download />}
                  onClick={handleDownloadPDF}
                >
                  Download Report
                </Button>
              </Box>
            )}
          </Box>
          
          {processingComplete ? (
            <>
              <Typography variant="body2" sx={{ color: 'text.secondary', mb: 2 }}>
                {uploadedFiles.length} files processed • {analysisSummary?.totalComponents ?? getTotalFileCount()} ICs analyzed
              </Typography>
              <Grid container spacing={2}>
                {summaryMetrics.map((metric) => (
                  <Grid item xs={12} md={4} key={metric.label}>
                    <Paper 
                      elevation={0} 
                      sx={{ 
                        p: 2,
                        bgcolor: '#111',
                        border: '1px solid #222',
                        borderRadius: 2,
                        textAlign: 'center'
                      }}
                    >
                      <Typography variant="h4" sx={{ fontWeight: 700, color: '#e0e0e0' }}>
                        {metric.value}
                      </Typography>
                      <Typography variant="body2" sx={{ color: '#b0b0b0', fontWeight: 600 }}>
                        {metric.label}
                      </Typography>
                      <Typography variant="caption" sx={{ display: 'block', mt: 0.5, color: 'text.secondary' }}>
                        {metric.helper}
                      </Typography>
                    </Paper>
                  </Grid>
                ))}
              </Grid>
            </>
          ) : (
            <Typography variant="body2" color="text.secondary">
              Upload patient data to view categorized ICs.
            </Typography>
          )}
        </Box>

      {processingComplete ? (
        <>
          {/* Folder Categories */}
          <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
            File Categories
          </Typography>
          <Grid container spacing={2} sx={{ mb: 3 }}>
            {folders.map((folder) => (
              <Grid item xs={12} md={4} key={folder.id}>
                <Card 
                  sx={{ 
                    height: '100%',
                    cursor: 'pointer',
                    transition: 'all 0.3s ease',
                    border: `2px solid ${folder.borderColor}`,
                    '&:hover': {
                      transform: 'translateY(-4px)',
                      boxShadow: 6
                    }
                  }}
                  onClick={() => handleFolderClick(folder)}
                >
                  <CardContent sx={{ textAlign: 'center', p: 3 }}>
                    <Badge 
                      badgeContent={folder.files.length} 
                      color={folder.badgeColor}
                      sx={{ mb: 1 }}
                    >
                      <Avatar 
                        sx={{ 
                          bgcolor: folder.color,
                          color: folder.borderColor,
                          width: 56,
                          height: 56,
                          fontSize: '2rem',
                          border: `2px solid ${folder.borderColor}`
                        }}
                      >
                        {folder.Icon && (
                          <folder.Icon sx={{ fontSize: 28, color: folder.iconColor || folder.borderColor }} />
                        )}
                      </Avatar>
                    </Badge>
                    <Typography variant="h6" sx={{ fontWeight: 600, mb: 1 }}>
                      {folder.name}
                    </Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                      {folder.description}
                    </Typography>
                    <Chip 
                      label={`${folder.files.length} ${folder.files.length === 1 ? 'file' : 'files'}`}
                      color={folder.badgeColor}
                      variant="outlined"
                    />
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>

          {/* Hint removed to save space */}
        </>
      ) : (
        <Paper sx={{ p: 4, textAlign: 'center' }}>
          <Typography variant="h6" color="text.secondary" sx={{ mb: 2 }}>
            No Results Available
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Please complete the processing workflow first to view analysis results.
          </Typography>
        </Paper>
      )}

      {/* Share via Email Dialog */}
      <Dialog open={shareDialogOpen} onClose={() => setShareDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Share Results via Email</DialogTitle>
        <DialogContent sx={{ pt: 1 }}>
          {shareError && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {shareError}
            </Alert>
          )}
          <TextField
            label="Recipient Email"
            fullWidth
            value={shareEmail}
            onChange={(e) => {
              setShareEmail(e.target.value);
              if (shareError) setShareError('');
            }}
            sx={{ mb: 2 }}
            type="email"
          />
          <TextField
            label="Message"
            fullWidth
            multiline
            minRows={4}
            value={shareMessage}
            onChange={(e) => setShareMessage(e.target.value)}
          />
        </DialogContent>
        <DialogActions sx={{ px: 3, pb: 2 }}>
          <Button onClick={() => setShareDialogOpen(false)} color="inherit">
            Cancel
          </Button>
          <Button variant="contained" onClick={handleShareSend}>
            Compose Email
          </Button>
        </DialogActions>
      </Dialog>

      {/* Folder Contents Dialog */}
      <Dialog 
        open={dialogOpen} 
        onClose={handleCloseDialog}
        maxWidth="xl"
        fullWidth
        PaperProps={{ 
          sx: { borderRadius: 2 }
        }}
      >
        <DialogTitle sx={{ 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'space-between',
          bgcolor: '#1a1a1a',
          borderBottom: `3px solid ${selectedFolder?.borderColor}`,
          color: '#e0e0e0'
        }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <Avatar sx={{ bgcolor: selectedFolder?.borderColor, color: 'white' }}>
              {selectedFolder?.icon}
            </Avatar>
            <Box>
              <Typography variant="h6" sx={{ fontWeight: 600 }}>
                {selectedFolder?.name} Category
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {selectedFolder?.description}
              </Typography>
            </Box>
            <Chip 
              label={`${selectedFolder?.files.length || 0} files`} 
              color={selectedFolder?.badgeColor}
              sx={{ ml: 2 }}
            />
          </Box>
          <IconButton onClick={handleCloseDialog}>
            <Close />
          </IconButton>
        </DialogTitle>
        
        <DialogContent sx={{ 
          p: 2,
          mt: 1.5,
          background: '#1a1a1a', // Dark background for dialog body
          border: '3px solid #1a1a1a',
          borderTop: 0,
          borderRadius: '0 0 18px 18px',
          boxShadow: '0 2px 12px 0 rgba(0,0,0,0.6)',
        }}>
          <Box sx={{ 
            background: '#1a1a1a',
            borderRadius: 3,
            p: 2,
            minHeight: 180,
          }}>
            {selectedFolder && selectedFolder.files && selectedFolder.files.length > 0 ? (
              <Grid container spacing={2}>
                {sortFilesByICNumber(selectedFolder.files).map((file) => {
                  const imagePreview = createImagePreview(file);
                  const analysis = file.analysisDetails || {};
                  const isSoz = Boolean(analysis.isSoz);
                  return (
                    <Grid item xs={12} key={file.id}>
                      <Paper 
                        sx={{ 
                          p: 3,
                          border: `2px solid ${selectedFolder.borderColor}`,
                          borderRadius: 2,
                          bgcolor: '#1a1a1a',
                          color: selectedFolder.borderColor,
                        }}
                      >
                        <Grid container spacing={3} alignItems="flex-start">
                          {/* Header Section */}
                          <Grid item xs={12}>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                              <Box>
                                <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                                  {file.name}
                                </Typography>
                                <Chip 
                                  label={selectedFolder.name}
                                  color={selectedFolder.badgeColor}
                                  size="small"
                                />
                              </Box>
                              
                              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                                <Select
                                  value={file && typeof file.clinicianApproval === 'string' ? file.clinicianApproval : ''}
                                  onChange={e => handleApprovalChange(selectedFolder.id, file.id, e.target.value)}
                                  displayEmpty
                                  size="small"
                                  sx={{ 
                                    minWidth: 140, 
                                    borderRadius: 1,
                                    bgcolor: file?.clinicianApproval === 'approved' ? '#e8f5e9' : 
                                            file?.clinicianApproval === 'disapproved' ? '#ffebee' : 'white',
                                    color: file?.clinicianApproval === 'approved' ? '#2e7d32' : 
                                           file?.clinicianApproval === 'disapproved' ? '#d32f2f' : 'inherit',
                                    fontWeight: file?.clinicianApproval ? 600 : 'normal',
                                    '& .MuiSelect-icon': {
                                      color: 'rgba(0, 0, 0, 0.8)',
                                      fontSize: '1.2rem'
                                    }
                                  }}
                                  IconComponent={KeyboardArrowDown}
                                >
                                  <MenuItem value=""><em>Pending Review</em></MenuItem>
                                  <MenuItem value="approved">
                                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                      <CheckCircleOutline sx={{ color: 'green' }} />
                                      Approved
                                    </Box>
                                  </MenuItem>
                                  <MenuItem value="disapproved">
                                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                      <HighlightOff sx={{ color: 'red' }} />
                                      Disapproved
                                    </Box>
                                  </MenuItem>
                                </Select>
                              </Box>
                            </Box>
                          </Grid>

                          {/* Images Section */}
                          <Grid item xs={12}>
                            <Grid container spacing={2}>
                              <Grid item xs={12} md={6}>
                                <Box>
                                  <Typography variant="caption" color="text.secondary" sx={{ mb: 1, display: 'block', fontWeight: 600 }}>
                                    Original Image
                                  </Typography>
                                  {imagePreview ? (
                                    <Box
                                      component="img"
                                      src={imagePreview}
                                      alt={file.name}
                                      sx={{ 
                                        width: '100%',
                                        height: 300,
                                        objectFit: 'contain',
                                        borderRadius: 2,
                                        border: '2px solid #444444',
                                        cursor: 'pointer',
                                        transition: 'all 0.2s ease',
                                        '&:hover': { 
                                          opacity: 0.8,
                                          transform: 'scale(1.02)',
                                          borderColor: '#4fc3f7'
                                        }
                                      }}
                                      onClick={() => setImagePreviewUrl(imagePreview)}
                                    />
                                  ) : (
                                    <Box
                                      sx={{ 
                                        width: '100%',
                                        height: 300,
                                        display: 'flex',
                                        alignItems: 'center',
                                        justifyContent: 'center',
                                        bgcolor: '#f5f5f5',
                                        borderRadius: 2,
                                        border: '2px solid #ddd'
                                      }}
                                    >
                                      <InsertDriveFile color="action" sx={{ fontSize: '2rem' }} />
                                    </Box>
                                  )}
                                </Box>
                              </Grid>
                              {file.aiHeatmap && (
                                <Grid item xs={12} md={6}>
                                  <Box>
                                    <Typography variant="caption" color="text.secondary" sx={{ mb: 1, display: 'block', fontWeight: 600 }}>
                                      Brain Slice
                                    </Typography>
                                    <Box
                                      component="img"
                                      src={file.aiHeatmap}
                                      alt="Brain Slice"
                                      sx={{ 
                                        width: '100%',
                                        height: 300,
                                        objectFit: 'contain',
                                        borderRadius: 2,
                                        border: '2px solid #444444',
                                        cursor: 'pointer',
                                        transition: 'all 0.2s ease',
                                        '&:hover': { 
                                          opacity: 0.8,
                                          transform: 'scale(1.02)',
                                          borderColor: '#4fc3f7'
                                        }
                                      }}
                                      onClick={() => setImagePreviewUrl(file.aiHeatmap)}
                                    />
                                  </Box>
                                </Grid>
                              )}
                            </Grid>
                          </Grid>

                          {/* Explanations Section */}
                          <Grid item xs={12}>
                            <Grid container spacing={2}>
                              <Grid item xs={12} md={6}>
                                <Box>
                                  <Typography variant="subtitle2" color="text.secondary" sx={{ mb: 1, fontWeight: 600 }}>
                                    Explanation
                                  </Typography>
                                  <Box
                                    component="textarea"
                                    value={file.clinicianExplanation !== undefined ? file.clinicianExplanation : (file.aiExplanation || 'No explanation available')}
                                    onChange={e => handleExplanationChange(selectedFolder.id, file.id, e.target.value)}
                                    style={{ 
                                      width: '100%',
                                      height: '80px',
                                      padding: '10px',
                                      borderRadius: '8px',
                                      border: '1px solid #444444',
                                      fontSize: '0.95rem',
                                      fontFamily: 'inherit',
                                      resize: 'vertical',
                                      backgroundColor: '#1a1a1a',
                                      color: '#e0e0e0'
                                    }}
                                  />
                                </Box>
                              </Grid>
                              <Grid item xs={12} md={6}>
                                <Box>
                                  <Typography variant="subtitle2" color="text.secondary" sx={{ mb: 1, fontWeight: 600 }}>
                                    Clinical Note
                                  </Typography>
                                  {file && file.clinicalNote ? (
                                    <Paper sx={{ 
                                      p: 2, 
                                      bgcolor: '#2a2a2a', 
                                      color: '#e0e0e0', 
                                      height: '80px', 
                                      overflow: 'auto', 
                                      borderRadius: 2,
                                      border: '1px solid #444444',
                                      boxShadow: 'inset 0 1px 3px rgba(0,0,0,0.3)'
                                    }}>
                                      <Typography variant="body2" sx={{ fontSize: '0.9rem', lineHeight: 1.4, color: '#e0e0e0' }}>
                                        {file.clinicalNote}
                                      </Typography>
                                    </Paper>
                                  ) :
                                    <Paper sx={{ 
                                      p: 2, 
                                      bgcolor: '#2a2a2a', 
                                      height: '80px', 
                                      display: 'flex', 
                                      alignItems: 'center', 
                                      justifyContent: 'center', 
                                      borderRadius: 2,
                                      border: '1px solid #444444'
                                    }}>
                                      <Typography variant="body2" color="text.secondary" sx={{ fontStyle: 'italic' }}>
                                        No clinical note provided
                                      </Typography>
                                    </Paper>
                                  }
                                </Box>
                              </Grid>
                            </Grid>
                          </Grid>
                        </Grid>
                      </Paper>
                    </Grid>
                  );
                })}
              </Grid>
            ) : (
              <Box sx={{ textAlign: 'center', py: 4, color: 'text.secondary' }}>
                <Typography variant="h6">
                  No files in this category
                </Typography>
              </Box>
            )}
          </Box>
        </DialogContent>
      </Dialog>

      {/* Image Lightbox Dialog */}
      <Dialog open={!!imagePreviewUrl} onClose={() => setImagePreviewUrl(null)} maxWidth="md">
        <Box sx={{ p: 2, display: 'flex', flexDirection: 'column', alignItems: 'center', bgcolor: '#222' }}>
          <IconButton onClick={() => setImagePreviewUrl(null)} sx={{ alignSelf: 'flex-end', color: 'white' }}>
            <Close />
          </IconButton>
          {imagePreviewUrl && (
            <Box component="img" src={imagePreviewUrl} alt="Preview" sx={{ maxWidth: '80vw', maxHeight: '70vh', borderRadius: 2, boxShadow: 6, mt: 1 }} />
          )}
        </Box>
      </Dialog>
      </Container>
    </Box>
  );
};

export default ResultsPage; 
