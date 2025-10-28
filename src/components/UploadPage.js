import React, { useState } from 'react';
import { 
  Card, 
  CardContent, 
  Typography, 
  Button, 
  List, 
  ListItem, 
  Box, 
  LinearProgress, 
  Alert, 
  Chip,
  Grid,
  Paper,
  Divider
} from '@mui/material';
import { useNavigate, useLocation } from 'react-router-dom';
import { 
  CloudUpload, 
  Folder, 
  Delete,
  Description,
  Image as ImageIcon,
  Info,
  PsychologyAlt
} from '@mui/icons-material';
import { useFiles } from '../App';

const EEG_FILE_TYPES = '.edf,.csv,.mat,.txt';
const MRI_FILE_TYPES = 'image/*,.nii,.nii.gz,.dcm';
const PET_FILE_TYPES = 'image/*,.dcm,.nii,.nii.gz';

const UploadPage = () => {
  const [files, setFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const navigate = useNavigate();
  const location = useLocation();
  const { uploadedFiles, setUploadedFiles, clearFiles } = useFiles();
  
  // Get dataType from navigation state, default to MRI
  const dataType = location.state?.dataType || 'MRI';

  // Log existing uploaded files for debugging
  console.log('UploadPage - existing uploadedFiles:', uploadedFiles);
  console.log('UploadPage - dataType from navigation:', dataType);
 
  const processFiles = (selectedFiles) => {
    // Filter out system files and hidden files
    const validFiles = selectedFiles.filter(file => {
      const fileName = file.name;
      // Exclude common system files
      const systemFiles = [
        '.DS_Store',           // macOS
        'Thumbs.db',           // Windows
        '.DS_Store?',          // macOS variants
        '._.DS_Store',         // macOS metadata
        '.Spotlight-V100',     // macOS Spotlight
        '.Trashes',            // macOS Trash
        'ehthumbs.db',         // Windows thumbnail cache
        'Desktop.ini'          // Windows folder settings
      ];
      
      return !systemFiles.some(sysFile => fileName.includes(sysFile)) && 
             !fileName.startsWith('.') && 
             !fileName.startsWith('_');
    });

    const processedFiles = validFiles.map(file => ({
      id: `${file.name}-${file.size}-${Date.now()}`,
      name: file.name,
      size: file.size,
      type: file.type,
      lastModified: file.lastModified,
      blobUrl: URL.createObjectURL(file),
      originalFile: file,
      relativePath: file.webkitRelativePath || file.name,
      dataType: dataType, // Use current dataType
    }));
    return processedFiles;
  };

  const handleFileChange = (e) => {
    const selectedFiles = Array.from(e.target.files);
    setFiles(selectedFiles);
    // Process files and store in context
    const processedFiles = processFiles(selectedFiles);
    console.log('UploadPage - setting uploadedFiles:', processedFiles);
    setUploadedFiles(processedFiles);
    // Simulate upload progress for demo
    if (selectedFiles.length > 0) {
      setUploading(true);
      setUploadProgress(0);
      const interval = setInterval(() => {
        setUploadProgress(prev => {
          if (prev >= 100) {
            clearInterval(interval);
            setUploading(false);
            return 100;
          }
          return prev + 10;
        });
      }, 200);
    }
  };

  const handleFolderUpload = (e) => {
    const selectedFiles = Array.from(e.target.files);
    setFiles(selectedFiles);
    // Process files and store in context
    const processedFiles = processFiles(selectedFiles);
    console.log('UploadPage - setting uploadedFiles (folder):', processedFiles);
    setUploadedFiles(processedFiles);
    if (selectedFiles.length > 0) {
      setUploading(true);
      setUploadProgress(0);
      const interval = setInterval(() => {
        setUploadProgress(prev => {
          if (prev >= 100) {
            clearInterval(interval);
            setUploading(false);
            return 100;
          }
          return prev + 5;
        });
      }, 100);
    }
  };

  const getFileCountMessage = (fileList) => {
    if (fileList.length === 0) return `No new files selected`;
    if (fileList.length === 1) return `1 new file selected`;
    return `${fileList.length} new files selected`;
  };

  const getTotalFileSize = (fileList) => {
    const totalBytes = fileList.reduce((sum, file) => sum + (file.size || 0), 0);
    const totalMB = totalBytes / (1024 * 1024);
    return totalMB.toFixed(2);
  };

  const getFileIcon = (file) => {
    if (file.dataType === 'EEG') {
      return <PsychologyAlt sx={{ color: 'secondary.main' }} />;
    }
    if (file.type && file.type.startsWith('image/')) {
      return <ImageIcon sx={{ color: '#e0e0e0' }} />;
    }
    return <Description sx={{ color: 'text.secondary' }} />;
  };

  const getFileTypes = () => {
    switch (dataType) {
      case 'EEG':
        return EEG_FILE_TYPES;
      case 'PET':
        return PET_FILE_TYPES;
      default:
        return MRI_FILE_TYPES;
    }
  };

  const getDataTypeDescription = () => {
    switch (dataType) {
      case 'EEG':
        return 'Choose your EEG files. Supported formats: EDF, CSV, MAT, TXT.';
      case 'PET':
        return 'Choose your PET images or DICOM files. Supported formats: JPEG, PNG, NIfTI (.nii, .nii.gz), DICOM (.dcm).';
      default:
        return 'Choose your MRI images or DICOM files. Supported formats: JPEG, PNG, NIfTI (.nii, .nii.gz), DICOM (.dcm).';
    }
  };

  const getSupportedFormats = () => {
    switch (dataType) {
      case 'EEG':
        return [
          '• EDF (European Data Format)',
          '• CSV, MAT, TXT files'
        ];
      case 'PET':
        return [
          '• JPEG, PNG, TIFF images',
          '• NIfTI files (.nii, .nii.gz)',
          '• DICOM files (.dcm)'
        ];
      default:
        return [
          '• JPEG, PNG, TIFF images',
          '• NIfTI files (.nii, .nii.gz)',
          '• DICOM files (.dcm)'
        ];
    }
  };

  const getRecommendedSize = () => {
    switch (dataType) {
      case 'EEG':
        return 'Typical EEG files are 1-100 MB';
      case 'PET':
        return '50-150 images for optimal results';
      default:
        return '100-200 images for optimal results';
    }
  };

  // Add or update clinical note for a file
  const handleNoteChange = (fileId, note) => {
    setUploadedFiles(prevFiles => prevFiles.map(f => f.id === fileId ? { ...f, clinicalNote: note } : f));
  };

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

  return (
    <Box sx={{ minHeight: '80vh', p: 3 }}>
      <Grid container maxWidth="lg" spacing={4} sx={{ mx: 'auto' }}>
        {/* Header Section */}
        <Grid item xs={12}>
          <Card sx={{ mb: 3, background: '#1a1a1a', color: '#e0e0e0', border: '1px solid #333333' }}>
            <CardContent sx={{ p: 4 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
                <Typography variant="h4" sx={{ fontWeight: 700, color: 'white' }}>
                  Upload {dataType} Data
                </Typography>
              </Box>
              <Typography variant="body1" sx={{ color: 'rgba(255, 255, 255, 0.95)', mb: 2 }}>
                Upload your {dataType} data for advanced analysis
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, flexWrap: 'wrap' }}>
                <Chip 
                  icon={<Info />}
                  label={`${dataType} Formats Supported`} 
                  sx={{ bgcolor: 'rgba(255, 255, 255, 0.2)', color: 'white', fontWeight: 600 }}
                />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Main Upload Section */}
        <Grid item xs={12} md={9}>
          <Card sx={{ height: 'fit-content' }}>
            <CardContent sx={{ p: 4 }}>
              {/* Existing Files Display */}
              {uploadedFiles.length > 0 && (
                <Alert 
                  severity="success" 
                  sx={{ mb: 3 }}
                  action={
                    <Button 
                      color="inherit" 
                      size="small" 
                      onClick={() => { clearFiles(); setFiles([]); }}
                      startIcon={<Delete />}
                    >
                      Clear All
                    </Button>
                  }
                >
                  <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                    {uploadedFiles.length} file{uploadedFiles.length !== 1 ? 's' : ''} ready for processing
                  </Typography>
                  <Typography variant="body2">
                    Total size: {getTotalFileSize(uploadedFiles)} MB
                  </Typography>
                  {/* Clinical context input for each file */}
                  <List dense sx={{ mt: 2, width: '100%' }}>
                    {sortFilesByICNumber(uploadedFiles).map((file, idx) => (
                      <ListItem key={file.id} alignItems="flex-start" sx={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start', mb: 2, width: '100%' }}>
                        <Box sx={{ display: 'flex', alignItems: 'center', width: '100%', gap: 2, mb: 1 }}>
                          {getFileIcon(file)}
                          <Box sx={{ flexGrow: 1 }}>
                            <Typography variant="body2" sx={{ fontWeight: 500 }}>
                              {file.name}
                            </Typography>
                            <Typography variant="caption" color="text.secondary">
                              {(file.size / 1024 / 1024).toFixed(2)} MB • {file.type || 'Unknown type'}
                            </Typography>
                          </Box>
                        </Box>
                        <Box sx={{ width: '100%', mt: 1, minWidth: '100%' }}>
                          <input
                            type="text"
                            placeholder="Add clinical context (optional)"
                            value={file.clinicalNote || ''}
                            onChange={e => handleNoteChange(file.id, e.target.value)}
                            style={{
                              width: '350px',
                              maxWidth: '100%',
                              padding: '12px 16px',
                              borderRadius: '8px',
                              border: '2px solid #444444',
                              fontSize: '1rem',
                              marginTop: 4,
                              backgroundColor: '#1a1a1a',
                              color: '#e0e0e0',
                              outline: 'none',
                              minHeight: '48px',
                              boxSizing: 'border-box',
                              transition: 'all 0.2s ease-in-out',
                            }}
                            onFocus={e => {
                              e.target.style.border = '2px solid #00ffff';
                              e.target.style.boxShadow = '0 0 8px 2px #00ffff44';
                            }}
                            onBlur={e => {
                              e.target.style.border = '2px solid #444444';
                              e.target.style.boxShadow = 'none';
                            }}
                          />
                        </Box>
                      </ListItem>
                    ))}
                  </List>
                </Alert>
              )}

              {/* Upload Instructions */}
              <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
                Select {dataType} Data
              </Typography>
              
              <Typography variant="body2" color="text.secondary" sx={{ mb: 4, lineHeight: 1.6 }}>
                {getDataTypeDescription()}
              </Typography>

              {/* Upload Buttons */}
              <Grid container spacing={2} sx={{ mb: 4 }}>
                <Grid item xs={12} sm={6}>
                  <Button
                    variant="outlined"
                    component="label"
                    startIcon={<CloudUpload />}
                    size="large"
                    fullWidth
                    sx={{
                      py: 2,
                      borderRadius: 2,
                      backgroundColor: '#1a1a1a',
                      color: '#e0e0e0',
                      border: files.length > 0 ? '2px solid #00ffff' : '2px solid #333333',
                      boxShadow: 'none',
                      '&:hover': {
                        backgroundColor: '#222',
                        borderColor: '#00ffff',
                        boxShadow: '0 0 12px 3px #00ffff88',
                      },
                      '&:focus': {
                        borderColor: '#00ffff',
                        boxShadow: '0 0 0 2px #00ffff44',
                      },
                      '&.Mui-disabled': {
                        color: '#e0e0e0',
                        border: '2px solid #333333',
                        backgroundColor: '#222',
                        opacity: 0.5,
                      },
                    }}
                  >
                    Select Individual Files
                    <input
                      type="file"
                      hidden
                      multiple
                      accept={getFileTypes()}
                      onChange={handleFileChange}
                    />
                  </Button>
                </Grid>
                <Grid item xs={12} sm={6}>
                  <Button
                    variant="outlined"
                    component="label"
                    startIcon={<Folder />}
                    size="large"
                    fullWidth
                    sx={{
                      py: 2,
                      borderRadius: 2,
                      backgroundColor: '#1a1a1a',
                      color: '#e0e0e0',
                      border: files.length > 0 ? '2px solid #00ffff' : '2px solid #333333',
                      boxShadow: 'none',
                      '&:hover': {
                        backgroundColor: '#222',
                        borderColor: '#00ffff',
                        boxShadow: '0 0 12px 3px #00ffff88',
                      },
                      '&:focus': {
                        borderColor: '#00ffff',
                        boxShadow: '0 0 0 2px #00ffff44',
                      },
                      '&.Mui-disabled': {
                        color: '#e0e0e0',
                        border: '2px solid #333333',
                        backgroundColor: '#222',
                        opacity: 0.5,
                      },
                    }}
                  >
                    Select Folder
                    <input
                      type="file"
                      hidden
                      webkitdirectory=""
                      multiple
                      accept={getFileTypes()}
                      onChange={handleFolderUpload}
                    />
                  </Button>
                </Grid>
              </Grid>

              {/* File Count Display */}
              {uploadedFiles.length > 0 && (
                <Box sx={{ mb: 3, mt: 2 }}>
                  <Chip 
                    label={getFileCountMessage(uploadedFiles)} 
                    sx={{
                      fontSize: '0.9rem',
                      fontWeight: 600,
                      backgroundColor: '#1a1a1a',
                      color: '#00ffff',
                      border: '2px solid #00ffff',
                      boxShadow: '0 0 8px 2px #00ffff55',
                    }}
                    size="medium"
                  />
                </Box>
              )}

              {/* Upload Progress */}
              {uploading && (
                <Paper sx={{ p: 3, mb: 3, bgcolor: '#1a1a1a' }}>
                  <Typography variant="subtitle2" sx={{ mb: 2, fontWeight: 600 }}>
                    Uploading {uploadedFiles.length} {uploadedFiles.length === 1 ? 'file' : 'files'}...
                  </Typography>
                  <LinearProgress 
                    variant="determinate" 
                    value={uploadProgress} 
                    sx={{ height: 8, borderRadius: 1, mb: 1, backgroundColor: '#222', '& .MuiLinearProgress-bar': { backgroundColor: '#00ffff' } }}
                  />
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Typography variant="body2" color="text.secondary">
                      {uploadProgress}% complete
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {getTotalFileSize(uploadedFiles)} MB
                    </Typography>
                  </Box>
                </Paper>
              )}

              {/* File List (show first 10 and summary) */}
              {uploadedFiles.length > 0 && !uploading && (
                <Paper sx={{ p: 3, mb: 3, bgcolor: '#1a1a1a', border: '1px solid #333333' }}>
                  <Typography variant="subtitle2" sx={{ mb: 2, fontWeight: 600, color: '#00ffff' }}>
                    New Files Ready ({uploadedFiles.length} total - {getTotalFileSize(uploadedFiles)} MB)
                  </Typography>
                  <Box sx={{ 
                    maxHeight: 200, 
                    overflow: 'auto', 
                    border: '1px solid #333333', 
                    borderRadius: 1,
                    bgcolor: '#222'
                  }}>
                    <List dense>
                      {sortFilesByICNumber(uploadedFiles).slice(0, 10).map((file, idx) => (
                        <ListItem key={idx} sx={{ py: 1, borderBottom: '1px solid #f5f5f5' }}>
                          <Box sx={{ display: 'flex', alignItems: 'center', width: '100%', gap: 2 }}>
                            {getFileIcon({ ...file, dataType })}
                            <Box sx={{ flexGrow: 1 }}>
                              <Typography variant="body2" sx={{ fontWeight: 500 }}>
                                {file.name}
                              </Typography>
                              <Typography variant="caption" color="text.secondary">
                                {(file.size / 1024 / 1024).toFixed(2)} MB • {file.type || 'Unknown type'}
                              </Typography>
                            </Box>
                          </Box>
                        </ListItem>
                      ))}
                      {uploadedFiles.length > 10 && (
                        <ListItem sx={{ py: 1, bgcolor: '#222' }}>
                          <Typography variant="body2" color="text.secondary" sx={{ fontStyle: 'italic' }}>
                            ... and {uploadedFiles.length - 10} more files
                          </Typography>
                        </ListItem>
                      )}
                    </List>
                  </Box>
                </Paper>
              )}

              {/* Proceed Button */}
              <Button
                variant="outlined"
                size="large"
                fullWidth
                onClick={() => navigate('/processing')}
                disabled={(files.length === 0 && uploadedFiles.length === 0) || uploading}
                sx={{
                  py: 1.5,
                  fontSize: '1rem',
                  backgroundColor: '#1a1a1a',
                  color: '#e0e0e0',
                  border: (files.length > 0 || uploadedFiles.length > 0) && !uploading ? '2px solid #00ffff' : '2px solid #333333',
                  boxShadow: 'none',
                  fontWeight: 600,
                  borderRadius: 2,
                  '&:hover': { backgroundColor: '#222', borderColor: '#00ffff', boxShadow: '0 0 12px 3px #00ffff88' },
                  '&.Mui-disabled': {
                    color: '#e0e0e0',
                    border: '2px solid #333333',
                    backgroundColor: '#222',
                    opacity: 0.5,
                  },
                }}
              >
                {uploading 
                  ? 'Uploading...' 
                  : (files.length === 0 && uploadedFiles.length === 0)
                    ? 'Select Files to Continue'
                    : uploadedFiles.length > 0 && files.length === 0
                      ? `Begin Analysis with ${uploadedFiles.length} ${uploadedFiles.length === 1 ? `${dataType} File` : `${dataType} Files`}`
                      : files.length === 1
                        ? `Begin Analysis with 1 New ${dataType} File`
                        : `Begin Analysis with ${files.length} New ${dataType} Files`
                }
              </Button>
            </CardContent>
          </Card>
        </Grid>

        {/* Information Panel */}
        <Grid item xs={12} md={3}>
          <Card sx={{ height: 'fit-content', mb: 3 }}>
            <CardContent sx={{ p: 3 }}>
              <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
                Upload Guidelines
              </Typography>
              
              <Box sx={{ mb: 3 }}>
                <Typography variant="subtitle2" sx={{ mb: 1, fontWeight: 600 }}>
                  Supported Formats:
                </Typography>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
                  {getSupportedFormats().map((format, index) => (
                    <Typography key={index} variant="body2" color="text.secondary">
                      {format}
                    </Typography>
                  ))}
                </Box>
              </Box>

              <Divider sx={{ my: 2 }} />

              <Box sx={{ mb: 3 }}>
                <Typography variant="subtitle2" sx={{ mb: 1, fontWeight: 600 }}>
                  Recommended Size:
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {getRecommendedSize()}
                </Typography>
              </Box>

            </CardContent>
          </Card>

          {/* Quick Stats */}
          {uploadedFiles.length > 0 && (
            <Card>
              <CardContent sx={{ p: 3 }}>
                <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
                  Upload Summary
                </Typography>
                
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                  <Typography variant="body2" color="text.secondary">
                    Total Files:
                  </Typography>
                  <Typography variant="body2" sx={{ fontWeight: 600 }}>
                    {uploadedFiles.length}
                  </Typography>
                </Box>
                
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                  <Typography variant="body2" color="text.secondary">
                    Total Size:
                  </Typography>
                  <Typography variant="body2" sx={{ fontWeight: 600 }}>
                    {getTotalFileSize(uploadedFiles)} MB
                  </Typography>
                </Box>
                
                <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                  <Typography variant="body2" color="text.secondary">
                    Status:
                  </Typography>
                  <Chip 
                    label={uploading ? "Uploading" : "Ready"} 
                    size="small" 
                    sx={{
                      backgroundColor: '#1a1a1a',
                      color: '#00ffff',
                      border: '2px solid #00ffff',
                      boxShadow: '0 0 8px 2px #00ffff55',
                      fontWeight: 600,
                    }}
                  />
                </Box>
              </CardContent>
            </Card>
          )}
        </Grid>
      </Grid>
    </Box>
  );
};

export default UploadPage; 
