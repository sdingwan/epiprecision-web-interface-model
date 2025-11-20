import React, { useState } from 'react';
import {
  Box,
  Typography,
  TextField,
  Button,
  Alert,
  Divider,
  Link,
  InputAdornment,
  IconButton,
  FormControlLabel,
  Checkbox
} from '@mui/material';
import { Visibility, VisibilityOff, Email, Lock, Person } from '@mui/icons-material';

const AuthPanel = ({
  onAuthSuccess,
  title = 'EpiPrecision',
  subtitle,
  showTitle = true,
  sx = {}
}) => {
  const [isSignUp, setIsSignUp] = useState(false);
  const [formData, setFormData] = useState({
    firstName: '',
    lastName: '',
    email: '',
    password: '',
    confirmPassword: '',
    institution: '',
    rememberMe: false
  });
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [errors, setErrors] = useState({});
  const [loading, setLoading] = useState(false);
  const [successMessage, setSuccessMessage] = useState('');

  const resolvedSubtitle = subtitle || (isSignUp ? 'Create your account' : 'Sign in to your account');

  const getStoredUsers = () => {
    const users = localStorage.getItem('registeredUsers');
    return users ? JSON.parse(users) : [];
  };

  const saveUser = (userData) => {
    const users = getStoredUsers();
    const newUser = {
      id: Date.now().toString(),
      firstName: userData.firstName,
      lastName: userData.lastName,
      email: userData.email.toLowerCase(),
      password: userData.password,
      institution: userData.institution,
      createdAt: new Date().toISOString()
    };
    users.push(newUser);
    localStorage.setItem('registeredUsers', JSON.stringify(users));
    return newUser;
  };

  const validateCredentials = (email, password) => {
    const users = getStoredUsers();
    return users.find(
      (user) => user.email === email.toLowerCase() && user.password === password
    );
  };

  const userExists = (email) => {
    const users = getStoredUsers();
    return users.some((user) => user.email === email.toLowerCase());
  };

  const handleInputChange = (e) => {
    const { name, value, checked } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: name === 'rememberMe' ? checked : value
    }));
    if (errors[name]) {
      setErrors((prev) => ({ ...prev, [name]: '' }));
    }
  };

  const validateForm = () => {
    const newErrors = {};
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

    if (!formData.email) {
      newErrors.email = 'Email is required';
    } else if (!emailRegex.test(formData.email)) {
      newErrors.email = 'Please enter a valid email address';
    }

    if (!formData.password) {
      newErrors.password = 'Password is required';
    } else if (formData.password.length < 6) {
      newErrors.password = 'Password must be at least 6 characters';
    }

    if (isSignUp) {
      if (!formData.firstName) {
        newErrors.firstName = 'First name is required';
      }
      if (!formData.lastName) {
        newErrors.lastName = 'Last name is required';
      }
      if (!formData.institution) {
        newErrors.institution = 'Institution is required';
      }
      if (!formData.confirmPassword) {
        newErrors.confirmPassword = 'Please confirm your password';
      } else if (formData.password !== formData.confirmPassword) {
        newErrors.confirmPassword = 'Passwords do not match';
      }
      if (formData.email && userExists(formData.email)) {
        newErrors.email = 'An account with this email already exists';
      }
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!validateForm()) {
      return;
    }

    setLoading(true);
    setSuccessMessage('');
    setErrors({});

    try {
      await new Promise((resolve) => setTimeout(resolve, 1000));

      if (isSignUp) {
        saveUser(formData);
        setSuccessMessage('Account created successfully! Please log in with your credentials.');
        setIsSignUp(false);
        setFormData({
          firstName: '',
          lastName: '',
          email: formData.email,
          password: '',
          confirmPassword: '',
          institution: '',
          rememberMe: false
        });
      } else {
        const user = validateCredentials(formData.email, formData.password);

        if (user) {
          localStorage.setItem('userLoggedIn', 'true');
          localStorage.setItem('userEmail', user.email);
          localStorage.setItem('userName', user.firstName);
          localStorage.setItem('userLastName', user.lastName);
          localStorage.setItem('userInstitution', user.institution);

          window.dispatchEvent(new Event('loginStateChanged'));

          if (onAuthSuccess) {
            onAuthSuccess(user);
          }
        } else {
          setErrors({
            general: 'Invalid email or password. Please check your credentials and try again.'
          });
        }
      }
    } catch (error) {
      setErrors({ general: 'An error occurred. Please try again.' });
    } finally {
      setLoading(false);
    }
  };

  const toggleMode = () => {
    setIsSignUp((prev) => !prev);
    setErrors({});
    setSuccessMessage('');
    setFormData({
      firstName: '',
      lastName: '',
      email: '',
      password: '',
      confirmPassword: '',
      institution: '',
      rememberMe: false
    });
  };

  return (
    <Box sx={{ width: '100%', ...sx }}>
      {showTitle && (
        <>
          <Typography variant="h4" gutterBottom align="center" sx={{ mb: 1 }}>
            {title}
          </Typography>
          <Typography variant="subtitle1" align="center" color="text.secondary" sx={{ mb: 4 }}>
            {resolvedSubtitle}
          </Typography>
        </>
      )}

      {successMessage && (
        <Alert severity="success" sx={{ mb: 3 }}>
          {successMessage}
        </Alert>
      )}

      {errors.general && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {errors.general}
        </Alert>
      )}

      <form onSubmit={handleSubmit}>
        {isSignUp && (
          <>
            <Box sx={{ display: 'flex', gap: 1.5, mb: 2.5, flexDirection: { xs: 'column', sm: 'row' } }}>
              <TextField
                fullWidth
                size="small"
                label="First Name"
                name="firstName"
                value={formData.firstName}
                onChange={handleInputChange}
                error={!!errors.firstName}
                helperText={errors.firstName}
                InputProps={{
                  startAdornment: (
                <InputAdornment position="start" sx={{ '& svg': { fontSize: '1rem' } }}>
                  <Person fontSize="inherit" />
                </InputAdornment>
                  )
                }}
              />
              <TextField
                fullWidth
                size="small"
                label="Last Name"
                name="lastName"
                value={formData.lastName}
                onChange={handleInputChange}
                error={!!errors.lastName}
                helperText={errors.lastName}
              />
            </Box>

            <TextField
              fullWidth
              size="small"
              label="Institution/Hospital"
              name="institution"
              value={formData.institution}
              onChange={handleInputChange}
              error={!!errors.institution}
              helperText={errors.institution}
              sx={{ mb: 2.5 }}
            />
          </>
        )}

        <TextField
          fullWidth
          size="small"
          label="Email Address"
          name="email"
          type="email"
          value={formData.email}
          onChange={handleInputChange}
          error={!!errors.email}
          helperText={errors.email}
          sx={{ mb: 2.5 }}
          InputProps={{
            startAdornment: (
              <InputAdornment position="start" sx={{ '& svg': { fontSize: '1rem' } }}>
                <Email fontSize="inherit" />
              </InputAdornment>
            )
          }}
        />

        <TextField
          fullWidth
          size="small"
          label="Password"
          name="password"
          type={showPassword ? 'text' : 'password'}
          value={formData.password}
          onChange={handleInputChange}
          error={!!errors.password}
          helperText={errors.password}
          sx={{ mb: 2.5 }}
          InputProps={{
            startAdornment: (
              <InputAdornment position="start" sx={{ '& svg': { fontSize: '1rem' } }}>
                <Lock fontSize="inherit" />
              </InputAdornment>
            ),
            endAdornment: (
              <InputAdornment position="end">
                <IconButton onClick={() => setShowPassword((prev) => !prev)} edge="end" size="small">
                  {showPassword ? <VisibilityOff fontSize="small" /> : <Visibility fontSize="small" />}
                </IconButton>
              </InputAdornment>
            )
          }}
        />

        {isSignUp && (
          <TextField
            fullWidth
            size="small"
            label="Confirm Password"
            name="confirmPassword"
            type={showConfirmPassword ? 'text' : 'password'}
            value={formData.confirmPassword}
            onChange={handleInputChange}
            error={!!errors.confirmPassword}
            helperText={errors.confirmPassword}
            sx={{ mb: 2.5 }}
            InputProps={{
              startAdornment: (
                <InputAdornment position="start" sx={{ '& svg': { fontSize: '1rem' } }}>
                  <Lock fontSize="inherit" />
                </InputAdornment>
              ),
              endAdornment: (
                <InputAdornment position="end">
                  <IconButton onClick={() => setShowConfirmPassword((prev) => !prev)} edge="end" size="small">
                    {showConfirmPassword ? <VisibilityOff fontSize="small" /> : <Visibility fontSize="small" />}
                  </IconButton>
                </InputAdornment>
              )
            }}
          />
        )}

        {!isSignUp && (
          <FormControlLabel
            control={
              <Checkbox name="rememberMe" checked={formData.rememberMe} onChange={handleInputChange} />
            }
            label="Remember me"
            sx={{ mt: -0.5, mb: 1.5 }}
          />
        )}

        <Button type="submit" fullWidth variant="contained" size="large" disabled={loading} sx={{ mb: 1.2, py: 1.1 }}>
          {loading ? 'Processing...' : isSignUp ? 'Create Account' : 'Sign In'}
        </Button>

        <Divider sx={{ mb: 1.2 }} />

        <Box textAlign="center">
          <Typography variant="body2">
            {isSignUp ? 'Already have an account?' : "Don't have an account?"}{' '}
            <Link component="button" type="button" onClick={toggleMode} sx={{ cursor: 'pointer' }}>
              {isSignUp ? 'Sign In' : 'Create Account'}
            </Link>
          </Typography>
        </Box>
      </form>
    </Box>
  );
};

export default AuthPanel;
