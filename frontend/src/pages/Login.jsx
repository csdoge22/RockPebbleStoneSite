import { useState } from 'react';
import { login } from '../services/LoginService';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../auth/AuthProvider';
import React from 'react';

const Login = () => {
  const [form, setForm] = useState({ username: '', password: '' });
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setForm(prev => ({ ...prev, [name]: value }));
    // Clear error when user starts typing
    if (error) setError('');
  };

  const validateForm = () => {
    if (!form.username.trim() || !form.password.trim()) {
      setError('Please fill in all fields');
      return false;
    }
    return true;
  };

  const handleSubmit = async (e) => {
  e.preventDefault();
  setError('');
  if (!validateForm()) return;
  
  setIsLoading(true);
  
  try {
    const response = await login(form);
    console.log('Login response:', response); // Debug log
    
    if (!response) {
      setError('No response from server. Please try again.');
      return;
    }

    if (response.success) {
      if (!response.token) {
        console.error('Login successful but no token received');
        setError('Authentication error. Please try again.');
        return;
      }
      await modifyToken(response.token);
      navigate('/board');
    } else {
      setError(response.error || 'Login failed');
    }
  } catch (err) {
    console.error('Login error:', err);
    setError('An unexpected error occurred. Please try again.');
  } finally {
    setIsLoading(false);
  }
};

  // REMOVED isFormValid - always enable the button to provide user feedback
  // const isFormValid = form.username.trim() && form.password.trim();

  return (
    <div className="flex items-center justify-center min-h-screen">
      <form 
        className="bg-white border-4 border-blue-300 rounded-md p-8 w-full max-w-md shadow-lg"
        onSubmit={handleSubmit}
        noValidate
      >
        <h1 className="text-3xl font-bold text-center mb-8 text-gray-800">WELCOME BACK</h1>
        
        <div className="mb-6">
          <label htmlFor="username" className="block text-lg font-medium text-gray-700 mb-2">
            Username
          </label>
          <input
            type="text"
            id="username"
            name="username"
            value={form.username}
            onChange={handleChange}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            placeholder="Enter your username"
            autoComplete="username"
          />
        </div>
        
        <div className="mb-6">
          <label htmlFor="password" className="block text-lg font-medium text-gray-700 mb-2">
            Password
          </label>
          <input
            type="password"
            id="password"
            name="password"
            value={form.password}
            onChange={handleChange}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            placeholder="Enter your password"
            autoComplete="current-password"
          />
        </div>
        
        <div className="mb-6 text-center">
          <span className="text-gray-600">Don't have an account yet? </span>
          <a href="/register" className="text-blue-600 hover:text-blue-800 font-medium">
            Register
          </a>
        </div>
        
        <button 
          type="submit" 
          className={`w-full bg-green-900 text-white py-3 rounded-md hover:bg-green-800 transition-colors duration-200 font-medium
            ${isLoading ? 'opacity-75 cursor-not-allowed' : 'hover:shadow-lg'}`}
          // REMOVED: Always enable the button to provide validation feedback
          // disabled={!isFormValid || isLoading}
        >
          {isLoading ? (
            <span className="flex items-center justify-center">
              <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              Logging in...
            </span>
          ) : 'Login'}
        </button>
        
        {error && (
          <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-md">
            <p className="text-red-600 text-sm text-center font-medium">{error}</p>
          </div>
        )}
      </form>
    </div>
  );
};

export default Login;