import { useState } from 'react';
import { register } from '../services/RegisterService';
import React from 'react';

const Register = () => {
  const [form, setForm] = useState({ username: '', password: '', email: '' });
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setForm(prev => ({ ...prev, [name]: value }));
    // Clear error when user starts typing
    if (error) setError('');
  };

  const validateForm = () => {
    // Check required fields
    if (!form.username.trim() || !form.password.trim() || !form.email.trim()) {
      setError('Please fill in all fields');
      return false;
    }
    
    // Validate email format
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(form.email)) {
      setError('Please enter a valid email address');
      return false;
    }
    
    return true;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    // Reset previous state
    setError('');
    
    // Validate form
    if (!validateForm()) return;
    
    setIsLoading(true);
    
    try {
      const response = await register(form);
      
      if (!response) {
        setError('No response from server. Please try again.');
      } else if (response.success) {
        window.location.href = '/login';
      } else {
        setError(response.error || response.message || 'Registration failed');
      }
    } catch (err) {
      setError('An unexpected error occurred. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  // REMOVED isFormValid - always enable the button to provide user feedback
  // const isFormValid = form.username.trim() && form.password.trim() && 
  //                     form.email.trim() && /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(form.email);

  return (
    <div className="flex items-center justify-center min-h-screen">
      <form 
        className="bg-white border-4 border-blue-300 rounded-md p-8 w-full max-w-md shadow-lg"
        onSubmit={handleSubmit}
        noValidate
      >
        <h1 className="text-3xl font-bold text-center mb-8 text-gray-800">HELLO THERE</h1>
        
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
            placeholder="Choose a username"
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
            placeholder="Create a password"
            autoComplete="new-password"
          />
        </div>
        
        <div className="mb-6">
          <label htmlFor="email" className="block text-lg font-medium text-gray-700 mb-2">
            Email
          </label>
          <input 
            type="email" 
            id="email" 
            name="email" 
            value={form.email} 
            onChange={handleChange} 
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            placeholder="your@email.com"
            autoComplete="email"
          />
        </div>
        
        <div className="mb-6 text-center">
          <span className="text-gray-600">Already have an account? </span>
          <a href="/login" className="text-blue-600 hover:text-blue-800 font-medium">
            Login
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
              Registering...
            </span>
          ) : 'Sign Up'}
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

export default Register;