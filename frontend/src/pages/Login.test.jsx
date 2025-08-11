import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import Login from './Login';
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { login } from '../services/LoginService';
import React from 'react';

vi.mock('../services/LoginService');

describe('Login Component', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    window.location = { href: '' };
  });

  it('renders login form with all fields', () => {
    render(<Login />);
    expect(screen.getByLabelText('Username')).toBeInTheDocument();
    expect(screen.getByLabelText('Password')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /login/i })).toBeInTheDocument();
  });

  it('shows validation error for empty fields', async () => {
    render(<Login />);
    fireEvent.click(screen.getByRole('button', { name: /login/i }));
    expect(await screen.findByText(/Please fill in all fields/i)).toBeInTheDocument();
  });

  it('redirects on successful login', async () => {
    login.mockResolvedValue({ success: true });
    render(<Login />);
    
    fireEvent.change(screen.getByLabelText('Username'), { target: { value: 'test' } });
    fireEvent.change(screen.getByLabelText('Password'), { target: { value: 'pass' } });
    fireEvent.click(screen.getByRole('button', { name: /login/i }));
    
    await waitFor(() => expect(window.location.href).toBe('/board'));
  });

  it('shows error for failed login', async () => {
    login.mockResolvedValue({ error: 'Invalid credentials' });
    render(<Login />);
    
    fireEvent.change(screen.getByLabelText('Username'), { target: { value: 'test' } });
    fireEvent.change(screen.getByLabelText('Password'), { target: { value: 'pass' } });
    fireEvent.click(screen.getByRole('button', { name: /login/i }));
    
    expect(await screen.findByText(/Invalid credentials/i)).toBeInTheDocument();
  });
});