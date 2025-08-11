import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import "@testing-library/jest-dom";
import Register from "./Register";
import { describe, it, expect, beforeEach, vi } from "vitest";
import { register } from "../services/RegisterService";
import React from 'react';

vi.mock("../services/RegisterService");

describe("Register Component", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    window.location = { href: '' };
  });

  it('renders register form with all fields', () => {
    render(<Register />);
    expect(screen.getByLabelText("Username")).toBeInTheDocument();
    expect(screen.getByLabelText("Password")).toBeInTheDocument();
    expect(screen.getByLabelText("Email")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /sign up/i })).toBeInTheDocument();
  });

  it('shows validation error for empty fields', async () => {
    render(<Register />);
    fireEvent.click(screen.getByRole("button", { name: /sign up/i }));
    expect(await screen.findByText(/Please fill in all fields/i)).toBeInTheDocument();
  });

  it('validates email format', async () => {
    render(<Register />);
    
    fireEvent.change(screen.getByLabelText("Username"), { target: { value: "test" } });
    fireEvent.change(screen.getByLabelText("Password"), { target: { value: "pass" } });
    fireEvent.change(screen.getByLabelText("Email"), { target: { value: "invalid" } });
    
    // Click the button (now always enabled)
    fireEvent.click(screen.getByRole("button", { name: /sign up/i }));
    
    // Wait for the specific email validation error
    const errorMessage = await screen.findByText(/Please enter a valid email address/i);
    expect(errorMessage).toBeInTheDocument();
  });

  it('redirects on successful registration', async () => {
    register.mockResolvedValue({ success: true });
    render(<Register />);
    
    fireEvent.change(screen.getByLabelText("Username"), { target: { value: "test" } });
    fireEvent.change(screen.getByLabelText("Password"), { target: { value: "pass" } });
    fireEvent.change(screen.getByLabelText("Email"), { target: { value: "test@example.com" } });
    
    fireEvent.click(screen.getByRole("button", { name: /sign up/i }));
    
    await waitFor(() => expect(window.location.href).toBe("/login"));
  });
});